import os
import gc
import logging
import importlib
from enum import Enum
import traceback
from typing import Optional, Any, Dict, List, Tuple, Union, Type
import psutil
from collections import OrderedDict
import time

import torch
import diffusers

from diffusers import (
    DiffusionPipeline,
    FluxInpaintPipeline,
    FluxPipeline,

)
from diffusers.loaders import FromSingleFileMixin
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.utils import EntryNotFoundError

from ..config import get_config
from ..globals import (
    get_hf_model_manager,
    get_architectures,
    get_available_torch_device,
    get_model_downloader,
)
from .model_downloader import ModelSource
from ..utils.load_models import load_state_dict_from_file
from ..utils.quantize_models import quantize_model_fp8
from ..base_types.config import PipelineConfig
from .model_config_manager import ModelConfigManager

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Constants
VRAM_SAFETY_MARGIN_GB = 7.0
DEFAULT_MAX_VRAM_BUFFER_GB = 2.0
RAM_SAFETY_MARGIN_GB = 4.0
VRAM_THRESHOLD = 1.4


class GPUEnum(Enum):
    """GPU memory thresholds in GB for different optimization levels."""

    LOW = 7
    MEDIUM = 14
    HIGH = 22
    VERY_HIGH = 30



# Keys correspond to diffusers pipeline classes
MODEL_COMPONENTS = {
    "FluxPipeline": [
        "vae",
        "transformer",
        "text_encoder",
        "text_encoder_2",
        "scheduler",
        "tokenizer",
        "tokenizer_2",
    ],
    "StableDiffusionXLPipeline": [
        "vae",
        "unet",
        "text_encoder",
        "text_encoder_2",
        "scheduler",
        "tokenizer",
        "tokenizer_2",
    ],
    "StableDiffusionPipeline": [
        "vae",
        "unet",
        "text_encoder",
        "scheduler",
        "tokenizer",
    ],
}


def get_pipeline_class(
    class_name: Union[str, Tuple[str, str]],
) -> Tuple[Type[DiffusionPipeline], Optional[str]]:
    """Get the appropriate pipeline class based on class_name configuration.
    Learn about custom-pipelines here: https://huggingface.co/docs/diffusers/v0.6.0/en/using-diffusers/custom_pipelines
    Note that from_single_file does not support custom pipelines, only from_pretrained does.

    Args:
        class_name: Either a string naming a diffusers class, or a tuple of (package, class)

    Returns:
        Tuple of (Pipeline class to use, custom_pipeline to include as a kwarg to the pipeline)
    """
    # class_name is in the form of [package, class]
    if isinstance(class_name, tuple):
        # Load from custom package
        package, cls = class_name
        module = importlib.import_module(package)
        return (getattr(module, cls), None)

    # Try loading class_name as a diffusers class
    try:
        pipeline_class = getattr(importlib.import_module("diffusers"), class_name)
        if not issubclass(pipeline_class, DiffusionPipeline):
            print("custompipeline2", class_name)
            raise TypeError(f"{class_name} does not inherit from DiffusionPipeline")
        return (pipeline_class, None)

    except (ImportError, AttributeError):
        # Assume the class name is the name of a custom pipeline
        print("custompipeline1", class_name)
        return (DiffusionPipeline, class_name)


class LRUCache:
    """
    Least Recently Used (LRU) Cache for tracking model usage.

    Maintains separate tracking for GPU and CPU cached models using timestamps
    to determine usage patterns and inform memory management decisions.
    """

    def __init__(self):
        """Initialize empty GPU and CPU caches."""
        self.gpu_cache: OrderedDict = OrderedDict()  # model_id -> last_used_timestamp
        self.cpu_cache: OrderedDict = OrderedDict()  # model_id -> last_used_timestamp

    def access(self, model_id: str, cache_type: str = "gpu") -> None:
        """
        Record access of a model, updating its position in the LRU cache.

        Args:
            model_id: Unique identifier for the model
            cache_type: Type of cache to update ("gpu" or "cpu")
        """
        cache = self.gpu_cache if cache_type == "gpu" else self.cpu_cache
        cache.pop(model_id, None)  # Remove if exists
        cache[model_id] = time.time()  # Add to end (most recently used)

    def remove(self, model_id: str, cache_type: str = "gpu") -> None:
        """
        Remove a model from cache tracking.

        Args:
            model_id: Unique identifier for the model to remove
            cache_type: Type of cache to remove from ("gpu" or "cpu")
        """
        cache = self.gpu_cache if cache_type == "gpu" else self.cpu_cache
        cache.pop(model_id, None)

    def get_lru_models(
        self, cache_type: str = "gpu", count: Optional[int] = None
    ) -> List[str]:
        """
        Get least recently used models.

        Args:
            cache_type: Type of cache to query ("gpu" or "cpu")
            count: Optional number of models to return. If None, returns all models

        Returns:
            List of model IDs ordered by least recently used first
        """
        cache = self.gpu_cache if cache_type == "gpu" else self.cpu_cache
        model_list = list(cache.keys())
        return model_list if count is None else model_list[:count]


class ModelMemoryManager:
    """
    Manages loading, unloading and memory allocation of machine learning models.

    Handles dynamic movement of models between GPU and CPU memory based on
    available resources and usage patterns. Implements optimization strategies
    for efficient memory usage and model performance.
    """

    def __init__(self):
        """Initialize the model memory manager with empty states and default configurations."""
        # Model tracking
        self.current_model: Optional[str] = None
        self.loaded_models: Dict[str, DiffusionPipeline] = {}
        self.cpu_models: Dict[str, DiffusionPipeline] = {}
        self.model_sizes: Dict[str, float] = {}
        self.model_types: Dict[str, torch.dtype] = {}
        self.loaded_model: Optional[DiffusionPipeline] = None

        # Memory tracking
        self.vram_usage: float = 0
        self.ram_usage: float = 0
        self.max_vram: float = self._get_total_vram() - DEFAULT_MAX_VRAM_BUFFER_GB
        self.system_ram: float = psutil.virtual_memory().total / (1024**3)

        # State flags
        self.is_in_device: bool = False
        self.should_quantize: bool = False

        # Managers and caches
        self.model_downloader = get_model_downloader()
        self.hf_model_manager = get_hf_model_manager()
        self.cache_dir = HF_HUB_CACHE
        self.lru_cache = LRUCache()
        self.model_config_manager = ModelConfigManager()

    # def _setup_scheduler(
    #     self, pipeline: DiffusionPipeline, scheduler_name: str
    # ) -> None:
    #     """
    #     Sets up the scheduler for the pipeline based on the config name.

    #     Args:
    #         pipeline: The diffusion pipeline
    #         scheduler_name: Name of the scheduler from the config
    #     """
    #     if not scheduler_name or scheduler_name not in SCHEDULER_CONFIGS:
    #         logger.warning(
    #             f"Scheduler '{scheduler_name}' not found in configs. Using default scheduler."
    #         )
    #         return

    #     scheduler_config = SCHEDULER_CONFIGS[scheduler_name]
    #     try:
    #         # Create new scheduler instance with the pipeline's existing config
    #         new_scheduler = scheduler_config["class"].from_config(
    #             pipeline.scheduler.config, **scheduler_config["kwargs"]
    #         )
    #         pipeline.scheduler = new_scheduler
    #         logger.info(f"Successfully set scheduler to {scheduler_name}")
    #     except Exception as e:
    #         logger.error(f"Error setting scheduler {scheduler_name}: {e}")
    #         logger.info("Falling back to default scheduler")

    def _setup_scheduler(self, pipeline: DiffusionPipeline, model_id: str) -> None:
        """Setup scheduler from component config"""
        config = get_config()
        model_config = config.pipeline_defs.get(model_id)
        print(f"Model config: {model_config}")
        components = model_config.get("components", {})
        scheduler_config = components.get("scheduler", {}) if components else {}
        if not scheduler_config and scheduler_config == {}:
            return
            
        scheduler_class = scheduler_config.get("class_name")
        scheduler_kwargs = scheduler_config.get("kwargs", {})
        
        try:
            new_scheduler = getattr(diffusers, scheduler_class).from_config(
                pipeline.scheduler.config, **scheduler_kwargs
            )
            pipeline.scheduler = new_scheduler

            print(f"Successfully set scheduler to {scheduler_class}")
        except Exception as e:
            logger.error(f"Error setting scheduler for {model_id}: {e}")

    def _get_memory_info(self) -> Tuple[float, float]:
        """
        Get current memory availability.

        Returns:
            Tuple containing:
            - Available RAM in GB
            - Available VRAM in GB
        """
        ram = self._get_available_ram()
        vram = self._get_available_vram()
        return ram, vram

    def _get_available_ram(self) -> float:
        """
        Get available system RAM in GB.

        Returns:
            Available RAM in GB
        """
        return psutil.virtual_memory().available / (1024**3)

    def _get_total_vram(self) -> float:
        """
        Get total VRAM available on the system.

        Returns:
            Total VRAM in GB, or 0 if no CUDA device is available
        """
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0

    def _get_available_vram(self) -> float:
        """
        Get currently available VRAM.

        Returns:
            Available VRAM in GB
        """
        if torch.cuda.is_available():
            available_vram_gb = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            ) / (1024**3)
            logger.debug(f"Available VRAM: {available_vram_gb:.2f} GB")
            return available_vram_gb
        return 100

    def _can_load_to_ram(self, model_size: float) -> bool:
        """
        Check if a model can be loaded into RAM.

        Args:
            model_size: Size of the model in GB

        Returns:
            Boolean indicating if model can fit in RAM
        """
        print("In Available RAM")
        available_ram = self._get_available_ram() - RAM_SAFETY_MARGIN_GB
        logger.debug(
            f"Available RAM: {available_ram:.2f} GB, Model size: {model_size:.2f} GB"
        )
        return available_ram >= model_size

    def _can_load_model(self, model_size: float) -> Tuple[bool, bool]:
        """
        Check if a model can be loaded to GPU.

        Args:
            model_size: Size of the model in GB

        Returns:
            Tuple of (can_load_without_optimization, needs_optimization)
        """
        # First check if model is bigger than total VRAM
        if model_size > self.max_vram:
            logger.debug(
                f"Model size ({model_size:.2f} GB) exceeds total VRAM ({self.max_vram:.2f} GB)"
            )
            return True, True  # Can load but needs optimization

        # Model can theoretically fit in VRAM
        available_vram = self._get_available_vram()
        if model_size <= available_vram - VRAM_SAFETY_MARGIN_GB:
            logger.debug("Can load without optimization - sufficient VRAM available")
            return True, False

        # Not enough available VRAM but model can fit in total VRAM
        # Will need to make space but no optimization needed
        logger.debug("Need to make space in VRAM")
        return False, False

    def _determine_load_location(self, model_size: float) -> str:
        """
        Determine optimal location to load the model based on available resources.

        Args:
            model_size: Size of the model in GB

        Returns:
            String indicating load location: "gpu", "gpu_optimized", "needs_space", or "none"
        """
        # If model is bigger than total VRAM, it must be optimized
        if model_size > self.max_vram:
            if self._can_load_to_ram(model_size):
                return "gpu_optimized"  # Will use CPU offload
            return "none"  # Can't even fit in CPU

        # Model can fit in total VRAM
        available_vram = self._get_available_vram() - VRAM_SAFETY_MARGIN_GB
        if model_size <= available_vram:
            return "gpu"

        # Need to make space in VRAM first
        return "needs_space"

    async def load(
        self, model_id: str, gpu: Optional[int] = None, pipe_type: Optional[str] = None
    ) -> Optional[DiffusionPipeline]:
        """
        Load a model into memory, handling placement and optimization.

        This method implements the main model loading logic, including:
        - Checking if model is already loaded
        - Managing memory allocation between GPU and CPU
        - Applying optimizations as needed
        - Handling model movement between devices

        Args:
            model_id: Identifier for the model to load
            gpu: Optional GPU device number
            pipe_type: Optional pipeline type specification

        Returns:
            Loaded pipeline or None if loading failed
        """
        logger.info(f"Loading model {model_id}")

        # Check existing loaded models
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} already loaded in GPU")
            self.lru_cache.access(model_id, "gpu")
            return self.loaded_models[model_id]

        if model_id in self.cpu_models:
            return await self._handle_cpu_model_load(model_id)

        # Load new model
        return await self._load_new_model(model_id, gpu, pipe_type)

    async def _handle_cpu_model_load(
        self, model_id: str
    ) -> Optional[DiffusionPipeline]:
        """
        Handle loading of a model that exists in CPU memory.

        Attempts to move the model to GPU if possible, applying optimizations
        or keeping in CPU if necessary.

        Args:
            model_id: Identifier for the model

        Returns:
            Pipeline object or None if operation failed
        """
        logger.info(f"Model {model_id} is in CPU memory")
        model_size = self.model_sizes[model_id]

        # Attempt GPU transfer if possible
        available_vram = self._get_available_vram() - VRAM_SAFETY_MARGIN_GB

        if model_size > available_vram:
            self._make_space_for_model(model_size)

        can_load_gpu, need_optimization = self._can_load_model(model_size)
        if not can_load_gpu:
            logger.info(f"Insufficient GPU memory for {model_id}, keeping in CPU")
            return self.cpu_models[model_id]

        # Move model from CPU to GPU
        try:
            pipeline = self.cpu_models.pop(model_id)
            self.ram_usage -= model_size

            if need_optimization:
                logger.info(f"Applying optimizations for {model_id}")
                self.apply_optimizations(pipeline, model_id, True)
                self._restore_cpu_model(model_id, pipeline, model_size)
                return pipeline

            if self._move_model_to_gpu(pipeline, model_id):
                logger.info(f"Successfully moved {model_id} to GPU")
                self._update_gpu_model(model_id, pipeline, model_size)

                return pipeline

            logger.info(f"Failed to move {model_id} to GPU, keeping in CPU")
            self._restore_cpu_model(model_id, pipeline, model_size)
            return pipeline

        except Exception as e:
            logger.error(f"Error handling CPU model load for {model_id}: {str(e)}")
            return None

    def _update_gpu_model(
        self, model_id: str, pipeline: DiffusionPipeline, model_size: float
    ) -> None:
        """
        Update GPU model tracking after successful model movement.

        Args:
            model_id: Model identifier
            pipeline: The pipeline that was moved
            model_size: Size of the model in GB
        """
        self.loaded_models[model_id] = pipeline
        self.vram_usage += model_size
        self.lru_cache.remove(model_id, "cpu")
        self.lru_cache.access(model_id, "gpu")

    def _restore_cpu_model(
        self, model_id: str, pipeline: DiffusionPipeline, model_size: float
    ) -> None:
        """
        Restore model to CPU tracking after failed GPU movement.

        Args:
            model_id: Model identifier
            pipeline: The pipeline to restore
            model_size: Size of the model in GB
        """
        self.cpu_models[model_id] = pipeline
        self.ram_usage += model_size

    async def _load_new_model(
        self, model_id: str, gpu: Optional[int] = None, pipe_type: Optional[str] = None
    ) -> Optional[DiffusionPipeline]:
        """
        Load a new model that isn't currently in memory.

        Handles the complete loading process including:
        - Configuration validation
        - Memory allocation
        - Model loading
        - Optimization application

        Args:
            model_id: Identifier for the model
            gpu: Optional GPU device number
            pipe_type: Optional pipeline type specification (deprecated, use class_name in config)

        Returns:
            Loaded pipeline or None if loading failed
        """
        try:
            config = get_config()
            model_config = config.pipeline_defs.get(model_id)

            if not model_config:
                logger.error(f"Model {model_id} not found in configuration")
                return None

            # Prepare memory and determine load location
            estimated_size = await self._get_model_size(model_config, model_id)
            load_location = self._determine_load_location(estimated_size)

            if load_location == "none":
                logger.error(f"Insufficient memory to load model {model_id}")
                return None

            if load_location in ["needs_space", "gpu_optimized"]:
                self._make_space_for_model(estimated_size)
                load_location = self._determine_load_location(estimated_size)
                print(f"Load location after making space: {load_location}")

            # Load the model
            pipeline = await self._load_model_by_source(model_id, model_config, gpu)
            if pipeline is None:
                return None

            print(f"Load location: {load_location}")

            # Get the model default configs from the model config manager
            # model_default_configs = self.model_config_manager.get_model_config(
            #     model_id, pipeline.__class__.__name__
            # )

            # Set up scheduler if specified in config
            # if (
            #     "scheduler" in model_default_configs
            #     and model_default_configs["scheduler"] != ""
            # ):
            self._setup_scheduler(pipeline, model_id)

            # Place in appropriate memory location
            return await self._place_model_in_memory(
                pipeline, model_id, estimated_size, load_location
            )

        except Exception as e:
            logger.error("Error loading new model {}: {}".format(model_id, str(e)))
            traceback.print_exc()
            return None

    async def _load_model_by_source(
        self, model_id: str, model_config: PipelineConfig, gpu: Optional[int]
    ) -> Optional[DiffusionPipeline]:
        """
        Load a model based on its source configuration.

        Args:
            model_id: Model identifier
            model_config: Model configuration from PipelineConfig
            gpu: Optional GPU device number

        Returns:
            Loaded pipeline or None if loading failed
        """

        if isinstance(model_config, dict):
            source = model_config.get("source")
            class_name = model_config.get("class_name")
        else:
            source = model_config.source
            class_name = model_config.class_name

        prefix, path = source.split(":", 1)

        try:
            if prefix == "hf":
                is_downloaded, variant = await self.model_downloader.is_downloaded(
                    model_id
                )

                if not is_downloaded:
                    logger.info(f"Model {model_id} not downloaded")
                    return None

                # Get model index and use as fallback in case class_name is unspecified
                if class_name is None or class_name == "":
                    model_index = await self.hf_model_manager.get_diffusers_multifolder_components(
                        path
                    )
                    if model_index and "_class_name" in model_index:
                        class_name = model_index["_class_name"]
                    else:
                        logger.error(f"Unknown diffusers class_name for {model_id}")
                        return None

                return await self.load_huggingface_model(
                    model_id, path, class_name, gpu, variant, model_config
                )
            elif prefix in ["file", "ct"]:
                return await self.load_single_file_model(
                    model_id, path, prefix, gpu, class_name
                )
            elif source.startswith(("http://", "https://")):
                # Handle Civitai/direct download models
                source_obj = ModelSource(source)
                is_downloaded = await self.model_downloader.is_downloaded(model_id)
                if not is_downloaded:
                    logger.info(f"Model {model_id} not downloaded")
                    return None

                cached_path = await self.model_downloader._get_cache_path(
                    model_id, source_obj
                )
                if not os.path.exists(cached_path):
                    logger.error(f"Cached model file not found at {cached_path}")
                    return None

                return await self.load_single_file_model(
                    model_id, cached_path, "file", gpu, class_name
                )
            else:
                logger.error(f"Unsupported model source prefix: {prefix}")
                return None
        except Exception as e:
            logger.error("Error loading model from source: {}".format(str(e)))
            return None

    def _make_space_for_model(self, model_size: float) -> None:
        """
        Attempt to free up memory space for a new model.

        Implements a sophisticated memory management strategy:
        1. First try to move models to CPU if possible
        2. If that's not possible, unload models completely
        3. Tracks all memory changes through LRU cache

        Args:
            model_size: Size of the model requiring space in GB
        """
        available_vram = self._get_available_vram() - VRAM_SAFETY_MARGIN_GB

        # If model is bigger than total VRAM, we need to clear everything
        if model_size > self.max_vram:
            logger.info("Model exceeds total VRAM - clearing all GPU models")
            gpu_models = list(self.loaded_models.keys())
            for model_id in gpu_models:
                self._move_model_to_cpu_for_space(model_id, self.model_sizes[model_id])
            return

        # If model can fit in total VRAM but needs space
        if available_vram < model_size:
            space_needed = model_size - available_vram
            logger.info(f"Need to free {space_needed:.2f} GB of VRAM")

            # Get GPU models in LRU order
            gpu_models = [
                (mid, self.model_sizes[mid])
                for mid in self.lru_cache.get_lru_models("gpu")
            ]

            freed_space = 0
            for model_id, size in gpu_models:
                if freed_space >= space_needed:
                    break

                # Try to move to CPU first
                if self._can_load_to_ram(size):
                    freed_space += self._move_model_to_cpu_for_space(model_id, size)
                else:
                    # If can't move to CPU, unload completely
                    freed_space += self._unload_model_for_space(model_id, size, "gpu")

                print(f"Freed space: {freed_space}")

                if self._get_available_vram() - VRAM_SAFETY_MARGIN_GB >= model_size:
                    break

    def _move_model_to_cpu_for_space(self, model_id: str, model_size: float) -> float:
        """
        Move a model from GPU to CPU to free up space.

        Args:
            model_id: Model identifier
            model_size: Size of the model in GB

        Returns:
            Amount of space freed in GB
        """
        logger.info(f"Moving {model_id} to CPU to make space in GPU")
        pipeline = self.loaded_models.pop(model_id)
        self.vram_usage -= model_size

        if self._move_model_to_cpu(pipeline, model_id):
            self.cpu_models[model_id] = pipeline
            self.ram_usage += model_size
            self.lru_cache.remove(model_id, "gpu")
            self.lru_cache.access(model_id, "cpu")
            return model_size

        del pipeline
        self.lru_cache.remove(model_id, "gpu")
        return model_size

    def _unload_model_for_space(
        self, model_id: str, model_size: float, device: str
    ) -> float:
        """
        Completely unload a model and free up GPU/CPU memory.

        Args:
            model_id: Model identifier
            model_size: Size of the model in GB
            device: Device to unload from ("gpu" or "cpu")
        Returns:
            Amount of space freed in GB
        """
        logger.info(f"Unloading {model_id} from memory")

        try:
            if device == "gpu" and model_id in self.loaded_models:
                pipeline = self.loaded_models[model_id]

                # # Move model to CPU first to clear CUDA memory
                # if hasattr(pipeline, "to"):
                #     pipeline.to("cpu", silence_dtype_warnings=True)

                # Explicitly delete model components
                for attr in [
                    "vae",
                    "unet",
                    "text_encoder",
                    "text_encoder_2",
                    "tokenizer",
                    "scheduler",
                    "transformer",
                    "tokenizer_2",
                    "text_encoder_3",
                    "tokenizer_3",
                ]:
                    if hasattr(pipeline, attr) and getattr(pipeline, attr) is not None:
                        component = getattr(pipeline, attr)
                        delattr(pipeline, attr)
                        del component

                # Delete pipeline reference
                del self.loaded_models[model_id]
                self.vram_usage -= model_size
                self.lru_cache.remove(model_id, "gpu")

            if device == "cpu" and model_id in self.cpu_models:
                pipeline = self.cpu_models[model_id]

                # Same cleanup for CPU models
                for attr in [
                    "vae",
                    "unet",
                    "text_encoder",
                    "text_encoder_2",
                    "tokenizer",
                    "scheduler",
                    "transformer",
                    "tokenizer_2",
                    "text_encoder_3",
                    "tokenizer_3",
                ]:
                    if hasattr(pipeline, attr) and getattr(pipeline, attr) is not None:
                        component = getattr(pipeline, attr)
                        delattr(pipeline, attr)
                        del component

                del self.cpu_models[model_id]
                self.ram_usage -= model_size
                self.lru_cache.remove(model_id, "cpu")

            # Force garbage collection and memory clearing
            self.flush_memory()

            return model_size

        except Exception as e:
            logger.error(
                "Error during model unloading for {}: {}".format(model_id, str(e))
            )
            # Still try to clean up references even if error occurs
            self.loaded_models.pop(model_id, None)
            self.cpu_models.pop(model_id, None)
            self.vram_usage -= model_size
            self.lru_cache.remove(model_id, "gpu")
            self.lru_cache.remove(model_id, "cpu")
            self.flush_memory()
            return model_size

    def _move_model_to_cpu(self, pipeline: DiffusionPipeline, model_id: str) -> bool:
        """
        Safely move a model to CPU memory with proper dtype handling.

        Args:
            pipeline: The pipeline to move
            model_id: Identifier of the model

        Returns:
            Boolean indicating success of the operation
        """
        # Skip if it's OmniGen pipeline
        if pipeline.__class__.__name__ == "OmniGenPipeline":
            return True

        try:
            self.flush_memory()

            # Store original dtype
            if model_id not in self.model_types:
                self._store_model_dtype(pipeline, model_id)

            pipeline = pipeline.to("cpu", silence_dtype_warnings=True)
            self.flush_memory()
            return True

        except Exception as e:
            logger.error("Failed to move model {} to CPU: {}".format(model_id, str(e)))
            return False

    def _store_model_dtype(self, pipeline: DiffusionPipeline, model_id: str) -> None:
        """
        Store the original dtype of a model for future reference.

        Args:
            pipeline: The pipeline to get dtype from
            model_id: Model identifier
        """
        if hasattr(pipeline, "dtype"):
            self.model_types[model_id] = pipeline.dtype
        else:
            for attr in ["vae", "unet", "text_encoder", "transformer"]:
                if hasattr(pipeline, attr):
                    component = getattr(pipeline, attr)
                    if hasattr(component, "dtype"):
                        self.model_types[model_id] = component.dtype
                        break

    def _move_model_to_gpu(self, pipeline: DiffusionPipeline, model_id: str) -> bool:
        """
        Safely move a model to GPU memory with proper dtype handling.

        Args:
            pipeline: The pipeline to move
            model_id: Identifier of the model

        Returns:
            Boolean indicating success of the operation
        """
        # Skip if it's OmniGen pipeline
        if pipeline.__class__.__name__ == "OmniGenPipeline":
            return True

        try:
            device = get_available_torch_device()
            # Store the original pipeline components for cleanup

            # cpu_components = {}
            # for attr in ["vae", "unet", "text_encoder", "text_encoder_2", "transformer"]:
            #     if hasattr(pipeline, attr) and getattr(pipeline, attr) is not None:
            #         cpu_components[attr] = getattr(pipeline, attr)

            pipeline = pipeline.to(device=device)

            print("Done with moving to GPU")

            # for attr, component in cpu_components.items():
            #     if hasattr(component, "parameters"):
            #         print(f"Clearing {attr} parameters")
            #     del component

            self.flush_memory()
            return True

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU out of memory while moving model {model_id} to GPU")
            else:
                logger.error(
                    "Runtime error moving model {} to GPU: {}".format(model_id, str(e))
                )
            self.flush_memory()
            return False

    async def _place_model_in_memory(
        self,
        pipeline: DiffusionPipeline,
        model_id: str,
        model_size: float,
        load_location: str,
    ) -> Optional[DiffusionPipeline]:
        """
        Place a loaded model in the appropriate memory location.

        Args:
            pipeline: The loaded pipeline
            model_id: Model identifier
            model_size: Size of the model in GB
            load_location: Target location ("gpu", "gpu_optimized", "needs_space")

        Returns:
            The pipeline in its final location or None if placement failed
        """
        try:
            # Handle OmniGen pipeline
            if pipeline.__class__.__name__ == "OmniGenPipeline":
                self.loaded_models[model_id] = pipeline
                self.model_sizes[model_id] = model_size
                self.vram_usage += model_size
                self.lru_cache.access(model_id, "gpu")
                return pipeline

            if load_location == "needs_space":
                self._make_space_for_model(model_size)
                if self._move_model_to_gpu(pipeline, model_id):
                    self.loaded_models[model_id] = pipeline
                    self.model_sizes[model_id] = model_size
                    self.vram_usage += model_size
                    self.lru_cache.access(model_id, "gpu")
                    return pipeline
                logger.error(
                    f"Failed to move model {model_id} to GPU after making space"
                )
                return None

            if load_location == "gpu":
                print(f"Moving {model_id} to GPU")
                if self._move_model_to_gpu(pipeline, model_id):
                    self.loaded_models[model_id] = pipeline
                    self.model_sizes[model_id] = model_size
                    self.vram_usage += model_size
                    self.lru_cache.access(model_id, "gpu")
                    return pipeline
                logger.error(f"Failed to move model {model_id} to GPU")
                return None

            elif load_location == "gpu_optimized":
                # Clear GPU completely before optimizing
                self._make_space_for_model(model_size)
                self.apply_optimizations(pipeline, model_id, True)
                self.cpu_models[model_id] = pipeline
                self.model_sizes[model_id] = model_size
                self.ram_usage += model_size
                self.lru_cache.access(model_id, "cpu")
                return pipeline

            return None

        except Exception as e:
            logger.error(
                "Error placing model {} in memory: {}".format(model_id, str(e))
            )
            return None

    async def load_huggingface_model(
        self,
        model_id: str,
        repo_id: str,
        class_name: Union[str, Tuple[str, str]],
        gpu: Optional[int] = None,
        variant: Optional[str] = None,
        model_config: Optional[PipelineConfig] = None,
    ) -> Optional[DiffusionPipeline]:
        """
        Load a model from HuggingFace.

        Args:
            model_id: Model identifier
            repo_id: HuggingFace repository ID
            gpu: Optional GPU device number
            class_name: Pipeline class name or (package, class) tuple
            variant: Optional model variant
            model_config: Optional model configuration

        Returns:
            Loaded pipeline or None if loading failed
        """
        try:
            pipeline_kwargs = await self._prepare_pipeline_kwargs(model_config)
            variant = None if variant == "" else variant
            # TO DO: make this more robust
            torch_dtype = (
                torch.bfloat16 if "flux" in model_id.lower() else torch.float16
            )

            # Get appropriate pipeline class
            (pipeline_class, custom_pipeline) = get_pipeline_class(class_name)
            print("custompipeline", custom_pipeline)

            if custom_pipeline is not None:
                pipeline_kwargs["custom_pipeline"] = custom_pipeline

            print(
                f"repo_id={repo_id},torch_dtype={torch_dtype},local_files_only=True,variant={variant},pipeline_kwargs={pipeline_kwargs},"
            )
            try:
                pipeline = pipeline_class.from_pretrained(
                    repo_id,
                    torch_dtype=torch_dtype,
                    local_files_only=True,
                    variant=variant,
                    **pipeline_kwargs,
                )
            except EntryNotFoundError as e:
                print(f"Custom pipeline '{custom_pipeline}' not found: {e}")
                print("Falling back to the default pipeline...")
                del pipeline_kwargs["custom_pipeline"]
                pipeline = pipeline_class.from_pretrained(
                    repo_id,
                    variant=variant,
                    torch_dtype=torch_dtype,
                    **pipeline_kwargs,
                )

            self.flush_memory()
            self.loaded_model = pipeline
            self.current_model = model_id
            logger.info(f"Model {model_id} loaded successfully")
            return pipeline

        except Exception as e:
            traceback.print_exc()
            logger.error("Failed to load model {}: {}".format(model_id, str(e)))
            return None

    async def _prepare_pipeline_kwargs(
        self, model_config: Optional[PipelineConfig]
    ) -> Dict[str, Any]:
        """
        Prepare kwargs for pipeline initialization.

        Args:
            model_config: Model configuration from PipelineConfig

        Returns:
            Dictionary of pipeline initialization arguments
        """
        if not model_config:
            return {}

        pipeline_kwargs = {}
        try:
            if isinstance(model_config, dict):
                class_name = model_config.get("class_name")
                components = model_config.get("components")
                custom_pipeline = model_config.get("custom_pipeline")
            else:
                class_name = model_config.class_name
                components = model_config.components

            if custom_pipeline:
                pipeline_kwargs["custom_pipeline"] = custom_pipeline

            if components:
                for key, component in components.items():
                    main_model_source = model_config.get("source")
                    # Check if component and also id component has .source attribute
                    if component:
                        component_source = component.get("source", None)
                        if component_source:
                            pipeline_kwargs[key] = await self._prepare_component(
                                main_model_source,component, class_name, key
                            )

            # Handle custom pipeline if specified as string
            # if isinstance(class_name, str):
            #     pipeline_kwargs["custom_pipeline"] = class_name

            return pipeline_kwargs

        except Exception as e:
            logger.error("Error preparing pipeline kwargs: {}".format(str(e)))
            return {}

    async def _prepare_component(
        self,
        main_model_source: str,
        component: PipelineConfig,
        model_class_name: Optional[Union[str, Tuple[str, str]]],
        key: str,
    ) -> Any:
        """
        Prepare a model component based on its configuration.

        Args:
            component: Component configuration
            model_class_name: Class name of the parent model
            key: Component key

        Returns:
            Loaded component or None if loading failed
        """
        try:
            if isinstance(component, dict):
                source = component.get("source")
            else:
                source = component.source

            if not source.endswith((".safetensors", ".bin", ".ckpt", ".pt")):
                # check if the url has more than 2 forward slashes. If it does, the last one is the subfolder, the source is the first part
                # e.g. hf:cozy-creator/Flux.1-schnell-8bit/transformer this will be, source = hf:cozy-creator/Flux.1-schnell-8bit, subfolder = transformer
                if source.count("/") > 2:
                    repo_id = "/".join(source.split("/")[:-1])
                    subfolder = source.split("/")[-1]
                    return await self._load_diffusers_component(
                        main_model_source.replace("hf:", ""), repo_id.replace("hf:", ""), subfolder
                    )
                else:
                    return await self._load_diffusers_component(
                        main_model_source.replace("hf:", ""), source.replace("hf:", "")
                    )
            else:
                return self._load_custom_component(source, model_class_name, key)
        except Exception as e:
            logger.error("Error preparing component {}: {}".format(key, str(e)))
            return None

    async def load_single_file_model(
        self,
        model_id: str,
        path: str,
        prefix: str,
        gpu: Optional[int] = None,
        class_name: Optional[Union[str, Tuple[str, str]]] = None,
    ) -> Optional[DiffusionPipeline]:
        """
        Load a model from a single file.

        Args:
            model_id: Model identifier
            path: Path to model file
            prefix: Source prefix (file/ct)
            gpu: Optional GPU device number
            class_name: Pipeline class name or (package, class) tuple

        Returns:
            Loaded pipeline or None if loading failed
        """
        logger.info(f"Loading single file model {model_id}")

        # TO DO: we could try inferring the class using our old detect_model code here!
        if class_name is None or class_name == "":
            logger.error("Model class_name must be specified for single file models")
            return None

        (pipeline_class, custom_pipeline) = get_pipeline_class(class_name)

        try:
            print(f"Model path: {path}")
            if prefix != "file":
                model_path = self._get_model_path(path, prefix)
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    return None
            else:
                model_path = path

            pipeline = await self._load_pipeline_from_file(
                pipeline_class, model_path, model_id, class_name
            )
            if pipeline:
                self.loaded_model = pipeline
                self.current_model = model_id
            return pipeline

        except Exception as e:
            logger.error("Error loading single file model: {}".format(str(e)))
            return None

    def _get_model_path(self, path: str, prefix: str) -> str:
        if prefix == "ct" and ("http" in path or "https" in path):
            path = path.split("/")[-1]
        return os.path.join(get_config().models_path, path)

    async def _load_pipeline_from_file(
        self,
        pipeline_class: Type[DiffusionPipeline],
        model_path: str,
        model_id: str,
        class_name: Optional[Union[str, Tuple[str, str]]],
    ) -> Optional[DiffusionPipeline]:
        """
        Load a pipeline from a file using appropriate loading method.

        Args:
            pipeline_class: Class to instantiate pipeline
            model_path: Path to model file
            model_id: Model identifier
            class_name: Pipeline class name or (package, class) tuple

        Returns:
            Loaded pipeline or None if loading failed
        """
        try:
            if issubclass(pipeline_class, FromSingleFileMixin):
                return self._load_from_single_file(pipeline_class, model_path, model_id)
            else:
                # Get the actual class name string if it's a tuple
                class_str = (
                    class_name[1] if isinstance(class_name, tuple) else class_name
                )
                return self._load_custom_architecture(
                    pipeline_class, model_path, class_str
                )
        except Exception as e:
            logger.error("Error loading pipeline from file: {}".format(str(e)))
            return None

    def _load_from_single_file(
        self, pipeline_class: Any, path: str, model_id: str
    ) -> DiffusionPipeline:
        """
        Load a model pipeline from a single file using the FromSingleFileMixin.

        Uses different torch datatypes based on model type:
        - bfloat16 for Flux models
        - float16 for other models

        Args:
            pipeline_class: The pipeline class to instantiate
            path: Path to the model file
            model_id: Model identifier (used to determine model type)

        Returns:
            Loaded pipeline instance

        Raises:
            Exception: If loading fails
        """
        try:
            # Determine appropriate dtype based on model type
            torch_dtype = (
                torch.bfloat16 if "flux" in model_id.lower() else torch.float16
            )

            # Load the model with appropriate dtype
            pipeline = pipeline_class.from_single_file(path, torch_dtype=torch_dtype)

            logger.info(f"Successfully loaded single file model {model_id}")
            return pipeline

        except Exception as e:
            logger.error("Error loading model from single file: {}".format(str(e)))
            raise

    def _load_custom_architecture(
        self, pipeline_class: Any, path: str, type: str
    ) -> DiffusionPipeline:
        """
        Load a model with custom architecture configuration.

        Handles loading of individual components and assembling them into
        a complete pipeline.

        Args:
            pipeline_class: The pipeline class to instantiate
            path: Path to the model file
            type: Model type (determines which components to load)

        Returns:
            Assembled pipeline instance

        Raises:
            Exception: If loading fails or architecture is not found
        """
        try:
            # Load the complete state dict
            state_dict = load_state_dict_from_file(path)

            # Create empty pipeline instance
            pipeline = pipeline_class()

            # Load each component specified for this model type
            for component_name in MODEL_COMPONENTS[type]:
                # Skip certain components that don't need loading
                if component_name in ["scheduler", "tokenizer", "tokenizer_2"]:
                    continue

                # Construct architecture key and get class
                arch_key = f"core_extension_1.{type}_{component_name}"
                architecture_class = get_architectures().get(arch_key)

                if not architecture_class:
                    logger.error(f"Architecture not found for {arch_key}")
                    continue

                try:
                    # Initialize and load the component
                    architecture = architecture_class()
                    architecture.load(state_dict)

                    # Set the component in the pipeline
                    setattr(pipeline, component_name, architecture.model)
                    logger.debug(f"Loaded component {component_name} for {type}")

                except Exception as component_error:
                    logger.error(
                        "Error loading component {}: {}".format(
                            component_name, str(component_error)
                        )
                    )
                    continue

            logger.info("Successfully loaded custom architecture model")
            return pipeline

        except Exception as e:
            logger.error("Error loading custom architecture: {}".format(str(e)))
            raise

    def apply_optimizations(
        self,
        pipeline: DiffusionPipeline,
        model_id: str,
        force_full_optimization: bool = False,
    ) -> None:
        """
        Apply memory and performance optimizations to a pipeline.
        Uses existing optimization functions when model exceeds total VRAM.

        Args:
            pipeline: The pipeline to optimize
            model_id: Model identifier
            force_full_optimization: Whether to force optimization
        """
        # Skip if it's OmniGen pipeline
        if pipeline.__class__.__name__ == "OmniGenPipeline":
            return

        if self.loaded_model is not None and self.is_in_device:
            logger.info(f"Model {model_id} already optimized")
            return

        device = get_available_torch_device()

        # Only optimize if model is bigger than total VRAM
        model_size = self.model_sizes.get(model_id, 0)
        if model_size > self.max_vram or force_full_optimization:
            logger.info(f"Applying optimizations for {model_id}")

            # Get list of optimizations
            optimizations = self._get_full_optimizations()

            # Apply the optimizations
            self._apply_optimization_list(pipeline, optimizations, device)
        else:
            logger.info(f"No optimization needed for {model_id}")

    # def _determine_optimizations(
    #     self,
    #     available_vram_gb: float,
    #     model_size_gb: float,
    #     force_full_optimization: bool
    # ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], bool]:
    #     """
    #     Determine which optimizations to apply based on available resources.

    #     Args:
    #         available_vram_gb: Available VRAM in GB
    #         model_size_gb: Model size in GB
    #         force_full_optimization: Whether to force full optimizations

    #     Returns:
    #         Tuple of (list of optimizations, whether full optimization is needed)
    #     """
    #     optimizations = []

    #     if self.should_quantize and available_vram_gb >= GPUEnum.HIGH.value:
    #         force_full_optimization = False
    #     elif available_vram_gb >= GPUEnum.VERY_HIGH.value:
    #         force_full_optimization = False
    #     elif available_vram_gb > model_size_gb:
    #         if force_full_optimization and available_vram_gb < GPUEnum.HIGH.value:
    #             optimizations = self._get_full_optimizations()
    #             force_full_optimization = True
    #     else:
    #         if not (self.should_quantize and available_vram_gb >= GPUEnum.HIGH.value):
    #             optimizations = self._get_full_optimizations()
    #             force_full_optimization = True

    #     return optimizations, force_full_optimization

    def _get_full_optimizations(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get list of all available optimizations.

        Returns:
            List of tuples containing (optimization_function, name, parameters)
        """
        optimizations = [
            ("enable_vae_slicing", "VAE Sliced", {}),
            ("enable_vae_tiling", "VAE Tiled", {}),
            (
                "enable_model_cpu_offload",
                "CPU Offloading",
                {"device": get_available_torch_device()},
            ),
        ]

        if not isinstance(self.loaded_model, (FluxPipeline, FluxInpaintPipeline)):
            optimizations.append(
                (
                    "enable_xformers_memory_efficient_attention",
                    "Memory Efficient Attention",
                    {},
                )
            )

        return optimizations

    def _apply_optimization_list(
        self,
        pipeline: DiffusionPipeline,
        optimizations: List[Tuple[str, str, Dict[str, Any]]],
        device: torch.device,
    ) -> None:
        """
        Apply a list of optimizations to a pipeline.

        Args:
            pipeline: The pipeline to optimize
            optimizations: List of optimization specifications
            device: Target device
        """
        device_type = device if isinstance(device, str) else device.type

        if device_type == "mps":
            setattr(torch, "mps", torch.backends.mps)

        for opt_func, opt_name, kwargs in optimizations:
            try:
                getattr(pipeline, opt_func)(**kwargs)
                logger.info("{} enabled".format(opt_name))
            except Exception as e:
                logger.error("Error enabling {}: {}".format(opt_name, str(e)))

        if device_type == "mps":
            delattr(torch, "mps")

    def flush_memory(self) -> None:
        """Clear unused memory from GPU and perform garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            pass  # MPS doesn't need explicit cache clearing

    async def warmup_pipeline(self, model_id: str) -> None:
        """
        Warm up a pipeline by running a test inference.

        Args:
            model_id: Model identifier
        """

        if model_id:
            logger.info(f"Loading model {model_id} for warm-up")
            pipeline = await self.load(model_id)

        if model_id not in self.loaded_models:
            logger.warning(f"Model {model_id} is not loaded")
            return

        # if model_id in self.cpu_models:
        #     logger.info(f"Loading model {model_id} from CPU to GPU")
        #     self.load(model_id)

        pipeline = self.loaded_models[model_id]
        if pipeline is None:
            logger.warning("Failed to load model {} for warm-up".format(model_id))
            return

        logger.info("Warming up pipeline for model {}".format(model_id))

        try:
            with torch.no_grad():
                if isinstance(pipeline, DiffusionPipeline) and callable(pipeline):
                    _ = pipeline(
                        prompt="This is a warm-up prompt",
                        num_inference_steps=20,
                        output_type="pil",
                    )
                else:
                    logger.warning(
                        "Unsupported pipeline type for warm-up: {}".format(
                            type(pipeline)
                        )
                    )
        except Exception as e:
            logger.error(
                "Error during warm-up for model {}: {}".format(model_id, str(e))
            )

        self.flush_memory()
        logger.info("Warm-up completed for model {}".format(model_id))

    def get_all_model_ids(self) -> List[str]:
        """
        Get list of all available model IDs from config.

        Returns:
            List of model identifiers
        """
        config = get_config()
        return list(config.pipeline_defs.keys())

    def get_enabled_models(self) -> List[str]:
        """
        Get list of models that should be warmed up.

        Returns:
            List of model identifiers to be used for generation
        """
        config = get_config()
        return config.enabled_models

    def unload(self, model_id: str) -> None:
        """
        Unload a model from memory and clean up associated resources.

        Handles unloading from both GPU and CPU memory, updates memory tracking,
        and cleans up LRU cache entries.

        Args:
            model_id: Model identifier to unload
        """
        try:
            # Unload from GPU if present
            if model_id in self.loaded_models:
                model_size = self.model_sizes.get(model_id, 0)
                pipeline = self.loaded_models.pop(model_id)

                del pipeline
                self.vram_usage -= model_size
                self.lru_cache.remove(model_id, "gpu")
                logger.info(f"Model {model_id} unloaded from GPU")

            # Unload from CPU if present
            if model_id in self.cpu_models:
                model_size = self.model_sizes.get(model_id, 0)
                pipeline = self.cpu_models.pop(model_id)

                del pipeline
                self.ram_usage -= model_size
                self.lru_cache.remove(model_id, "cpu")
                logger.info(f"Model {model_id} unloaded from CPU")

            # Clean up current model reference if it matches
            if model_id == self.current_model:
                self.loaded_model = None
                self.current_model = None

            # Remove from model sizes tracking
            if model_id in self.model_sizes:
                del self.model_sizes[model_id]

            # Remove from model types tracking
            if model_id in self.model_types:
                del self.model_types[model_id]

            # Force memory cleanup
            self.flush_memory()

        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {str(e)}")

    def is_loaded(self, model_id: str) -> bool:
        """
        Check if a model is currently loaded.

        Args:
            model_id: Model identifier

        Returns:
            Boolean indicating if model is loaded
        """
        return model_id == self.current_model and self.loaded_model is not None

    def get_model(self, model_id: str) -> Optional[DiffusionPipeline]:
        """
        Get a loaded model pipeline.

        Args:
            model_id: Model identifier

        Returns:
            Pipeline if model is loaded, None otherwise
        """
        if not self.is_loaded(model_id):
            return None
        return self.loaded_model

    def get_model_device(self, model_id: str) -> Optional[torch.device]:
        """
        Get the device where a model is loaded.

        Args:
            model_id: Model identifier

        Returns:
            Device if model is loaded, None otherwise
        """
        model = self.get_model(model_id)
        return model.device if model else None

    # def _get_model_size(self, model_config: Dict[str, Any]) -> float:
    #     """
    #     Calculate the total size of a model including all its components.

    #     Args:
    #         model_config: Model configuration dictionary

    #     Returns:
    #         Total size in GB
    #     """
    #     if any(prefix in model_config["source"] for prefix in ["ct:", "file:"]):
    #         path = model_config["source"].replace("ct:", "").replace("file:", "")
    #         return os.path.getsize(path) / (1024**3)

    #     repo_id = model_config["source"].replace("hf:", "")
    #     return self._calculate_repo_size(repo_id, model_config)

    async def _get_model_size(
        self, model_config: PipelineConfig, model_id: str
    ) -> float:
        """
        Calculate the total size of a model including all its components.

        Args:
            model_config: Model configuration dictionary from pipeline_defs
            model_id: The model identifier (key from pipeline_defs)

        Returns:
            Size in GB
        """

        if isinstance(model_config, dict):
            source = model_config.get("source")
        else:
            source = model_config.source

        if source.startswith(("http://", "https://")):
            # For downloaded HTTP(S) models, get size from cache
            try:
                source_obj = ModelSource(source)  # Create ModelSource object
                cache_path = await self.model_downloader._get_cache_path(
                    model_id, source_obj
                )
                print(f"Cache Path: {cache_path}")
                if os.path.exists(cache_path):
                    return os.path.getsize(cache_path) / (1024**3)
                else:
                    logger.warning(
                        f"Cache file not found for {model_id}, assuming default size"
                    )
                    return 7.0  # Default size assumption (never going to be used)
            except Exception as e:
                logger.error(f"Error getting cached model size: {e}")
                return (
                    7.0  # Default fallback size (never going to be used for anything)
                )
        elif source.startswith("file:"):
            path = source.replace("file:", "")
            return os.path.getsize(path) / (1024**3) if os.path.exists(path) else 0
        elif source.startswith("hf:"):
            # Handle HuggingFace models as before
            repo_id = source.replace("hf:", "")
            return self._calculate_repo_size(repo_id, model_config)
        else:
            logger.error(f"Unsupported source type for size calculation: {source}")
            return 0

    def _calculate_repo_size(self, repo_id: str, model_config: PipelineConfig) -> float:
        """
        Calculate the total size of a HuggingFace repository.

        Args:
            repo_id: Repository identifier
            model_config: Model configuration dictionary

        Returns:
            Total size in GB
        """
        total_size = self._get_size_for_repo(repo_id)

        if isinstance(model_config, dict):
            components = model_config.get("components")
        else:
            components = model_config.components

        if components and components:
            for key, component in components.items():
                if isinstance(component, dict) and "source" in component:
                    component_size = self._calculate_component_size(
                        component, repo_id, key
                    )
                    total_size += component_size

        total_size_gb = total_size / (1024**3)
        logger.debug(f"Total size: {total_size_gb:.2f} GB")
        return total_size_gb

    def _calculate_component_size(
        self, component: Dict[str, Any], repo_id: str, key: str
    ) -> float:
        """
        Calculate the size of a model component.

        Args:
            component: Component configuration
            repo_id: Repository identifier
            key: Component key

        Returns:
            Component size in bytes
        """
        component_source = component["source"]
        if len(component_source.split("/")) > 2:
            component_repo = "/".join(component_source.split("/")[0:2]).replace(
                "hf:", ""
            )
        else:
            component_repo = component_source.replace("hf:", "")

        component_name = (
            key
            if not component_source.endswith((".safetensors", ".bin", ".ckpt", ".pt"))
            else component_source.split("/")[-1]
        )

        total_size = self._get_size_for_repo(
            component_repo, component_name
        ) - self._get_size_for_repo(repo_id, key)

        return total_size

    def _get_size_for_repo(
        self, repo_id: str, component_name: Optional[str] = None
    ) -> int:
        """
        Get the size of a specific repository or component.

        Args:
            repo_id: Repository identifier
            component_name: Optional component name

        Returns:
            Size in bytes
        """
        if component_name == "scheduler":
            return 0

        print(f"Getting size for {repo_id} {component_name}")
        storage_folder = os.path.join(
            self.cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model")
        )

        if not os.path.exists(storage_folder):
            logger.warning(f"Storage folder for {repo_id} not found")
            return 0

        commit_hash = self._get_commit_hash(storage_folder)
        if not commit_hash:
            return 0

        snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
        if component_name:
            snapshot_folder = os.path.join(snapshot_folder, component_name)

        return self._calculate_folder_size(snapshot_folder)

    def _get_commit_hash(self, storage_folder: str) -> Optional[str]:
        """
        Get the commit hash for a repository.

        Args:
            storage_folder: Path to the repository storage folder

        Returns:
            Commit hash string or None if not found
        """
        refs_path = os.path.join(storage_folder, "refs", "main")
        if not os.path.exists(refs_path):
            logger.warning("No commit hash found for {}".format(storage_folder))
            return None
        try:
            with open(refs_path, "r") as f:
                return f.read().strip()
        except Exception as e:
            logger.error("Error reading commit hash: {}".format(str(e)))
            return None

    def _calculate_folder_size(self, folder: str) -> int:
        """
        Calculate the total size of model files in a folder.

        Args:
            folder: Path to the folder

        Returns:
            Total size in bytes
        """
        if not os.path.isdir(folder):
            return os.path.getsize(folder)

        variants = ["bf16", "fp8", "fp16", ""]
        selected_variant = next(
            (v for v in variants if self._check_variant_files(folder, v)), None
        )

        if selected_variant is None:
            return 0

        total_size = 0
        for root, _, files in os.walk(folder):
            for file in files:
                if self._is_valid_file(file, selected_variant):
                    total_size += os.path.getsize(os.path.join(root, file))

        return total_size

    def _check_variant_files(self, folder: str, variant: str) -> bool:
        """
        Check if a folder contains files of a specific variant.

        Args:
            folder: Path to the folder
            variant: Variant to check for

        Returns:
            Boolean indicating if variant files exist
        """
        for root, _, files in os.walk(folder):
            if any(self._is_valid_file(f, variant) for f in files):
                return True
        return False

    def _is_valid_file(self, file: str, variant: str) -> bool:
        """
        Check if a file is a valid model file of a specific variant.

        Args:
            file: Filename to check
            variant: Variant to check for

        Returns:
            Boolean indicating if file is valid
        """
        if variant:
            return file.endswith(f"{variant}.safetensors") or file.endswith(
                f"{variant}.bin"
            )
        return file.endswith((".safetensors", ".bin", ".ckpt"))

    async def _load_diffusers_component(
        self, main_model_repo: str, component_repo: str, component_name: Optional[str] = None
    ) -> Any:
        """
        Load a diffusers component.

        Args:
            component_repo: Repository identifier
            component_name: Name of the component

        Returns:
            Loaded component
        """
        try:
            model_index = (
                await self.hf_model_manager.get_diffusers_multifolder_components(
                    main_model_repo
                )
            )
            if model_index is None:
                raise ValueError(f"model_index does not exist for {main_model_repo}")

            component_info = model_index.get(component_name)
            if not component_info:
                raise ValueError(f"Invalid component info for {component_name}")

            module_path, class_name = component_info
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            if component_name:
                component = model_class.from_pretrained(
                    component_repo,
                    subfolder=component_name,
                    torch_dtype=torch.bfloat16
                    if "flux" in component_repo.lower()
                    else torch.float16,
                )
            else:
                component = model_class.from_pretrained(component_repo, torch_dtype=torch.bfloat16
                    if "flux" in component_repo.lower()
                    else torch.float16,)

            if self.should_quantize:
                quantize_model_fp8(component)

            return component

        except Exception as e:
            logger.error(
                "Error loading component {} from {}: {}".format(
                    component_name, component_repo, str(e)
                )
            )
            raise

    def _load_custom_component(
        self, repo_id: str, category: str, component_name: str
    ) -> Any:
        """
        Load a custom component.

        Args:
            repo_id: Repository identifier
            category: Component category
            component_name: Name of the component

        Returns:
            Loaded component
        """
        try:
            file_path = self._get_component_file_path(repo_id)
            state_dict = load_state_dict_from_file(file_path)

            architectures = get_architectures()
            arch_key = f"core_extension_1.{category.lower()}_{component_name.lower()}"
            architecture_class = architectures.get(arch_key)

            if not architecture_class:
                raise ValueError(f"Architecture not found for {arch_key}")

            architecture = architecture_class()
            architecture.load(state_dict)
            model = architecture.model

            if self.should_quantize:
                quantize_model_fp8(model)

            return model

        except Exception as e:
            logger.error("Error loading custom component: {}".format(str(e)))
            raise

    def _get_component_file_path(self, repo_id: str) -> str:
        """
        Get the file path for a component.

        Args:
            repo_id: Repository identifier

        Returns:
            Path to component file
        """
        repo_folder = os.path.dirname(repo_id.replace("hf:", ""))
        weights_name = repo_id.split("/")[-1]

        model_folder = os.path.join(
            self.cache_dir, repo_folder_name(repo_id=repo_folder, repo_type="model")
        )

        if not os.path.exists(model_folder):
            model_folder = os.path.join(self.cache_dir, repo_folder)
            if not os.path.exists(model_folder):
                raise FileNotFoundError(f"Cache folder for {repo_id} not found")
            return os.path.join(model_folder, weights_name)

        commit_hash = self._get_commit_hash(model_folder) or ""
        return os.path.join(model_folder, "snapshots", commit_hash, weights_name)

import os
import gc
import logging
import importlib
from enum import Enum
from typing import Optional, Any, Dict, List, Tuple, Union
import psutil
from collections import OrderedDict
import time

import torch
from diffusers import (
    DiffusionPipeline,
    FluxInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    FluxPipeline,
)
from diffusers.loaders import FromSingleFileMixin
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name

from ..config import get_config
from ..globals import (
    get_hf_model_manager,
    get_architectures,
    get_available_torch_device,
)
from ..utils.load_models import load_state_dict_from_file
from ..utils.utils import serialize_config
from ..utils.quantize_models import quantize_model_fp8
# from OmniGen import OmniGenPipeline

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class GPUEnum(Enum):
    """GPU memory thresholds in GB for different optimization levels."""
    LOW = 7
    MEDIUM = 14
    HIGH = 22
    VERY_HIGH = 30

# Constants
VRAM_SAFETY_MARGIN_GB = 5.0
RAM_SAFETY_MARGIN_GB = 5.0
VRAM_THRESHOLD = 1.4

MODEL_COMPONENTS = {
    "flux": [
        "vae", "transformer", "text_encoder", "text_encoder_2",
        "scheduler", "tokenizer", "tokenizer_2",
    ],
    "sdxl": [
        "vae", "unet", "text_encoder", "text_encoder_2",
        "scheduler", "tokenizer", "tokenizer_2",
    ],
    "sd": ["vae", "unet", "text_encoder", "scheduler", "tokenizer"],
}

PIPELINE_MAPPING = {
    "sdxl": StableDiffusionXLPipeline,
    "sd": StableDiffusionPipeline,
    "flux": FluxPipeline,
}

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

    def get_lru_models(self, cache_type: str = "gpu", count: Optional[int] = None) -> List[str]:
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
    
    Attributes:
        current_model: Currently active model identifier
        loaded_models: Dictionary of models loaded in GPU memory
        cpu_models: Dictionary of models loaded in CPU memory
        model_sizes: Dictionary tracking model sizes in GB
        model_types: Dictionary tracking model dtype information
        loaded_model: Currently loaded active model
        vram_usage: Current VRAM usage in GB
        ram_usage: Current RAM usage in GB
        is_in_device: Flag indicating if current model is on target device
        should_quantize: Flag indicating if quantization should be applied
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
        self.max_vram: float = self._get_total_vram()
        self.system_ram: float = psutil.virtual_memory().total / (1024**3)
        
        # State flags
        self.is_in_device: bool = False
        self.should_quantize: bool = False
        
        # Managers and caches
        self.hf_model_manager = get_hf_model_manager()
        self.cache_dir = HF_HUB_CACHE
        self.lru_cache = LRUCache()

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
                torch.cuda.get_device_properties(0).total_memory -
                torch.cuda.memory_allocated()
            ) / (1024**3)
            logger.debug(f"Available VRAM: {available_vram_gb:.2f} GB")
            return available_vram_gb
        return 0

    def _can_load_to_ram(self, model_size: float) -> bool:
        """
        Check if a model can be loaded into RAM.
        
        Args:
            model_size: Size of the model in GB
            
        Returns:
            Boolean indicating if model can fit in RAM
        """
        available_ram = self._get_available_ram() - RAM_SAFETY_MARGIN_GB
        logger.debug(f"Available RAM: {available_ram:.2f} GB, Model size: {model_size:.2f} GB")
        return available_ram >= model_size

    def _can_load_model(self, model_size: float) -> Tuple[bool, bool]:
        """
        Check if a model can be loaded to GPU.
        
        Args:
            model_size: Size of the model in GB
            
        Returns:
            Tuple of (can_load_without_optimization, needs_optimization)
        """
        available_vram = self._get_available_vram()

        if not self.loaded_models:
            if model_size <= available_vram - VRAM_SAFETY_MARGIN_GB:
                logger.debug("Can load without optimization")
                return True, False
            elif model_size <= available_vram * VRAM_THRESHOLD:
                logger.debug("Can load with optimization")
                return True, True
            logger.debug("Cannot load even with optimization")
            return False, False

        return available_vram - VRAM_SAFETY_MARGIN_GB >= model_size, False

    def _determine_load_location(self, model_size: float) -> str:
        """
        Determine optimal location to load the model based on available resources.
        
        Args:
            model_size: Size of the model in GB
            
        Returns:
            String indicating load location: "gpu", "gpu_optimized", "cpu", or "none"
        """
        can_load_gpu, need_optimization = self._can_load_model(model_size)
        
        if can_load_gpu:
            return "gpu_optimized" if need_optimization else "gpu"
        if self._can_load_to_ram(model_size):
            return "cpu"
        return "none"

    async def load(
        self,
        model_id: str,
        gpu: Optional[int] = None,
        pipe_type: Optional[str] = None
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
        self,
        model_id: str
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
        self,
        model_id: str,
        pipeline: DiffusionPipeline,
        model_size: float
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
        self,
        model_id: str,
        pipeline: DiffusionPipeline,
        model_size: float
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
        self,
        model_id: str,
        gpu: Optional[int] = None,
        pipe_type: Optional[str] = None
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
            pipe_type: Optional pipeline type specification
            
        Returns:
            Loaded pipeline or None if loading failed
        """
        try:
            config = serialize_config(get_config())
            model_config = config["enabled_models"].get(model_id)

            if not model_config:
                logger.error(f"Model {model_id} not found in configuration")
                return None

            # Prepare memory and determine load location
            estimated_size = self._get_model_size(model_config)
            load_location = self._determine_load_location(estimated_size)

            if load_location in ["none", "cpu", "gpu_optimized"]:
                self._make_space_for_model(estimated_size)
                load_location = self._determine_load_location(estimated_size)
                if load_location == "none":
                    logger.error(f"Insufficient memory to load model {model_id}")
                    return None

            # Load the model
            pipeline = await self._load_model_by_source(model_id, model_config, gpu)
            if pipeline is None:
                return None
            
            

            # Place in appropriate memory location
            return await self._place_model_in_memory(
                pipeline, model_id, estimated_size, load_location
            )

        except Exception as e:
            logger.error(f"Error loading new model {model_id}: {str(e)}")
            return None

    async def _load_model_by_source(
        self,
        model_id: str,
        model_config: Dict[str, Any],
        gpu: Optional[int]
    ) -> Optional[DiffusionPipeline]:
        """
        Load a model based on its source configuration.
        
        Args:
            model_id: Model identifier
            model_config: Model configuration dictionary
            gpu: Optional GPU device number
            
        Returns:
            Loaded pipeline or None if loading failed
        """
        source = model_config["source"]
        prefix, path = source.split(":", 1)
        type = model_config["type"]
        self.should_quantize = model_config.get("quantize", False)

        try:
            if prefix == "hf":
                is_downloaded, variant = self.hf_model_manager.is_downloaded(model_id)
                if not is_downloaded:
                    logger.info(f"Model {model_id} not downloaded")
                    return None
                return await self.load_huggingface_model(
                    model_id, path, gpu, type, variant, model_config
                )
            elif prefix in ["file", "ct"]:
                return await self.load_single_file_model(
                    model_id, path, prefix, gpu, type
                )
            else:
                logger.error(f"Unsupported model source prefix: {prefix}")
                return None
        except Exception as e:
            logger.error(f"Error loading model from source: {str(e)}")
            return None

    def _make_space_for_model(self, model_size: float) -> None:
        """
        Attempt to free up memory space for a new model.
        
        Implements a sophisticated memory management strategy:
        1. First tries to move GPU models to CPU
        2. If needed, unloads models that can't be moved to CPU
        3. Tracks all memory changes through LRU cache
        
        Args:
            model_size: Size of the model requiring space in GB
        """
        available_vram = self._get_available_vram() - VRAM_SAFETY_MARGIN_GB
        need_gpu_space = model_size > available_vram

        if need_gpu_space:
            logger.debug(f"Need to free up GPU space for {model_size:.2f} GB model")
            space_needed = model_size - available_vram

            # Get all GPU models in LRU order
            gpu_models = [
                (mid, self.model_sizes[mid])
                for mid in self.lru_cache.get_lru_models("gpu", None)
            ]

            freed_space = 0
            for model_id, size in gpu_models:
                if freed_space >= space_needed:
                    break

                if self._can_load_to_ram(size):
                    freed_space += self._move_model_to_cpu_for_space(
                        model_id, size
                    )
                else:
                    freed_space += self._unload_model_for_space(
                        model_id, size
                    )

    def _move_model_to_cpu_for_space(
        self,
        model_id: str,
        model_size: float
    ) -> float:
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
        self,
        model_id: str,
        model_size: float
    ) -> float:
        """
        Completely unload a model to free up space.
        
        Args:
            model_id: Model identifier
            model_size: Size of the model in GB
            
        Returns:
            Amount of space freed in GB
        """
        logger.info(f"Unloading {model_id} as it cannot be moved to CPU")
        del self.loaded_models[model_id]
        self.vram_usage -= model_size
        self.lru_cache.remove(model_id, "gpu")
        return model_size

    def _move_model_to_cpu(
        self,
        pipeline: DiffusionPipeline,
        model_id: str
    ) -> bool:
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
            logger.error(f"Failed to move model {model_id} to CPU: {str(e)}")
            return False

    def _store_model_dtype(
        self,
        pipeline: DiffusionPipeline,
        model_id: str
    ) -> None:
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

    def _move_model_to_gpu(
        self,
        pipeline: DiffusionPipeline,
        model_id: str
    ) -> bool:
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
            self.flush_memory()

            # Restore original dtype if known
            original_dtype = self.model_types.get(model_id)
            pipeline = pipeline.to(
                device=device,
                # dtype=original_dtype if original_dtype else torch.float16
            )

            self.flush_memory()
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU out of memory while moving model {model_id} to GPU")
            else:
                logger.error(f"Runtime error moving model {model_id} to GPU: {str(e)}")
            self.flush_memory()
            return False
        
    async def _place_model_in_memory(
        self,
        pipeline: DiffusionPipeline,
        model_id: str,
        model_size: float,
        load_location: str
    ) -> Optional[DiffusionPipeline]:
        """
        Place a loaded model in the appropriate memory location.
        
        Args:
            pipeline: The loaded pipeline
            model_id: Model identifier
            model_size: Size of the model in GB
            load_location: Target location ("gpu", "gpu_optimized", "cpu")
            
        Returns:
            The pipeline in its final location or None if placement failed
        """
        try:
            if pipeline.__class__.__name__ == "OmniGenPipeline":
                # OmniGen manages its own device placement and optimization
                self.loaded_models[model_id] = pipeline
                self.model_sizes[model_id] = model_size
                self.vram_usage += model_size
                self.lru_cache.access(model_id, "gpu")
                return pipeline
            if load_location == "gpu":
                if self._move_model_to_gpu(pipeline, model_id):
                    self.loaded_models[model_id] = pipeline
                    self.model_sizes[model_id] = model_size
                    self.vram_usage += model_size
                    self.lru_cache.access(model_id, "gpu")
                    return pipeline
                logger.error(f"Failed to move model {model_id} to GPU")
                return None

            elif load_location == "gpu_optimized":
                self.apply_optimizations(pipeline, model_id, True)
                self.cpu_models[model_id] = pipeline
                self.model_sizes[model_id] = model_size
                self.ram_usage += model_size
                self.lru_cache.access(model_id, "cpu")
                return pipeline

            elif load_location == "cpu":
                if self._move_model_to_cpu(pipeline, model_id):
                    self.cpu_models[model_id] = pipeline
                    self.model_sizes[model_id] = model_size
                    self.ram_usage += model_size
                    self.lru_cache.access(model_id, "cpu")
                    return pipeline
                logger.error(f"Failed to move model {model_id} to CPU")
                return None

            return None

        except Exception as e:
            logger.error(f"Error placing model {model_id} in memory: {str(e)}")
            return None

    async def load_huggingface_model(
        self,
        model_id: str,
        repo_id: str,
        gpu: Optional[int] = None,
        type: Optional[str] = None,
        variant: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[DiffusionPipeline]:
        """
        Load a model from HuggingFace.
        
        Args:
            model_id: Model identifier
            repo_id: HuggingFace repository ID
            gpu: Optional GPU device number
            type: Optional model type
            variant: Optional model variant
            model_config: Optional model configuration
            
        Returns:
            Loaded pipeline or None if loading failed
        """
        try:
            # Check if custom pipeline is specified in model_config and custom_pipeline is a list
            if "custom_pipeline" in model_config and isinstance(model_config["custom_pipeline"], list):
                print("custom pipeline is a list")
                # using package to import custom pipeline
                module_path, class_name = model_config["custom_pipeline"]
                module = importlib.import_module(module_path)
                pipeline_class = getattr(module, class_name)
            
                pipeline = pipeline_class.from_pretrained(repo_id)

                return pipeline
            
            pipeline_kwargs = await self._prepare_pipeline_kwargs(model_config)

            variant = None if variant == "" else variant

            torch_dtype = (
                torch.bfloat16 if "flux" in model_id.lower() else torch.float16
            )

            pipeline = DiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch_dtype,
                local_files_only=True,
                variant=variant,
                **pipeline_kwargs,
            )

            self.flush_memory()
            self.loaded_model = pipeline
            self.current_model = model_id
            logger.info(f"Model {model_id} loaded successfully")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            return None

    async def _prepare_pipeline_kwargs(
        self,
        model_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Prepare kwargs for pipeline initialization.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Dictionary of pipeline initialization arguments
        """
        if not model_config:
            return {}

        pipeline_kwargs = {}
        try:
            if "components" in model_config and model_config["components"]:
                for key, component in model_config["components"].items():
                    if isinstance(component, dict) and "source" in component:
                        pipeline_kwargs[key] = await self._prepare_component(
                            component, model_config["type"], key
                        )
                    elif component.get("source") is None:
                        pipeline_kwargs[key] = None

            if "custom_pipeline" in model_config:
                pipeline_kwargs["custom_pipeline"] = model_config["custom_pipeline"]

            return pipeline_kwargs

        except Exception as e:
            logger.error(f"Error preparing pipeline kwargs: {str(e)}")
            return {}

    async def _prepare_component(
        self,
        component: Dict[str, Any],
        model_type: str,
        key: str
    ) -> Any:
        """
        Prepare a model component based on its configuration.
        
        Args:
            component: Component configuration
            model_type: Type of the model
            key: Component key
            
        Returns:
            Loaded component or None if loading failed
        """
        try:
            if not component["source"].endswith(
                (".safetensors", ".bin", ".ckpt", ".pt")
            ):
                return await self._load_diffusers_component(
                    component["source"].replace("hf:", ""),
                    key
                )
            else:
                return self._load_custom_component(
                    component["source"],
                    model_type,
                    key
                )
        except Exception as e:
            logger.error(f"Error preparing component {key}: {str(e)}")
            return None

    async def load_single_file_model(
        self,
        model_id: str,
        path: str,
        prefix: str,
        gpu: Optional[int] = None,
        type: Optional[str] = None,
    ) -> Optional[DiffusionPipeline]:
        """
        Load a model from a single file.
        
        Args:
            model_id: Model identifier
            path: Path to model file
            prefix: Source prefix (file/ct)
            gpu: Optional GPU device number
            type: Model type
            
        Returns:
            Loaded pipeline or None if loading failed
        """
        logger.info(f"Loading single file model {model_id}")

        if type is None:
            logger.error("Model type must be specified for single file models")
            return None

        pipeline_class = PIPELINE_MAPPING.get(type)
        if not pipeline_class:
            logger.error(f"Unsupported model type: {type}")
            return None

        try:
            model_path = self._get_model_path(path, prefix)
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None

            pipeline = await self._load_pipeline_from_file(
                pipeline_class, model_path, model_id, type
            )
            if pipeline:
                self.loaded_model = pipeline
                self.current_model = model_id
            return pipeline

        except Exception as e:
            logger.error(f"Error loading single file model: {str(e)}")
            return None
    def _get_model_path(self, path: str, prefix: str) -> str:
        if prefix == "ct" and ("http" in path or "https" in path):
            path = path.split("/")[-1]
        return os.path.join(get_config().models_path, path)


    async def _load_pipeline_from_file(
        self,
        pipeline_class: Any,
        model_path: str,
        model_id: str,
        type: str
    ) -> Optional[DiffusionPipeline]:
        """
        Load a pipeline from a file using appropriate loading method.
        
        Args:
            pipeline_class: Class to instantiate pipeline
            model_path: Path to model file
            model_id: Model identifier
            type: Model type
            
        Returns:
            Loaded pipeline or None if loading failed
        """
        try:
            if issubclass(pipeline_class, FromSingleFileMixin):
                return self._load_from_single_file(
                    pipeline_class, model_path, model_id
                )
            else:
                return self._load_custom_architecture(
                    pipeline_class, model_path, type
                )
        except Exception as e:
            logger.error(f"Error loading pipeline from file: {str(e)}")
            return None


    def _load_from_single_file(
        self,
        pipeline_class: Any,
        path: str,
        model_id: str
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
                torch.bfloat16 
                if "flux" in model_id.lower() 
                else torch.float16
            )
            
            # Load the model with appropriate dtype
            pipeline = pipeline_class.from_single_file(
                path,
                torch_dtype=torch_dtype
            )
            
            logger.info(f"Successfully loaded single file model {model_id}")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error loading model from single file: {str(e)}")
            raise

    def _load_custom_architecture(
        self,
        pipeline_class: Any,
        path: str,
        type: str
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
                        f"Error loading component {component_name}: {str(component_error)}"
                    )
                    continue
            
            logger.info("Successfully loaded custom architecture model")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error loading custom architecture: {str(e)}")
            raise

    def apply_optimizations(
        self,
        pipeline: DiffusionPipeline,
        model_id: str,
        force_full_optimization: bool = False
    ) -> None:
        """
        Apply memory and performance optimizations to a pipeline.
        
        Args:
            pipeline: The pipeline to optimize
            model_id: Model identifier
            force_full_optimization: Whether to force full optimizations
        """
        # Skip if it's OmniGen pipeline
        if pipeline.__class__.__name__ == "OmniGenPipeline":
            return
        
        if self.loaded_model is not None and self.is_in_device:
            logger.info(f"Model {model_id} already optimized")
            return

        device = get_available_torch_device()
        config = serialize_config(get_config())
        model_config = config["enabled_models"][model_id]

        model_size_gb = self._get_model_size(model_config)
        available_vram_gb = self._get_available_vram()

        optimizations, force_full_optimization = self._determine_optimizations(
            available_vram_gb, model_size_gb, force_full_optimization
        )

        self._apply_optimization_list(pipeline, optimizations, device)

        if not force_full_optimization:
            if self._move_model_to_gpu(pipeline, model_id):
                pass
            else:
                logger.error(f"Failed to move model {model_id} to GPU")

    def _determine_optimizations(
        self,
        available_vram_gb: float,
        model_size_gb: float,
        force_full_optimization: bool
    ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], bool]:
        """
        Determine which optimizations to apply based on available resources.
        
        Args:
            available_vram_gb: Available VRAM in GB
            model_size_gb: Model size in GB
            force_full_optimization: Whether to force full optimizations
            
        Returns:
            Tuple of (list of optimizations, whether full optimization is needed)
        """
        optimizations = []

        if self.should_quantize and available_vram_gb >= GPUEnum.HIGH.value:
            force_full_optimization = False
        elif available_vram_gb >= GPUEnum.VERY_HIGH.value:
            force_full_optimization = False
        elif available_vram_gb > model_size_gb:
            if force_full_optimization and available_vram_gb < GPUEnum.HIGH.value:
                optimizations = self._get_full_optimizations()
                force_full_optimization = True
        else:
            if not (self.should_quantize and available_vram_gb >= GPUEnum.HIGH.value):
                optimizations = self._get_full_optimizations()
                force_full_optimization = True

        return optimizations, force_full_optimization

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
        device: torch.device
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
                logger.info(f"{opt_name} enabled")
            except Exception as e:
                logger.error(f"Error enabling {opt_name}: {str(e)}")

        if device_type == "mps":
            delattr(torch, "mps")

    def flush_memory(self) -> None:
        """Clear unused memory from GPU and perform garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            pass  # MPS doesn't need explicit cache clearing
        gc.collect()

    async def warm_up_pipeline(self, model_id: str) -> None:
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


        # pipeline = self.loaded_models[model_id]
        if pipeline is None:
            logger.warning(f"Failed to load model {model_id} for warm-up")
            return

        logger.info(f"Warming up pipeline for model {model_id}")
        
        try:
            with torch.no_grad():
                if isinstance(pipeline, DiffusionPipeline):
                    _ = pipeline(
                        prompt="This is a warm-up prompt",
                        num_inference_steps=20,
                        output_type="pil"
                    )
                else:
                    logger.warning(
                        f"Unsupported pipeline type for warm-up: {type(pipeline)}"
                    )
        except Exception as e:
            logger.error(f"Error during warm-up for model {model_id}: {str(e)}")

        self.flush_memory()
        logger.info(f"Warm-up completed for model {model_id}")

    def get_all_model_ids(self) -> List[str]:
        """
        Get list of all available model IDs.
        
        Returns:
            List of model identifiers
        """
        config = serialize_config(get_config())
        return list(config["enabled_models"].keys())

    def get_warmup_models(self) -> List[str]:
        """
        Get list of models that should be warmed up.
        
        Returns:
            List of model identifiers for warm-up
        """
        config = serialize_config(get_config())
        return config["warmup_models"]

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
        return (model_id == self.current_model and 
                self.loaded_model is not None)

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

    def _get_model_size(self, model_config: Dict[str, Any]) -> float:
        """
        Calculate the total size of a model including all its components.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Total size in GB
        """
        if any(prefix in model_config["source"] for prefix in ["ct:", "file:"]):
            path = model_config["source"].replace("ct:", "").replace("file:", "")
            return os.path.getsize(path) / (1024**3)

        repo_id = model_config["source"].replace("hf:", "")
        return self._calculate_repo_size(repo_id, model_config)

    def _calculate_repo_size(
        self,
        repo_id: str,
        model_config: Dict[str, Any]
    ) -> float:
        """
        Calculate the total size of a HuggingFace repository.
        
        Args:
            repo_id: Repository identifier
            model_config: Model configuration dictionary
            
        Returns:
            Total size in GB
        """
        total_size = self._get_size_for_repo(repo_id)

        if "components" in model_config and model_config["components"]:
            for key, component in model_config["components"].items():
                if isinstance(component, dict) and "source" in component:
                    component_size = self._calculate_component_size(
                        component, repo_id, key
                    )
                    total_size += component_size

        total_size_gb = total_size / (1024**3)
        logger.debug(f"Total size: {total_size_gb:.2f} GB")
        return total_size_gb

    def _calculate_component_size(
        self,
        component: Dict[str, Any],
        repo_id: str,
        key: str
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
            component_repo = "/".join(
                component_source.split("/")[0:2]
            ).replace("hf:", "")
        else:
            component_repo = component_source.replace("hf:", "")

        component_name = (
            key
            if not component_source.endswith(
                (".safetensors", ".bin", ".ckpt", ".pt")
            )
            else component_source.split("/")[-1]
        )

        total_size = (
            self._get_size_for_repo(component_repo, component_name) -
            self._get_size_for_repo(repo_id, key)
        )
        
        return total_size

    def _get_size_for_repo(
        self,
        repo_id: str,
        component_name: Optional[str] = None
    ) -> int:
        """
        Get the size of a specific repository or component.
        
        Args:
            repo_id: Repository identifier
            component_name: Optional component name
            
        Returns:
            Size in bytes
        """
        storage_folder = os.path.join(
            self.cache_dir,
            repo_folder_name(repo_id=repo_id, repo_type="model")
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
            logger.warning(f"No commit hash found for {storage_folder}")
            return None
        try:
            with open(refs_path, "r") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading commit hash: {str(e)}")
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
            (v for v in variants if self._check_variant_files(folder, v)),
            None
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
            return (file.endswith(f"{variant}.safetensors") or
                   file.endswith(f"{variant}.bin"))
        return file.endswith((".safetensors", ".bin", ".ckpt"))

    async def _load_diffusers_component(
        self,
        component_repo: str,
        component_name: str
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
                    component_repo
                )
            )
            component_info = model_index.get(component_name)
            if not component_info:
                raise ValueError(f"Invalid component info for {component_name}")

            module_path, class_name = component_info
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            component = model_class.from_pretrained(
                component_repo,
                subfolder=component_name,
                torch_dtype=torch.bfloat16
                if "flux" in component_repo.lower()
                else torch.float16,
            )

            if self.should_quantize:
                quantize_model_fp8(component)

            return component

        except Exception as e:
            logger.error(
                f"Error loading component {component_name} from {component_repo}: {str(e)}"
            )
            raise

    def _load_custom_component(
        self,
        repo_id: str,
        category: str,
        component_name: str
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
            logger.error(f"Error loading custom component: {str(e)}")
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
            self.cache_dir,
            repo_folder_name(repo_id=repo_folder, repo_type="model")
        )

        if not os.path.exists(model_folder):
            model_folder = os.path.join(self.cache_dir, repo_folder)
            if not os.path.exists(model_folder):
                raise FileNotFoundError(f"Cache folder for {repo_id} not found")
            return os.path.join(model_folder, weights_name)

        commit_hash = self._get_commit_hash(model_folder)
        return os.path.join(model_folder, "snapshots", commit_hash, weights_name)
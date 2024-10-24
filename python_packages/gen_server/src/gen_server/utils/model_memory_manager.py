import os
import gc
import logging
import importlib
from enum import Enum
from typing import Optional, Any, Dict
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

logger = logging.getLogger(__name__)


class GPUEnum(Enum):
    LOW = 7
    MEDIUM = 14
    HIGH = 22
    VERY_HIGH = 30


VRAM_SAFETY_MARGIN_GB = 4.0

MODEL_COMPONENTS = {
    "flux": [
        "vae",
        "transformer",
        "text_encoder",
        "text_encoder_2",
        "scheduler",
        "tokenizer",
        "tokenizer_2",
    ],
    "sdxl": [
        "vae",
        "unet",
        "text_encoder",
        "text_encoder_2",
        "scheduler",
        "tokenizer",
        "tokenizer_2",
    ],
    "sd": ["vae", "unet", "text_encoder", "scheduler", "tokenizer"],
}

PIPELINE_MAPPING = {
    "sdxl": StableDiffusionXLPipeline,
    "sd": StableDiffusionPipeline,
    "flux": FluxPipeline,
}



class LRUCache:
    """LRU Cache for tracking model usage"""
    def __init__(self):
        self.gpu_cache = OrderedDict()  # model_id -> last_used_timestamp
        self.cpu_cache = OrderedDict()  # model_id -> last_used_timestamp

    def access(self, model_id: str, cache_type: str = "gpu"):
        """Record access of a model"""
        cache = self.gpu_cache if cache_type == "gpu" else self.cpu_cache
        cache.pop(model_id, None)  # Remove if exists
        cache[model_id] = time.time()  # Add to end (most recently used)

    def remove(self, model_id: str, cache_type: str = "gpu"):
        """Remove a model from cache tracking"""
        cache = self.gpu_cache if cache_type == "gpu" else self.cpu_cache
        cache.pop(model_id, None)

    def get_lru_models(self, cache_type: str = "gpu", count: int = 1) -> list[str]:
        """Get least recently used models"""
        cache = self.gpu_cache if cache_type == "gpu" else self.cpu_cache
        return list(cache.keys())[:count]  # First items are least recently used



class ModelMemoryManager:
    def __init__(self):
        self.current_model: Optional[str] = None
        self.loaded_models: Dict[str, DiffusionPipeline] = {}
        self.cpu_models: Dict[str, DiffusionPipeline] = {}

        self.model_sizes: Dict[str, int] = {}
        self.hf_model_manager = get_hf_model_manager()
        self.cache_dir = HF_HUB_CACHE
        self.is_in_device = False
        self.should_quantize = False
        self.vram_usage = 0
        self.max_vram = self._get_total_vram()
        self.vram_buffer = VRAM_SAFETY_MARGIN_GB
        self.VRAM_THRESHOLD = 1.4

        # LRU Cache
        self.lru_cache = LRUCache()

        # system RAM
        self.system_ram = psutil.virtual_memory().total / (1024**3)
        self.ram_buffer = 4.0
        self.ram_usage = 0

        self.model_types = {}



    def _get_available_ram(self) -> float:
        """Get the available RAM in GB"""
        return psutil.virtual_memory().available / (1024**3)

    def _get_total_vram(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0  # Default to 0 for non-CUDA devices

    def _get_available_vram(self) -> float:
        if torch.cuda.is_available():
            available_vram_gb = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            ) / (1024**3)
            print(f"available vram: {available_vram_gb}")
            return available_vram_gb
        return 0  # Default to 0 for non-CUDA devices
    

    def _can_load_to_ram(self, model_size: int) -> bool:
        """Check if the model can be loaded to RAM"""
        available_ram = self._get_available_ram() - self.ram_buffer
        # available_ram = self._get_available_ram()
        print(f"available_ram: {available_ram}, model_size: {model_size}")
        return available_ram >= model_size
    

    def _can_load_model(self, model_size: float) -> tuple[bool, bool]:
        available_vram = self._get_available_vram()

        if not self.loaded_models:
            if model_size <= available_vram - self.vram_buffer:
                print("can load without full optimization")
                return True, False  # Can load without full optimization
            elif model_size <= available_vram * self.VRAM_THRESHOLD:
                print("can load with full optimization")
                return True, True  # Can load with full optimization
            else:
                print("cannot load even with full optimization")
                return False, False  # Cannot load even with full optimization

        return available_vram - self.vram_buffer >= model_size, False
    
    
    def _determine_load_location(self, model_size: float) -> str:
        """Determine where to load the model based on available resources"""
        can_load_gpu, need_optimization = self._can_load_model(model_size)

        if can_load_gpu:
            return "gpu" if not need_optimization else "gpu_optimized"
        
        if self._can_load_to_ram(model_size):
            return "cpu"
        
        return "none"

    def _get_model_size(self, model_config: dict[str, Any]) -> float:
        if "ct:" in model_config["source"] or "file:" in model_config["source"]:
            return os.path.getsize(
                model_config["source"].replace("ct:", "").replace("file:", "")
            ) / (1024**3)

        repo_id = model_config["source"].replace("hf:", "")
        return self._calculate_repo_size(repo_id, model_config)

    def _calculate_repo_size(self, repo_id: str, model_config: dict[str, Any]) -> float:
        total_size = self._get_size_for_repo(repo_id)

        if "components" in model_config and model_config["components"]:
            for key, component in model_config["components"].items():
                if isinstance(component, dict) and "source" in component:
                    if len(component["source"].split("/")) > 2:
                        component_repo = "/".join(
                            component["source"].split("/")[0:2]
                        ).replace("hf:", "")
                    else:
                        component_repo = component["source"].replace("hf:", "")
                    component_name = (
                        key
                        if not component["source"].endswith(
                            (".safetensors", ".bin", ".ckpt", ".pt")
                        )
                        else component["source"].split("/")[-1]
                    )
                    component_size = self._get_size_for_repo(
                        component_repo, component_name
                    )
                    total_size += component_size
                    total_size -= self._get_size_for_repo(repo_id, key)

        # Convert total_size to GB
        total_size_gb = total_size / (1024**3)
        print(f"total size: {total_size_gb}")

        return total_size_gb

    def _get_size_for_repo(
        self, repo_id: str, component_name: Optional[str] = None
    ) -> int:
        storage_folder = os.path.join(
            self.cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model")
        )
        if not os.path.exists(storage_folder):
            logger.warning(f"Storage folder for {repo_id} not found.")
            return 0

        commit_hash = self._get_commit_hash(storage_folder)
        if not commit_hash:
            return 0

        snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
        if component_name:
            snapshot_folder = os.path.join(snapshot_folder, component_name)

        return self._calculate_folder_size(snapshot_folder)

    def _get_commit_hash(self, storage_folder: str) -> Optional[str]:
        refs_path = os.path.join(storage_folder, "refs", "main")
        if not os.path.exists(refs_path):
            logger.warning(f"No commit hash found for {storage_folder}")
            return None
        with open(refs_path, "r") as f:
            return f.read().strip()

    def _calculate_folder_size(self, folder: str) -> int:
        if not os.path.isdir(folder):
            return os.path.getsize(folder)

        variants = ["bf16", "fp8", "fp16", ""]
        selected_variant = next(
            (v for v in variants if self._check_variant_files(folder, v)), None
        )

        total_size = 0
        if selected_variant is not None:
            for root, _, files in os.walk(folder):
                for file in files:
                    if self._is_valid_file(file, selected_variant):
                        total_size += os.path.getsize(os.path.join(root, file))

        return total_size

    def _check_variant_files(self, folder: str, variant: str) -> bool:
        for root, _, files in os.walk(folder):
            for file in files:
                if self._is_valid_file(file, variant):
                    return True
        return False

    def _is_valid_file(self, file: str, variant: str) -> bool:
        if variant:
            return file.endswith(f"{variant}.safetensors") or file.endswith(
                f"{variant}.bin"
            )
        return file.endswith((".safetensors", ".bin", ".ckpt"))

    def get_all_model_ids(self) -> list[str]:
        config = serialize_config(get_config())
        return list(config["enabled_models"].keys())

    def get_warmup_models(self) -> list[str]:
        config = serialize_config(get_config())
        return config["warmup_models"]

    async def warm_up_pipeline(self, model_id: str):
        if model_id not in self.loaded_models:
            logger.warning(f"Model {model_id} is not loaded")
            return

        pipeline = self.loaded_models[model_id]
        if pipeline is None:
            logger.warning(f"Failed to load model {model_id} for warm-up")
            return

        logger.info(f"Warming up pipeline for model {model_id}")
        dummy_prompt = "This is a warm-up prompt"

        try:
            with torch.no_grad():
                if isinstance(pipeline, (DiffusionPipeline)):
                    _ = pipeline(
                        prompt=dummy_prompt, num_inference_steps=20, output_type="pil"
                    )
                else:
                    logger.warning(
                        f"Unsupported pipeline type for warm-up: {type(pipeline)}"
                    )
        except Exception as e:
            logger.error(f"Error during warm-up for model {model_id}: {str(e)}")

        self.flush_memory()

        logger.info(f"Warm-up completed for model {model_id}")


    async def load(
        self, model_id: str, gpu: Optional[int] = None, pipe_type: Optional[str] = None
    ) -> Optional[DiffusionPipeline]:
        logger.info(f"Loading model {model_id}")

        # Check if already in GPU
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} is already loaded.")
            self.lru_cache.access(model_id, "gpu")
            return self.loaded_models[model_id]

        # Check if in CPU
        if model_id in self.cpu_models:
            logger.info(f"Model {model_id} is already loaded to CPU.")
            
            # Try to move from CPU to GPU
            model_size = self.model_sizes[model_id]
            available_vram = self._get_available_vram() - self.vram_buffer

            # If we need more space, try to free up VRAM
            if model_size > available_vram:
                # Calculate how much space we need
                needed_space = model_size - available_vram

                # Get candidate for removal from GPU by LRU
                gpu_models = [
                    (mid, self.model_sizes[mid])
                    for mid in self.lru_cache.get_lru_models("gpu")
                ]

                # Try to free up enough space
                freed_space = 0
                models_to_move = []

                for gpu_model_id, gpu_model_size in gpu_models:
                    if freed_space >= needed_space:
                        break
                    models_to_move.append(gpu_model_id)
                    freed_space += gpu_model_size

                # Move selected models to CPU if possible
                for gpu_model_id in models_to_move:
                    moved_size = self.model_sizes[gpu_model_id]
                    if self._can_load_to_ram(moved_size):
                        logger.info(f"Moving {gpu_model_id} to CPU to make space")
                        pipeline = self.loaded_models[gpu_model_id]
                        available_vram = self._get_available_vram()
                        if self._move_model_to_cpu(pipeline, gpu_model_id):
                            available_vram = self._get_available_vram()
                            self.cpu_models[gpu_model_id] = pipeline
                            del self.loaded_models[gpu_model_id]
                            self.flush_memory()
                            self.vram_usage -= moved_size
                            self.ram_usage += moved_size
                            self.lru_cache.remove(gpu_model_id, "gpu")
                            self.lru_cache.access(gpu_model_id, "cpu")
                    else:
                        # If we can't move to CPU, unload completely
                        logger.info(f"Unloading {gpu_model_id} as it cannot be moved to CPU")
                        # print(f"unloading {self.loaded_models}")
                        pipeline = self.loaded_models[gpu_model_id]

                        pipeline = pipeline.to("cpu", silence_dtype_warnings=True)

                        del pipeline

                        del self.loaded_models[gpu_model_id]
                        # print(f"unloaded {self.loaded_models}")
                        self.flush_memory()
                        self.vram_usage -= moved_size
                        self.lru_cache.remove(gpu_model_id, "gpu")

            # Check if we can move the model to GPU now
            available_vram = self._get_available_vram() - self.vram_buffer
            can_load_gpu, need_optimization = self._can_load_model(model_size)

            if can_load_gpu:
                logger.info(f"Moving {model_id} from CPU to GPU")

                device = get_available_torch_device()
                pipeline = self.cpu_models.pop(model_id)  # Remove from cpu_models first
                self.ram_usage -= model_size

                if need_optimization:
                    self.apply_optimizations(pipeline, model_id, True)
                    # return pipeline
                else:
                    if self._move_model_to_gpu(pipeline, model_id):

                        self.loaded_models[model_id] = pipeline
                        self.vram_usage += model_size
                        # del self.cpu_models[model_id]

                        self.flush_memory()
                        # self.ram_usage -= model_size
                        self.lru_cache.remove(model_id, "cpu")
                        self.lru_cache.access(model_id, "gpu")
                        logger.info(f"Model {model_id} moved to GPU")
                        return pipeline
                    else:
                        # If GPU move failed, put back in CPU models
                        self.cpu_models[model_id] = pipeline
                        self.ram_usage += model_size
            
                        logger.info(f"Model {model_id} is too big to move to GPU or apply optimizations (like cpu offloading). Try quantizing to reduce size.\nGen server will still try to run inference on CPU but might be really slow.")
                        return self.cpu_models[model_id]

        self.is_in_device = False
        config = serialize_config(get_config())
        model_config = config["enabled_models"].get(model_id)

        if not model_config:
            logger.error(f"Model {model_id} not found in configuration.")
            return None

        estimated_size = self._get_model_size(model_config)
        load_location = self._determine_load_location(estimated_size)

        print(f"load_location (first): {load_location}")


        if load_location == "none" or load_location == "cpu" or load_location == "gpu_optimized":
            # Try to make space in both GPU and CPU
            self._make_space_for_model(estimated_size)
            load_location = self._determine_load_location(estimated_size)
            print(f"load_location (after make space): {load_location}")
            if load_location == "none":
                logger.error(f"Not enough space to load model {model_id}")
                return None


        source = model_config["source"]
        prefix, path = source.split(":", 1)
        type = model_config["type"]
        self.should_quantize = model_config.get("quantize", False)

        try:
            pipeline = None
            if prefix == "hf":
                is_downloaded, variant = self.hf_model_manager.is_downloaded(model_id)
                if not is_downloaded:
                    logger.info(
                        f"Model {model_id} not downloaded. Please ensure the model is downloaded first."
                    )
                    return None
                pipeline = await self.load_huggingface_model(
                    model_id, path, gpu, type, variant, model_config
                )
            elif prefix in ["file", "ct"]:
                pipeline = await self.load_single_file_model(
                    model_id, path, prefix, gpu, type
                )
            else:
                logger.error(f"Unsupported model source prefix: {prefix}")
                return None
            
            if pipeline is None:
                logger.error(f"Failed to load model {model_id}")
                return None
            
            # Place in appropriate memory location
            if load_location == "gpu":
                device = get_available_torch_device()
                self._move_model_to_device(pipeline, device)
                self.loaded_models[model_id] = pipeline
                self.model_sizes[model_id] = estimated_size
                self.vram_usage += estimated_size
                self.lru_cache.access(model_id, "gpu")
            elif load_location == "gpu_optimized":
                self.apply_optimizations(pipeline, model_id, True)
                self.cpu_models[model_id] = pipeline
                self.model_sizes[model_id] = estimated_size
                self.ram_usage += estimated_size
                self.lru_cache.access(model_id, "cpu")
            elif load_location == "cpu":
                pipeline =pipeline.to("cpu")
                self.cpu_models[model_id] = pipeline
                self.model_sizes[model_id] = estimated_size
                self.ram_usage += estimated_size
                self.lru_cache.access(model_id, "cpu")

            # if pipeline is not None:
            #     self.loaded_models[model_id] = pipeline
            #     self.model_sizes[model_id] = estimated_size
            #     self.vram_usage += estimated_size
            #     self.apply_optimizations(pipeline, model_id, force_full_optimization)

            return pipeline
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            return None


    def _move_model_to_cpu(self, pipeline: DiffusionPipeline, model_id: str) -> bool:
        """Safely move model to CPU with proper dtype handling"""
        try:
            # Clear CUDA cache before moving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # Store original dtype if not already stored
            if model_id not in self.model_types:
                if hasattr(pipeline, "dtype"):
                    self.model_types[model_id] = pipeline.dtype
                else:
                    # Try to get dtype from a model component
                    for attr in ["vae", "unet", "text_encoder", "transformer"]:
                        if hasattr(pipeline, attr):
                            component = getattr(pipeline, attr)
                            if hasattr(component, "dtype"):
                                self.model_types[model_id] = component.dtype
                                break

            pipeline = pipeline.to(device="cpu", silence_dtype_warnings=True)

            # Force synchronization and cleanup
            self.flush_memory()

            return True
        except Exception as e:
            logger.error(f"Failed to move model {model_id} to CPU: {str(e)}")
            return False
        
    def _move_model_to_gpu(self, pipeline: DiffusionPipeline, model_id: str) -> bool:
        """Safely move model to GPU with proper dtype handling"""
        try:
            device = get_available_torch_device()

            # Clear GPU memory first
            self.flush_memory()

            # Restore original dtype if known
            original_dtype = self.model_types.get(model_id)
            if original_dtype:
                pipeline = pipeline.to(device=device, dtype=original_dtype)
            else:
                pipeline = pipeline.to(device=device, dtype=torch.float16)

            # Force cleanup
            self.flush_memory()

            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU out of memory while moving model {model_id} to GPU")
                self.flush_memory()
            else:
                logger.error(f"Runtime error moving model {model_id} to GPU: {str(e)}")
            return False

    def _make_space_for_model(self, model_size: float):
        """Try to make space for a model by unloading others"""

        # First, calculate how much space we need
        available_vram = self._get_available_vram() - self.vram_buffer
        need_gpu_space = model_size > available_vram

        print(f"model_size: {model_size}, available_vram: {available_vram}")

        if need_gpu_space:
            print("need_gpu_space")
            print(self.model_sizes)
            print(self.lru_cache.get_lru_models("gpu"))
            # Calculate how much GPU space we need to free
            space_needed_in_gpu = model_size - available_vram

            # Try to move GPU models to CPU
            gpu_models = [
                (mid, self.model_sizes[mid])
                for mid in self.lru_cache.get_lru_models("gpu")
            ]

            print(f"gpu_models: {gpu_models}")

            freed_space = 0

            for model_id, size in gpu_models:
                print(f"model_id: {model_id}, size: {size}")
                if freed_space >= space_needed_in_gpu:
                    break

                # If we can move this model to CPU, do it
                if self._can_load_to_ram(size):
                    print("can_load_to_ram")
                    logger.info(f"Moving {model_id} to CPU to make space in the GPU")
                    pipeline = self.loaded_models.pop(model_id)  # Remove from loaded_models first
                    self.vram_usage -= size
                    
                    # pipeline.to(torch.float)
                    if self._move_model_to_cpu(pipeline, model_id):
                        self.cpu_models[model_id] = pipeline
                        # del self.loaded_models[model_id]
                        # self.vram_usage -= size
                        self.ram_usage += size
                        self.lru_cache.remove(model_id, "gpu")
                        self.lru_cache.access(model_id, "cpu")
                        freed_space += size
                    else:
                        # If conversion to float32 failed, unload completely
                        logger.info(f"Failed to move {model_id} to CPU, unloading instead")
                        # del self.loaded_models[model_id]
                        # self.vram_usage -= size
                        del pipeline
                        self.lru_cache.remove(model_id, "gpu")
                        freed_space += size
                else:
                    print("cannot_load_to_ram")
                    # If we can't move to CPU, unload completely
                    logger.info(f"Unloading {model_id} as it cannot be moved to CPU")
                    del self.loaded_models[model_id]
                    self.vram_usage -= size
                    self.lru_cache.remove(model_id, "gpu")
                    freed_space += size

        # Now check if we need CPU space
        # available_ram = self._get_available_ram() - self.ram_buffer
        # if model_size > available_ram:
        #     needed_size = model_size - available_ram
        #     cpu_models = [
        #         (mid, self.model_sizes[mid])
        #         for mid in self.lru_cache.get_lru_models("cpu")
        #     ]

        #     freed_space = 0

        #     for model_id, size in cpu_models:
        #         if freed_space >= needed_size:
        #             break

        #         logger.info(f"Unloading {model_id} to make space in the CPU")
        #         del self.loaded_models[model_id]
        #         self.ram_usage -= size
        #         self.lru_cache.remove(model_id, "cpu")
        #         freed_space += size



    async def load_huggingface_model(
        self,
        model_id: str,
        repo_id: str,
        gpu: Optional[int] = None,
        type: Optional[str] = None,
        variant: Optional[str] = None,
        model_config: Optional[dict[str, Any]] = None,
    ) -> Optional[DiffusionPipeline]:
        try:
            pipeline_kwargs = await self._prepare_pipeline_kwargs(model_config)
            variant = None if variant == "" else variant

            # pipeline_class = FluxInpaintPipeline if "flux" in model_id.lower() and type == "inpaint" else DiffusionPipeline
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
            logger.info(f"Model {model_id} loaded successfully.")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            return None

    async def _prepare_pipeline_kwargs(self, model_config: dict[str, Any]) -> dict:
        pipeline_kwargs = {}
        if "components" in model_config and model_config["components"]:
            for key, component in model_config["components"].items():
                if (
                    isinstance(component, dict)
                    and "source" in component
                    and isinstance(component["source"], str)
                ):
                    if not component["source"].endswith(
                        (".safetensors", ".bin", ".ckpt", ".pt")
                    ):
                        component_name = key
                        component_repo = component["source"].replace("hf:", "")
                        pipeline_kwargs[key] = await self._load_diffusers_component(
                            component_repo, component_name
                        )
                    else:
                        pipeline_kwargs[key] = self._load_custom_component(
                            component["source"], model_config["type"], key
                        )
                elif component.get("source") is None:
                    pipeline_kwargs[key] = None

        if "custom_pipeline" in model_config:
            pipeline_kwargs["custom_pipeline"] = model_config["custom_pipeline"]

        return pipeline_kwargs

    async def load_single_file_model(
        self,
        model_id: str,
        path: str,
        prefix: str,
        gpu: Optional[int] = None,
        type: Optional[str] = None,
    ) -> Optional[DiffusionPipeline]:
        logger.info(f"Loading single file model {model_id}")

        if type is None:
            logger.error("Model type must be specified for single file models.")
            return None

        pipeline_class = PIPELINE_MAPPING.get(type)
        if not pipeline_class:
            logger.error(f"Unsupported model type: {type}")
            return None

        path = self._get_model_path(path, prefix)
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return None

        try:
            if issubclass(pipeline_class, FromSingleFileMixin):
                pipeline = self._load_from_single_file(pipeline_class, path, model_id)
            else:
                pipeline = self._load_custom_architecture(pipeline_class, path, type)

            self.loaded_model = pipeline
            self.current_model = model_id
            return pipeline
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    def _get_model_path(self, path: str, prefix: str) -> str:
        if prefix == "ct" and ("http" in path or "https" in path):
            path = path.split("/")[-1]
        return os.path.join(get_config().models_path, path)

    def _load_from_single_file(
        self, pipeline_class, path: str, model_id: str
    ) -> DiffusionPipeline:
        torch_dtype = torch.bfloat16 if "flux" in model_id.lower() else torch.float16
        return pipeline_class.from_single_file(path, torch_dtype=torch_dtype)

    def _load_custom_architecture(
        self, pipeline_class, path: str, type: str
    ) -> DiffusionPipeline:
        state_dict = load_state_dict_from_file(path)
        pipeline = pipeline_class()

        for component_name in MODEL_COMPONENTS[type]:
            if component_name in ["scheduler", "tokenizer", "tokenizer_2"]:
                continue

            arch_key = f"core_extension_1.{type}_{component_name}"
            architecture_class = get_architectures().get(arch_key)

            if not architecture_class:
                logger.error(f"Architecture not found for {arch_key}")
                continue

            architecture = architecture_class()
            architecture.load(state_dict)
            setattr(pipeline, component_name, architecture.model)

        return pipeline

    async def _load_diffusers_component(
        self, component_repo: str, component_name: str
    ) -> Any:
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
        self, repo_id: str, category: str, component_name: str
    ) -> Any:
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

    def _get_component_file_path(self, repo_id: str) -> str:
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

        commit_hash = self._get_commit_hash(model_folder)
        return os.path.join(model_folder, "snapshots", commit_hash, weights_name)

    def apply_optimizations(
        self,
        pipeline: DiffusionPipeline,
        model_id: str,
        force_full_optimization: bool = False,
    ):
        if self.loaded_model is not None and self.is_in_device:
            logger.info(
                f"Model {model_id} is already loaded in memory and in device. Not applying optimizations."
            )
            return

        device = get_available_torch_device()
        config = serialize_config(get_config())
        model_config = config["enabled_models"][model_id]

        print(f"\nmodel_config: {model_config}\n")

        model_size_gb = self._get_model_size(model_config)
        available_vram_gb = self._get_available_vram()
        logger.info(
            f"Model size: {model_size_gb:.2f} GB, Available VRAM: {available_vram_gb:.2f} GB"
        )

        optimizations, force_full_optimization = self._determine_optimizations(
            available_vram_gb, model_size_gb, force_full_optimization
        )

        self._apply_optimizations(pipeline, optimizations, device)

        if not force_full_optimization:
            self._move_model_to_device(pipeline, device)

    def _determine_optimizations(
        self,
        available_vram_gb: float,
        model_size_gb: float,
        force_full_optimization: bool,
    ) -> tuple[list, bool]:
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

    def _get_full_optimizations(self) -> list:
        optimizations = [
            ("enable_vae_slicing", "VAE Sliced", {}),
            ("enable_vae_tiling", "VAE Tiled", {}),
            (
                "enable_model_cpu_offload",
                "CPU Offloading",
                {"device": get_available_torch_device()},
            ),
        ]
        if self.loaded_model.__class__.__name__ not in [
            "FluxPipeline",
            "FluxInpaintPipeline",
        ]:
            optimizations.append(
                (
                    "enable_xformers_memory_efficient_attention",
                    "Memory Efficient Attention",
                    {},
                )
            )
        return optimizations

    def _apply_optimizations(
        self, pipeline: DiffusionPipeline, optimizations: list, device: torch.device
    ):
        device_type = device if isinstance(device, str) else device.type
        if device_type == "mps":
            setattr(torch, "mps", torch.backends.mps)

        for opt_func, opt_name, kwargs in optimizations:
            try:
                getattr(pipeline, opt_func)(**kwargs)
                logger.info(f"{opt_name} enabled")
            except Exception as e:
                logger.error(f"Error enabling {opt_name}: {e}")

        if device_type == "mps":
            delattr(torch, "mps")

    def _move_model_to_device(self, pipeline: DiffusionPipeline, device: torch.device):
        logger.info("Moving model to device")
        pipeline = pipeline.to(device)
        # if pipeline.__class__.__name__ in ["FluxPipeline", "FluxInpaintPipeline"] and device.type != "mps":
        #     torch.set_float32_matmul_precision("high")
        #     pipeline.transformer.to(memory_format=torch.channels_last)
        #     pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)

        self.is_in_device = True

    def flush_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            # print("Emptying MPS cache...")
            # setattr(torch, "mps", torch.backends.mps)
            # torch.mps.empty_cache()
            pass
        gc.collect()

    def unload(self, model_id: str) -> None:
        if model_id == self.current_model and self.loaded_model is not None:
            del self.loaded_model
            self.loaded_model = None
            self.current_model = None
            self.flush_memory()
            logger.info(f"Model {model_id} unloaded.")
        else:
            logger.warning(f"Model {model_id} is not currently loaded.")

    def is_loaded(self, model_id: str) -> bool:
        return model_id == self.current_model and self.loaded_model is not None

    def get_model(self, model_id: str) -> Optional[DiffusionPipeline]:
        return self.loaded_model if self.is_loaded(model_id) else None

    def get_model_device(self, model_id: str) -> Optional[torch.device]:
        model = self.get_model(model_id)
        return model.device if model else None



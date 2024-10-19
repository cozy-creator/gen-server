import os
import gc
import logging
import importlib
from enum import Enum
from typing import Optional, Any, Dict

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


class ModelMemoryManager:
    def __init__(self):
        self.current_model: Optional[str] = None
        self.loaded_models: Dict[str, DiffusionPipeline] = {}
        self.model_sizes: Dict[str, int] = {}
        self.hf_model_manager = get_hf_model_manager()
        self.cache_dir = HF_HUB_CACHE
        self.is_in_device = False
        self.should_quantize = False
        self.vram_usage = 0
        self.max_vram = self._get_total_vram()
        self.vram_buffer = VRAM_SAFETY_MARGIN_GB
        self.VRAM_THRESHOLD = 1.4

    def _get_total_vram(self) -> int:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return 0  # Default to 0 for non-CUDA devices

    def _get_available_vram(self) -> int:
        if torch.cuda.is_available():
            available_vram_gb = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            ) / (1024**3)
            print(f"available vram: {available_vram_gb}")
            return available_vram_gb
        return 0  # Default to 0 for non-CUDA devices

    def _can_load_model(self, model_size: int) -> bool:
        available_vram = self._get_available_vram()

        if not self.loaded_models:
            if model_size <= available_vram - self.vram_buffer:
                return True, False  # Can load without full optimization
            elif model_size <= available_vram * self.VRAM_THRESHOLD:
                return True, True  # Can load with full optimization
            else:
                return False, False  # Cannot load even with full optimization

        return available_vram - self.vram_buffer >= model_size, False

    def _get_model_size(self, model_config: dict[str, Any]) -> int:
        if "ct:" in model_config["source"] or "file:" in model_config["source"]:
            return os.path.getsize(
                model_config["source"].replace("ct:", "").replace("file:", "")
            ) / (1024**3)

        repo_id = model_config["source"].replace("hf:", "")
        return self._calculate_repo_size(repo_id, model_config)

    def _calculate_repo_size(self, repo_id: str, model_config: dict[str, Any]) -> int:
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
                        prompt=dummy_prompt, num_inference_steps=1, output_type="pil"
                    )
                # elif isinstance(pipeline, FluxPipeline):
                #     _ = pipeline(prompt=dummy_prompt, num_inference_steps=1, output_type="pil")
                else:
                    logger.warning(
                        f"Unsupported pipeline type for warm-up: {type(pipeline)}"
                    )
        except Exception as e:
            logger.error(f"Error during warm-up for model {model_id}: {str(e)}")

        self.flush_memory()

        logger.info(f"Warm-up completed for model {model_id}")

    async def load(
        self, model_id: str, gpu: Optional[int] = None, type: Optional[str] = None
    ) -> Optional[DiffusionPipeline]:
        logger.info(f"Loading model {model_id}")

        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} is already loaded.")
            return self.loaded_models[model_id]

        self.is_in_device = False
        config = serialize_config(get_config())
        model_config = config["enabled_models"].get(model_id)

        if not model_config:
            logger.error(f"Model {model_id} not found in configuration.")
            return None

        estimated_size = self._get_model_size(model_config)

        can_load, force_full_optimization = self._can_load_model(estimated_size)

        if not can_load:
            logger.error(
                f"Not enough VRAM to load model {model_id}, even with full optimizations."
            )
            return None

        if force_full_optimization:
            logger.warning(
                f"Model {model_id} requires full optimizations to load. This may impact performance."
            )

        source = model_config["source"]
        prefix, path = source.split(":", 1)
        type = model_config["type"]
        self.should_quantize = model_config.get("quantize", False)

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

        if pipeline is not None:
            self.loaded_models[model_id] = pipeline
            self.model_sizes[model_id] = estimated_size
            self.vram_usage += estimated_size
            self.apply_optimizations(pipeline, model_id, force_full_optimization)

        return pipeline

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
        pipeline.to(device)
        # if pipeline.__class__.__name__ in ["FluxPipeline", "FluxInpaintPipeline"] and device.type != "mps":
        #     torch.set_float32_matmul_precision("high")
        #     pipeline.transformer.to(memory_format=torch.channels_last)
        #     pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)

        self.is_in_device = True

    def flush_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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


# def _determine_optimizations(self, available_vram_gb: float, model_size_gb: float, force_full_optimization: bool) -> tuple[list, bool]:
#         optimizations = []

#         if self.should_quantize and available_vram_gb >= GPUEnum.HIGH.value:
#             force_full_optimization = False
#         elif available_vram_gb >= GPUEnum.VERY_HIGH.value:
#             force_full_optimization = False
#         elif available_vram_gb > model_size_gb:
#             if force_full_optimization and available_vram_gb < GPUEnum.HIGH.value:
#                 optimizations = self._get_full_optimizations()
#                 force_full_optimization = True
#         else:
#             if not (self.should_quantize and available_vram_gb >= GPUEnum.HIGH.value):
#                 optimizations = self._get_full_optimizations()
#                 force_full_optimization = True

#         return optimizations, force_full_optimization

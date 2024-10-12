import os
import gc
import logging
import importlib
from enum import Enum
from typing import Optional, Any, Dict

import torch
from diffusers import DiffusionPipeline, FluxInpaintPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline
from diffusers.loaders import FromSingleFileMixin
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name

# from optimum.quanto import freeze, qfloat8, quantize
from ..utils.utils import serialize_config
from diffusers import (
    FluxInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    FluxPipeline,
)
import gc
from enum import Enum
from ..utils.quantize_models import quantize_model_fp8

logger = logging.getLogger(__name__)

class GPUEnum(Enum):
    LOW = 7
    MEDIUM = 14
    HIGH = 22
    VERY_HIGH = 30


# safety margin (in GB)
VRAM_SAFETY_MARGIN_GB = 2.0

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

    def _get_available_vram(self) -> int:
        if torch.cuda.is_available():
            return (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            )
        elif torch.backends.mps.is_available():
            return 0  # MPS doesn't provide VRAM info, so we'll assume it needs optimization
        return 0

    def _get_model_size(self, model_config: dict[str, Any]) -> int:
        if "ct:" in model_config["source"] or "file:" in model_config["source"]:
            return os.path.getsize(
                model_config["source"].replace("ct:", "").replace("file:", "")
            )

        repo_id = model_config["source"].replace("hf:", "")

        def get_size_for_repo(
            repo_id: str, component_name: Optional[str] = None
        ) -> int:
            size = 0
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

            # Check if component name is a folder or file
            if not os.path.isdir(snapshot_folder):
                return os.path.getsize(snapshot_folder)
            variants = ["bf16", "fp8", "fp16", ""]  # Empty string for default variant

            def check_variant_files(folder: str, variant: str) -> bool:
                for root, _, files in os.walk(folder):
                    for file in files:
                        if (
                            file.endswith(f"{variant}.safetensors")
                            or file.endswith(f"{variant}.bin")
                            or (
                                variant == ""
                                and (
                                    file.endswith(".safetensors")
                                    or file.endswith(".bin")
                                    or file.endswith(".ckpt")
                                )
                            )
                        ):
                            return True
                return False

            selected_variant = next(
                (v for v in variants if check_variant_files(snapshot_folder, v)), None
            )

            if selected_variant is not None:
                for root, _, files in os.walk(snapshot_folder):
                    for file in files:
                        if (
                            selected_variant
                            and (
                                file.endswith(f"{selected_variant}.safetensors")
                                or file.endswith(f"{selected_variant}.bin")
                            )
                        ) or (
                            selected_variant == ""
                            and (
                                file.endswith(".safetensors")
                                or file.endswith(".bin")
                                or file.endswith(".ckpt")
                            )
                        ):
                            size += os.path.getsize(os.path.join(root, file))

            return size

        # Calculate size of the main model
        total_size += get_size_for_repo(repo_id)

        # Calculate size of replacement components
        if "components" in model_config and model_config["components"]:
            for key, component in model_config["components"].items():
                if isinstance(component, dict) and "source" in component:
                    if "source" in component and isinstance(component["source"], str):
                        # Diffusers format
                        if not component["source"].endswith(
                            (".safetensors", ".bin", ".ckpt", ".pt")
                        ):
                            component_name = key

                            # Component repo is the source without the hf: prefix
                            component_repo = component["source"].replace("hf:", "")
                        else:
                            # last element after the last '/'
                            component_name = component["source"].split("/")[-1]

                            component_repo = "/".join(
                                component["source"].split("/")[:-1]
                            ).replace("hf:", "")

                    component_size = get_size_for_repo(component_repo, component_name)
                    total_size += component_size
                    # Subtract the size of the replaced component from the main model
                    total_size -= get_size_for_repo(repo_id, key)

        return total_size

    async def load(
        self, model_id: str, gpu: Optional[int] = None, type: Optional[str] = None
    ) -> Optional[DiffusionPipeline]:
        print(f"Loading model {model_id}")
        if model_id == self.current_model and self.loaded_model is not None:
            logger.info(f"Model {model_id} is already loaded.")
            return self.loaded_model

        self.flush_memory()

        self.is_in_device = False
        config = serialize_config(get_config())
        model_config = config["enabled_models"].get(model_id)

        if not model_config:
            logger.error(f"Model {model_id} not found in configuration.")
            return None

        source = model_config["source"]
        prefix, path = source.split(":", 1)
        type = model_config["type"]
        self.should_quantize = model_config.get("quantize", False)

        if prefix == "hf":
            is_downloaded, variant = self.hf_model_manager.is_downloaded(model_id)
            if not is_downloaded:
                logger.info(f"Model {model_id} not downloaded. Please ensure the model is downloaded first.")
                return None
        else:
            path = os.path.join(get_config().models_path, path)
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return None

        if prefix == "hf":
            return await self.load_huggingface_model(
                model_id, path, gpu, type, variant, model_config
            )
        elif prefix in ["file", "ct"]:
            pipeline = await self.load_single_file_model(model_id, path, prefix, gpu, type)
        else:
            logger.error(f"Unsupported model source prefix: {prefix}")
            return None

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
            pipeline_kwargs = {}
            if "components" in model_config and model_config["components"]:
                print(f"Components: {model_config['components']}")
                for key, component in model_config["components"].items():
                    # Check if 'source' key is in the component dict and is a string
                    if "source" in component and isinstance(component["source"], str):
                        if not component["source"].endswith(
                            (".safetensors", ".bin", ".ckpt", ".pt")
                        ):
                            component_name = key

                            # Component repo is the source without the hf: prefix
                            component_repo = component["source"].replace("hf:", "")

                            pipeline_kwargs[key] = await self._load_diffusers_component(
                                repo_id, component_repo, component_name
                            )
                        else:
                            # Custom format
                            pipeline_kwargs[key] = self._load_custom_component(
                                component["source"], type, key
                            )
                    elif component["source"] is None:
                        pipeline_kwargs[key] = None

            if "custom_pipeline" in model_config and model_config["custom_pipeline"]:
                pipeline_kwargs["custom_pipeline"] = model_config["custom_pipeline"]

            if variant == "":
                variant = None

            # Temporary: We can use bfloat16 as standard dtype but I just noticed that float16 loads the pipeline faster.
            # Although, it's compulsory to use bfloat16 for Flux models.
            # Load the pipeline
            # Check if the model is a Flux model else use DiffusionPipeline
            if "flux" in model_id.lower() and type == "inpaint":
                pipeline = FluxInpaintPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                    variant=variant,
                    **pipeline_kwargs,
                )
            else:
                pipeline = DiffusionPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=torch.bfloat16
                    if "flux" in model_id.lower()
                    else torch.float16,
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

    async def load_single_file_model(
        self,
        model_id: str,
        path: str,
        prefix: str,
        gpu: Optional[int] = None,
        type: Optional[str] = None,
    ) -> Optional[DiffusionPipeline]:
        print(f"\n\n === Loading single file model {model_id} === \n\n")

        if type is None:
            logger.error("Model type must be specified for single file models.")
            return None

        pipeline_class = PIPELINE_MAPPING.get(type)
        if not pipeline_class:
            logger.error(f"Unsupported model type: {type}")
            return None

        if prefix == "ct":
            # check if the file is an http or https link is so, get just the file name which is the last element after the last '/'
            if "http" in path or "https" in path:
                path = path.split("/")[-1]

            # For now, we'll assume the file is already downloaded
            path = os.path.join(get_config().models_path, path)

        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return None

        if issubclass(pipeline_class, FromSingleFileMixin):
            try:
                pipeline = pipeline_class.from_single_file(
                    path,
                    torch_dtype=torch.bfloat16
                    if "flux" in model_id.lower()
                    else torch.float16,
                )
                self.loaded_model = pipeline
                self.current_model = model_id
                return pipeline
            except Exception as e:
                logger.error(f"Error loading model using from_single_file: {str(e)}")
                # Fall back to custom architecture loading

        # Custom architecture loading
        try:
            state_dict = load_state_dict_from_file(path)
            pipeline = pipeline_class()

            for component_name in MODEL_COMPONENTS[type]:
                if component_name in ["scheduler", "tokenizer", "tokenizer_2"]:
                    # These components are not in the state dict. Should I handle here or in the architecture?
                    continue

                arch_key = f"core_extension_1.{type}_{component_name}"
                architecture_class = get_architectures().get(arch_key)

                if not architecture_class:
                    logger.error(f"Architecture not found for {arch_key}")
                    continue

                architecture = architecture_class()
                architecture.load(state_dict)

                setattr(pipeline, component_name, architecture.model)

            self.loaded_model = pipeline
            self.current_model = model_id
            return pipeline
        except Exception as e:
            logger.error(f"Error loading model using custom architecture: {str(e)}")
            return None

    async def _load_diffusers_component(
        self, repo_id: str, component_repo: str, component_name: str = None
    ) -> Any:
        print(f"Loading diffusers component {component_name} from {repo_id}")
        try:
            # Get the model index using HFModelManager
            model_index = (
                await self.hf_model_manager.get_diffusers_multifolder_components(
                    repo_id
                )
            )

            # Get the component info
            component_info = model_index.get(component_name) if model_index else None
            if not component_info:
                raise ValueError(f"Invalid component info for {component_name}")

            module_path, class_name = component_info
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            component = model_class.from_pretrained(
                component_repo,
                subfolder=component_name,
                # local_files_only=True,
                torch_dtype=torch.bfloat16
                if "flux" in repo_id.lower()
                else torch.float16,
            )

            if self.should_quantize:
                quantize_model_fp8(component)

            return component

        except Exception as e:
            logger.error(f"Error loading component {component_name} from {component_repo}: {str(e)}")
            raise

    def _load_custom_component(self, repo_id: str, category: str, component_name: str):
        print(f"Loading custom component {component_name} from {repo_id}")
        file_path = None
        # Keep only the name between and after the first slash including the slash
        repo_folder = os.path.dirname(repo_id)

        repo_folder = repo_folder.replace("hf:", "")

        # repo_id, weights_name = _extract_repo_id_and_weights_name(repo_id)

        # Load the state dict
        model_folder = os.path.join(
            self.cache_dir, repo_folder_name(repo_id=repo_folder, repo_type="model")
        )

        # Get the safetensors file name by splitting the repo_id by '/' and getting the last element
        weights_name = repo_id.split("/")[-1]

        # Get the safetensors file
        if not os.path.exists(model_folder):
            model_folder = os.path.join(self.cache_dir, repo_folder)
            if not os.path.exists(model_folder):
                raise FileNotFoundError(f"Cache folder for {repo_id} not found")
            else:
                file_path = os.path.join(model_folder, weights_name)

        if not file_path:
            # Get the latest commit hash
            refs_path = os.path.join(model_folder, "refs", "main")
            if not os.path.exists(refs_path):
                raise FileNotFoundError(f"refs/main not found for {repo_id}")

            with open(refs_path, "r") as f:
                commit_hash = f.read().strip()

            # Construct the path to model_index.json
            file_path = os.path.join(
                model_folder, "snapshots", commit_hash, weights_name
            )

        state_dict = load_state_dict_from_file(file_path)

        architectures = get_architectures()
        arch_key = f"core_extension_1.{category.lower()}_{component_name.lower()}"
        architecture_class = architectures.get(arch_key)

        if not architecture_class:
            raise ValueError(f"Architecture not found for {arch_key}")

        architecture = architecture_class()

        # Load the state dict into the architecture
        architecture.load(state_dict)
        model = architecture.model

        if self.should_quantize:
            quantize_model_fp8(model)

        return model

    def apply_optimizations(
        self, pipeline: DiffusionPipeline, force_full_optimization: bool = False
    ):
        # Check if the model is loaded in memory already
        if self.loaded_model is not None:
            if self.is_in_device:
                print(
                    f"Model {self.current_model} is already loaded in memory and in device. Not applying optimizations."
                )
                return

        device = get_available_torch_device()
        config = serialize_config(get_config())
        model_config = config["enabled_models"][model_id]

        print(f"\nmodel_config: {model_config}\n")

        model_size_gb = self._get_model_size(model_config) / (1024**3)
        available_vram_gb = self._get_available_vram() / (1024**3)
        print(f"Model size: {model_size_gb} GB, Available VRAM: {available_vram_gb} GB")

        optimizations, force_full_optimization = self._determine_optimizations(available_vram_gb, model_size_gb, force_full_optimization)

        self._apply_optimizations(pipeline, optimizations, device)

        if not force_full_optimization:
            self._move_model_to_device(pipeline, device)

    def _determine_optimizations(self, available_vram_gb: float, model_size_gb: float, force_full_optimization: bool) -> tuple[list, bool]:
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
            ("enable_model_cpu_offload", "CPU Offloading", {"device": device}),
        ]
        # optimizations = [
        #     ("enable_vae_slicing", "VAE Sliced", {}),
        #     ("enable_vae_tiling", "VAE Tiled", {}),
        # ]

        # ------------------------------------ DO SOME CHECKS ------------------------------------

        # Check 1: If should quantize and available VRAM equal or greater than High, don't apply optimizations
        if self.should_quantize:
            if available_vram_gb >= GPUEnum.HIGH.value:
                force_full_optimization = False
            else:
                force_full_optimization = True
                print(
                    f"Available VRAM: {available_vram_gb} GB is less than 24GB. Applying optimizations."
                )

        # Check 2: If available VRAM equal or greater than Very High, don't apply optimizations
        if available_vram_gb >= GPUEnum.VERY_HIGH.value:
            force_full_optimization = False

        # Check 3: If available VRAM is greater than model size, apply optimizations if force_full_optimization is True
        if available_vram_gb > model_size_gb:
            if force_full_optimization:
                if available_vram_gb >= GPUEnum.HIGH.value:
                    force_full_optimization = False
                    print(
                        f"Available VRAM: {available_vram_gb} GB is greater than 24GB. Not applying optimizations."
                    )
                else:
                    optimizations = FULL_OPTIMIZATION
                    if pipeline.__class__.__name__ not in [
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
        # Check 4: If should quantize and available VRAM is less than High, apply optimizations
        else:
            if self.should_quantize and available_vram_gb >= GPUEnum.HIGH.value:
                force_full_optimization = False
                print(
                    f"Available VRAM: {available_vram_gb} GB is greater than 24GB. Not applying optimizations."
                )
            else:
                optimizations = FULL_OPTIMIZATION
                if pipeline.__class__.__name__ not in [
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
            setattr(torch, "mps", torch.backends.mps)
            torch.mps.empty_cache()

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
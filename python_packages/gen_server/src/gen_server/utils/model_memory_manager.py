import torch
from typing import Optional, Any
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from ..globals import (
    get_hf_model_manager,
    get_architectures,
    get_available_torch_device,
)
from ..config import get_config
from ..utils.load_models import load_state_dict_from_file
import logging
import importlib
from huggingface_hub.constants import HF_HUB_CACHE
import os
from huggingface_hub.file_download import repo_folder_name
# from optimum.quanto import freeze, qfloat8, quantize
from ..utils.utils import serialize_config
from diffusers import FluxInpaintPipeline
import gc
from enum import Enum

logger = logging.getLogger(__name__)



class GPUTier(Enum):
    LOW = 1     # 2-4 GB VRAM
    MEDIUM = 2  # 6-8 GB VRAM
    HIGH = 3    # 12-16 GB VRAM
    VERY_HIGH = 4  # 24+ GB VRAM


class ModelMemoryManager:
    def __init__(self):
        self.current_model: Optional[str] = None
        self.loaded_model: Optional[DiffusionPipeline] = None
        self.hf_model_manager = get_hf_model_manager()
        self.cache_dir = HF_HUB_CACHE
        self.gpu_tier = self._determine_gpu_tier()


    def _determine_gpu_tier(self) -> GPUTier:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 24:
                return GPUTier.VERY_HIGH
            elif vram_gb >= 12:
                return GPUTier.HIGH
            elif vram_gb >= 6:
                return GPUTier.MEDIUM
            # Check if the GPU is an Apple Silicon GPU
            elif torch.backends.mps.is_available():
                return GPUTier.HIGH

        return GPUTier.LOW
    

    def _get_model_size(self, model_config: dict[str, Any]) -> int:
        total_size = 0
        repo_id = model_config["source"].replace("hf:", "")
        
        def get_size_for_repo(repo_id: str, component_name: Optional[str] = None) -> int:
            size = 0
            storage_folder = os.path.join(self.cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model"))
            if not os.path.exists(storage_folder):
                logger.warning(f"Storage folder for {repo_id} not found.")
                return 0

            refs_path = os.path.join(storage_folder, "refs", "main")
            if not os.path.exists(refs_path):
                logger.warning(f"No commit hash found for {repo_id}")
                return 0

            with open(refs_path, "r") as f:
                commit_hash = f.read().strip()

            snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
            if component_name:
                snapshot_folder = os.path.join(snapshot_folder, component_name)

            variants = ["bf16", "fp8", "fp16", ""]  # Empty string for default variant
            
            def check_variant_files(folder: str, variant: str) -> bool:
                for root, _, files in os.walk(folder):
                    for file in files:
                        if file.endswith(f"{variant}.safetensors") or \
                        file.endswith(f"{variant}.bin") or \
                        (variant == "" and (file.endswith(".safetensors") or file.endswith(".bin") or file.endswith(".ckpt"))):
                            return True
                return False

            selected_variant = next((v for v in variants if check_variant_files(snapshot_folder, v)), None)
            
            if selected_variant is not None:
                for root, _, files in os.walk(snapshot_folder):
                    for file in files:
                        if (selected_variant and (file.endswith(f"{selected_variant}.safetensors") or file.endswith(f"{selected_variant}.bin"))) or \
                        (selected_variant == "" and (file.endswith(".safetensors") or file.endswith(".bin") or file.endswith(".ckpt"))):
                            size += os.path.getsize(os.path.join(root, file))

            return size

        # Calculate size of the main model
        total_size += get_size_for_repo(repo_id)

        # Calculate size of replacement components
        if "components" in model_config and model_config["components"]:
            for _, component in model_config["components"].items():
                if isinstance(component, dict) and "source" in component:
                    if "source" in component and isinstance(component["source"], str):
                        # Diffusers format
                        if not component["source"].endswith(
                            (".safetensors", ".bin", ".ckpt", ".pt")
                        ):
                            # Split the source by '/' and get the last element
                            component_name = component["source"].split("/")[-1]

                            # Component repo is the source without the last element, join it with '/'
                            component_repo = "/".join(
                                component["source"].split("/")[:-1]
                            ).replace("hf:", "")
                    
                    component_size = get_size_for_repo(component_repo, component_name)
                    total_size += component_size
                    # Subtract the size of the replaced component from the main model
                    total_size -= get_size_for_repo(repo_id, component_name)

        return total_size

    async def load(
        self, model_id: str, gpu: Optional[int] = None, type: Optional[str] = None
    ) -> Optional[DiffusionPipeline]:
        print(f"Loading model {model_id}")
        if model_id == self.current_model and self.loaded_model is not None:
            logger.info(f"Model {model_id} is already loaded.")
            return self.loaded_model, False
        
        self.flush_memory()

        # Unload the current model if it exists and is different
        if self.current_model is not None and self.current_model != model_id:
            self.unload(self.current_model)

        config = get_config()

        config = serialize_config(config)

        model_config = config["enabled_models"].get(model_id)
        if not model_config:
            logger.error(f"Model {model_id} not found in configuration.")
            return None

        repo_id = model_config["source"].replace("hf:", "")

        category = model_config.get("category", None)

        is_downloaded, variant = self.hf_model_manager.is_downloaded(model_id)
        if not is_downloaded:
            logger.info(
                f"Model {model_id} not downloaded. Please ensure the model is downloaded first."
            )
            return None

        try:
            pipeline_kwargs = {}
            if "components" in model_config and model_config["components"]:
                print(f"Components: {model_config['components']}")
                for key, component in model_config["components"].items():
                    # Check if 'source' key is in the component dict and is a string
                    if "source" in component and isinstance(component["source"], str):
                        # Diffusers format
                        if not component["source"].endswith(
                            (".safetensors", ".bin", ".ckpt", ".pt")
                        ):
                            # Split the source by '/' and get the last element
                            component_name = component["source"].split("/")[-1]

                            # Component repo is the source without the last element, join it with '/'
                            component_repo = "/".join(
                                component["source"].split("/")[:-1]
                            ).replace("hf:", "")
                            print(component_repo, component_name)
                            pipeline_kwargs[key] = await self._load_diffusers_component(
                                repo_id, component_repo, component_name
                            )
                        else:
                            # Custom format
                            pipeline_kwargs[key] = self._load_custom_component(
                                component["source"], category, key
                            )
                    elif component["source"] is None:
                        pipeline_kwargs[key] = None

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

    async def _load_diffusers_component(
        self, repo_id: str, component_repo: str, component_name: str
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

            # Get the module path and class name
            module_path, class_name = component_info

            # Import the class
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            # Check for quantized models
            if model_index["_class_name"] == "FluxPipeline":
                quantized_model_list = ["FluxTransformer2DModel", "T5EncoderModel"]


                # Check if the VRAM is greater than 16gb before quantizing. If it is greater than 16gb, we do not quantize.
                if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 > 16:
                    logger.info(f"VRAM is greater than 16gb. Not quantizing.")
                elif class_name in quantized_model_list:
                    quantized_class_name = (
                        "QuantizedFluxTransformer2DModel"
                        if class_name == "FluxTransformer2DModel"
                        else "QuantizedT5EncoderModelForCausalLM"
                    )
                    quantized_module_path = "gen_server.utils.quantize_models"
                    storage_folder = os.path.join(
                        self.cache_dir, "models--" + component_repo.replace("/", "--")
                    )

                    if not os.path.exists(storage_folder):
                        raise FileNotFoundError(
                            f"Quantized model {component_repo} not found"
                        )

                    # Get the latest commit hash
                    refs_path = os.path.join(storage_folder, "refs", "main")
                    if not os.path.exists(refs_path):
                        return FileNotFoundError(f"No commit hash found")

                    with open(refs_path, "r") as f:
                        commit_hash = f.read().strip()

                    repo_id = os.path.join(
                        storage_folder, "snapshots", commit_hash, component_name
                    )

                    # Import the class
                    quantized_module = importlib.import_module(quantized_module_path)
                    quantized_model_class = getattr(
                        quantized_module, quantized_class_name
                    )

                    if class_name == "T5EncoderModel":
                        model_class.from_config = lambda config: model_class(config)

                    print(
                        f"Loading component {component_name} from {repo_id}. Class_name: {quantized_class_name}, module_path: {quantized_module_path}"
                    )

                    # Load the component
                    component = quantized_model_class.from_pretrained(
                        repo_id,
                    ).to(torch.bfloat16)

                    return component

            print(
                f"Loading component {component_name} from {repo_id}. Class_name: {class_name}, module_path: {module_path}"
            )

            # Load the component
            component = model_class.from_pretrained(
                repo_id,
                subfolder=component_name,
                local_files_only=True,
                torch_dtype=torch.float16,
            )

            return component

        except Exception as e:
            logger.error(
                f"Error loading component {component_name} from {repo_id}: {str(e)}"
            )
            raise

    def _load_custom_component(self, repo_id: str, category: str, component_name: str):
        file_path = None
        # Keep only the name between and after the first slash including the slash
        repo_folder = os.path.dirname(repo_id)

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

        # Get the architectures registry
        architectures = get_architectures()
        # print(f"Architectures: {architectures}")

        # Find the correct architecture class
        arch_key = f"core_extension_1.{category.lower()}_{component_name.lower()}"

        architecture_class = architectures.get(arch_key)

        # Initialize the architecture
        architecture = architecture_class()

        # Load the state dict into the architecture
        architecture.load(state_dict)

        return architecture.model

    def apply_optimizations(self, pipeline: DiffusionPipeline):
        device = get_available_torch_device()
        optimizations = [
            ("enable_vae_slicing", "VAE Sliced", {}),
            ("enable_vae_tiling", "VAE Tiled", {}),
            (
                "enable_xformers_memory_efficient_attention",
                "Memory Efficient Attention",
                {},
            ),
            (
                "enable_model_cpu_offload",
                "CPU Offloading",
                {"device": device},
            ),
        ]

        # Check if pipeline is a FluxPipeline so as to not use Xformers optimizations
        if pipeline.__class__.__name__ in ["FluxPipeline", "FluxInpaintPipeline"]:
            optimizations.remove(
                (
                    "enable_xformers_memory_efficient_attention",
                    "Memory Efficient Attention",
                    {},
                )
            )

        # Patch torch.mps to torch.backends.mps
        device_type = device if isinstance(device, str) else device.type
        if device_type == "mps":
            setattr(torch, "mps", torch.backends.mps)

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 > 16:
            print(f"VRAM is greater than 16gb. Not enabling optimizations")
            pipeline.to("cuda")
            return

        for opt_func, opt_name, kwargs in optimizations:
            try:
                getattr(pipeline, opt_func)(**kwargs)
                print(f"{opt_name} enabled")
            except Exception as e:
                print(f"Error enabling {opt_name}: {e}")

        if device_type == "mps":
            delattr(torch, "mps")

    def flush_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
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

    # def list_loaded(self, gpu: Optional[int] = None) -> List[str]:
    #     if gpu is not None:
    #         return [
    #             repo_id
    #             for repo_id, pipeline in self.loaded_models.items()
    #             if pipeline.device.type == "cuda" and pipeline.device.index == gpu
    #         ]
    #     return list(self.loaded_models.keys())

    def is_loaded(self, model_id: str) -> bool:
        return model_id == self.current_model and self.loaded_model is not None

    def get_model(self, model_id: str) -> Optional[DiffusionPipeline]:
        if self.is_loaded(model_id):
            return self.loaded_model
        return None

    def get_model_device(self, model_id: str) -> Optional[torch.device]:
        model = self.get_model(model_id)
        return model.device if model else None

    # Check the current memory usage (method)
    # def get_memory_usage(self):
    #     memory_usage = {}
    #     for model_id, pipeline in self.loaded_models.items():
    #         memory_usage[model_id] = pipeline.get_memory_usage()
    #     return memory_usage

    # dynamically unload models
    # def unload_models(self, max_memory_usage: int):
    #     for model_id, pipeline in self.loaded_models.items():
    #         if pipeline.get_memory_usage() > max_memory_usage:
    #             self.unload(model_id)

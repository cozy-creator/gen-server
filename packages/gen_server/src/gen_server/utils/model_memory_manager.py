import torch
from typing import Dict, List, Optional, Any
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from ..globals import get_hf_model_manager, get_architectures, get_available_torch_device
from .config_manager import get_model_config
from ..utils.load_models import load_state_dict_from_file
import logging
import importlib
from huggingface_hub.constants import HF_HUB_CACHE
import os
from huggingface_hub.file_download import repo_folder_name
from optimum.quanto import freeze, qfloat8, quantize

logger = logging.getLogger(__name__)

class ModelMemoryManager:
    def __init__(self):
        self.loaded_models: dict[str, DiffusionPipeline] = {}
        self.hf_model_manager = get_hf_model_manager()
        self.cache_dir = HF_HUB_CACHE


    def load(self, model_id: str, gpu: Optional[int] = None) -> Optional[DiffusionPipeline]:
        print(f"Loading model {model_id}")
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} is already loaded.")
            return self.loaded_models[model_id]
        
        model_config = get_model_config()

        model_config = model_config['models'].get(model_id)
        if not model_config:
            logger.error(f"Model {model_id} not found in configuration.")
            return None

        repo_id = model_config['repo'].replace('hf:', '')
        category = model_config['category']


        is_downloaded, variant = self.hf_model_manager.is_downloaded(model_id)
        if not is_downloaded:
            logger.info(f"Model {model_id} not downloaded. Please ensure the model is downloaded first.")
            return None

        try:
            pipeline_kwargs = {}
            if 'model_index' in model_config:
                print(f"Model Index: {model_config}")
                for component, source in model_config['model_index'].items():
                    if isinstance(source, list):
                        # Diffusers format
                        component_repo, component_name = source
                        pipeline_kwargs[component] = self._load_diffusers_component(component_repo, component_name)
                    elif isinstance(source, str) and source.endswith(('.safetensors', '.bin', '.ckpt')):
                        # Custom format
                        pipeline_kwargs[component] = self._load_custom_component(source, category, component)
                    elif source is None:
                        pipeline_kwargs[component] = None

            if variant == "":
                variant = None
                
            pipeline = DiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                local_files_only=True,
                variant=variant,
                **pipeline_kwargs
            )

            pipeline.to(torch.float16)



            self.loaded_models[model_id] = pipeline
            logger.info(f"Model {model_id} loaded successfully.")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            return None
        

    def _load_diffusers_component(self, repo_id: str, component_name: str) -> Any:
        try:
            # Get the model index using HFModelManager
            model_index = self.hf_model_manager.get_model_index(repo_id)
            
            # Get the component info
            component_info = model_index.get(component_name)
            if not component_info or not isinstance(component_info, list) or len(component_info) != 2:
                raise ValueError(f"Invalid component info for {component_name}")

            # Get the module path and class name
            module_path, class_name = component_info

            # Import the class
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            # Load the component
            component = model_class.from_pretrained(
                repo_id,
                subfolder=component_name,
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )

            # if model_index["_class_name"] == "FluxPipeline":
            #     if torch.cuda.is_available():
            #         component.to("cuda")
            #         quantize(component, weights=qfloat8)
            #         freeze(component)
            #         print(f"FluxPipeline component: {component_name} quantized and frozen")

            #         torch.cuda.empty_cache()
            #     elif torch.backends.mps.is_available():
            #         component.to("mps")
            #         quantize(component, weights=qfloat8)
            #         freeze(component)
            #         print(f"FluxPipeline component: {component_name} quantized and frozen")

            #         torch.mps.empty_cache()

            return component

        except Exception as e:
            logger.error(f"Error loading component {component_name} from {repo_id}: {str(e)}")
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
        weights_name = repo_id.split('/')[-1]

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
            file_path = os.path.join(model_folder, "snapshots", commit_hash, weights_name)

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

        # transformer = architecture.model


        # if arch_key == "core_extension_1.flux_transformer":
        #     if torch.cuda.is_available():
        #         print("Moving to cuda")
        #         transformer.to("cuda")
        #         print("Quantizing")
        #         quantize(transformer, weights=qfloat8)
        #         print("Freezing")
        #         freeze(transformer)
        #         print(f"FluxTransformer component: {component_name} quantized and frozen")
        #         print("Emptying cache")
        #         torch.cuda.empty_cache()
        #     elif torch.backends.mps.is_available():
        #         transformer.to("mps")
        #         quantize(transformer, weights=qfloat8)
        #         freeze(transformer)
        #         print(f"FluxTransformer component: {component_name} quantized and frozen")


        #         torch.mps.empty_cache()

        return architecture.model
    

    def apply_optimizations(self, pipeline: DiffusionPipeline):
        device = get_available_torch_device()
        optimizations = [
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
        if pipeline.__class__.__name__ == "FluxPipeline":
            optimizations.remove(("enable_xformers_memory_efficient_attention", "Memory Efficient Attention", {}))

        # Patch torch.mps to torch.backends.mps
        device_type = device if isinstance(device, str) else device.type
        if device_type == "mps":
            setattr(torch, "mps", torch.backends.mps)
        for opt_func, opt_name, kwargs in optimizations:
            try:
                getattr(pipeline, opt_func)(**kwargs)
                print(f"{opt_name} enabled")
            except Exception as e:
                print(f"Error enabling {opt_name}: {e}")

        delattr(torch, "mps")

    def unload(self, repo_id: str) -> None:
        if repo_id in self.loaded_models:
            del self.loaded_models[repo_id]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

            logger.info(f"Model {repo_id} unloaded.")
        else:
            logger.warning(f"Model {repo_id} is not currently loaded.")

    def list_loaded(self, gpu: Optional[int] = None) -> List[str]:
        if gpu is not None:
            return [
                repo_id
                for repo_id, pipeline in self.loaded_models.items()
                if pipeline.device.type == "cuda" and pipeline.device.index == gpu
            ]
        return list(self.loaded_models.keys())

    def is_loaded(self, repo_id: str) -> bool:
        return repo_id in self.loaded_models

    def get_model(self, repo_id: str) -> Optional[DiffusionPipeline]:
        return self.loaded_models.get(repo_id)

    def get_model_device(self, repo_id: str) -> Optional[torch.device]:
        model = self.loaded_models.get(repo_id)
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







import torch
from typing import Dict, List, Optional
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from ..globals import _HF_MODEL_MANAGER, get_model_config
import logging

logger = logging.getLogger(__name__)

class ModelMemoryManager:
    def __init__(self):
        self.loaded_models: Dict[str, DiffusionPipeline] = {}
        

    # def load(self, repo_id: str, gpu: Optional[int] = None) -> Optional[DiffusionPipeline]:
    #     if not _HF_MODEL_MANAGER.is_downloaded(repo_id):
    #         logger.info(f"Model {repo_id} not downloaded. Please ensure the model is downloaded first.")
    #         return None

    #     if repo_id in self.loaded_models:
    #         logger.info(f"Model {repo_id} is already loaded.")
    #         return self.loaded_models[repo_id]

    #     variants = ["bf16", "fp8", "fp16", None]  # None represents no variant
        
    #     for variant in variants:
    #         try:
    #             if variant:
    #                 logger.info(f"Attempting to load {repo_id} with {variant} variant...")
    #             else:
    #                 logger.info(f"Attempting to load {repo_id} without variant...")

    #             pipeline = DiffusionPipeline.from_pretrained(
    #                 repo_id,
    #                 variant=variant,
    #                 torch_dtype=torch.float16,
    #                 local_files_only=True,
    #             )
                
    #             if gpu is not None and torch.cuda.is_available():
    #                 pipeline = pipeline.to(f"cuda:{gpu}")
    #             else:
    #                 pipeline = pipeline.to("cpu")

    #             self.loaded_models[repo_id] = pipeline
    #             logger.info(f"Model {repo_id} loaded successfully with variant: {variant if variant else 'default'}")
    #             return pipeline

    #         except Exception as e:
    #             if variant:
    #                 # logger.error(f"Failed to load {variant} variant for {repo_id}: {str(e)}. Trying next variant...")
    #                 continue
    #             else:
    #                 logger.error(f"Failed to load default variant for {repo_id}: {str(e)}")

    #     logger.error(f"Failed to load model {repo_id} with any variant.")
    #     return None


    def load(self, model_id: str, gpu: Optional[int] = None) -> Optional[DiffusionPipeline]:
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} is already loaded.")
            return self.loaded_models[model_id]

        model_config = get_model_config()['models'].get(model_id)
        if not model_config:
            logger.error(f"Model {model_id} not found in configuration.")
            return None

        repo_id = model_config['repo'].replace('hf:', '')
        category = model_config['category']

        if not _HF_MODEL_MANAGER.is_downloaded(repo_id):
            logger.info(f"Model {repo_id} not downloaded. Please ensure the model is downloaded first.")
            return None

        try:
            pipeline_kwargs = {}
            if 'model_index' in model_config:
                for component, source in model_config['model_index'].items():
                    if isinstance(source, list):
                        # Diffusers format
                        component_repo, component_name = source
                        pipeline_kwargs[component] = self._load_diffusers_component(component_repo, component_name)
                    elif isinstance(source, str) and source.endswith('.safetensors'):
                        # Custom format
                        pipeline_kwargs[component] = self._load_custom_component(source, category, component)
                    elif source is None:
                        pipeline_kwargs[component] = None

            pipeline = DiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                local_files_only=True,
                **pipeline_kwargs
            )

            if gpu is not None and torch.cuda.is_available():
                pipeline = pipeline.to(f"cuda:{gpu}")
            else:
                pipeline = pipeline.to("cpu")

            self.loaded_models[model_id] = pipeline
            logger.info(f"Model {model_id} loaded successfully.")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            return None

    def _load_diffusers_component(self, repo: str, component_name: str):
        return DiffusionPipeline.from_pretrained(repo, subfolder=component_name, local_files_only=True)

    def _load_custom_component(self, file_path: str, category: str, component_name: str):
        # Dynamically import the correct Architecture class based on category
        # module = importlib.import_module(f"..architectures.{category.lower()}", package=__name__)
        

        architecture_class = getattr(module, f"{category}Architecture")

        # Initialize the architecture
        architecture = architecture_class()

        # Load the state dict
        state_dict = torch.load(file_path, map_location="cpu")

        # Load the state dict into the architecture
        architecture.load(state_dict)

        return architecture.model

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

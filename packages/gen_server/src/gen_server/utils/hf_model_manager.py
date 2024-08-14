import os
import shutil
import asyncio
from typing import Optional, List, Dict, Any
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir
from huggingface_hub.file_download import repo_folder_name
import torch
import logging
from huggingface_hub.constants import HF_HUB_CACHE
import json
from .config_manager import get_model_config




logger = logging.getLogger(__name__)



class HFModelManager:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = HF_HUB_CACHE
        # self.cache_dir = None
        self.loaded_models: Dict[str, DiffusionPipeline] = {}
        self.hf_api = HfApi()


    def is_downloaded(self, model_id: str) -> bool:
        """
        Checks if a model is fully downloaded in the cache.
        Returns True if at least one variant is completely downloaded.
        """
        try:
            # Get the repo_id from the YAML configuration
            model_config = get_model_config()
            model_info = model_config['models'].get(model_id)
            if not model_info:
                logger.error(f"Model {model_id} not found in configuration.")
                return False
            
            print(f"Model Config: {model_config}")
            
            if model_info['repo']:
                repo_id = model_info['repo'].replace('hf:', '')
            else:
                repo_id = model_id

            print(f"Repo ID: {repo_id}")

            storage_folder = os.path.join(
                self.cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model")
            )

            print(f"Storage Folder: {storage_folder}")
            if not os.path.exists(storage_folder):
                return False

            # Get the latest commit hash
            refs_path = os.path.join(storage_folder, "refs", "main")
            if not os.path.exists(refs_path):
                return False

            with open(refs_path, "r") as f:
                commit_hash = f.read().strip()


            snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
            if not os.path.exists(snapshot_folder):
                return False

            # Check model_index.json for required folders
            model_index_path = os.path.join(snapshot_folder, "model_index.json")


            if os.path.exists(model_index_path):
                with open(model_index_path, "r") as f:
                    model_index = json.load(f)
                    required_folders = {
                        k
                        for k, v in model_index.items()
                        if isinstance(v, list)
                        and len(v) == 2
                        and v[0] is not None
                        and v[1] is not None
                    }


                # Remove known non-folder keys and ignored folders
                ignored_folders = {
                    "_class_name",
                    "_diffusers_version",
                    "scheduler",
                    "feature_extractor",
                    "tokenizer",
                    "tokenizer_2",
                    "tokenizer_3",
                    "safety_checker",
                }
                required_folders -= ignored_folders

                # Define variant hierarchy
                variants = [
                    "bf16",
                    "fp8",
                    "fp16",
                    "",
                ]  # empty string for normal variant

                def check_folder_completeness(folder_path: str, variant: str) -> bool:
                    if not os.path.exists(folder_path):
                        return False
                    
                    for _, _, files in os.walk(folder_path):
                        for file in files:
                            if file.endswith('.incomplete'):
                                print(f"Incomplete File: {file}")
                                return False

                            
                            if (file.endswith(f"{variant}.safetensors") or 
                                file.endswith(f"{variant}.bin") or
                                (variant == "" and (file.endswith('.safetensors') or file.endswith('.bin')))):
                                return True

                    return False

                def check_variant_completeness(variant: str) -> bool:
                    for folder in required_folders:
                        folder_path = os.path.join(snapshot_folder, folder)

                        if not check_folder_completeness(folder_path, variant):
                            return False

                    return True

                # Check variants in hierarchy
                for variant in variants:
                    print(f"Checking variant: {variant}")
                    if check_variant_completeness(variant):
                        return True

            else:
                # For repos without model_index.json, check the blob folder
                blob_folder = os.path.join(storage_folder, "blobs")
                if os.path.exists(blob_folder):
                    for _root, _, files in os.walk(blob_folder):
                        if any(file.endswith(".incomplete") for file in files):
                            return False

                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking download status for {repo_id}: {str(e)}")
            return False

    def list(self) -> List[str]:
        cache_info = scan_cache_dir()
        return [repo.repo_id for repo in cache_info.repos if self.is_downloaded(repo.repo_id)]
    
    def get_model_index(self, repo_id: str) -> Dict[str, Any]:
        storage_folder = os.path.join(
            self.cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model")
        )

        if not os.path.exists(storage_folder):
            raise FileNotFoundError(f"Cache folder for {repo_id} not found")

        # Get the latest commit hash
        refs_path = os.path.join(storage_folder, "refs", "main")
        if not os.path.exists(refs_path):
            raise FileNotFoundError(f"refs/main not found for {repo_id}")

        with open(refs_path, "r") as f:
            commit_hash = f.read().strip()

        # Construct the path to model_index.json
        model_index_path = os.path.join(storage_folder, "snapshots", commit_hash, "model_index.json")

        if not os.path.exists(model_index_path):
            raise FileNotFoundError(f"model_index.json not found for {repo_id}")

        with open(model_index_path, 'r') as f:
            return json.load(f)


    async def download(
        self,
        repo_id: str,
        variant: Optional[str] = "fp16",
        file_name: Optional[str] = None,
        sub_folder: Optional[str] = None,
    ) -> None:
        if file_name:
            try:
                await asyncio.to_thread(
                    hf_hub_download,
                    repo_id,
                    file_name,
                    cache_dir=self.cache_dir,
                    subfolder=sub_folder,
                )
                logger.info(f"File {file_name} from {repo_id} downloaded successfully.")
                self.list()  # Refresh the cached list
                return
            except Exception as e:
                logger.error(f"Failed to download file {file_name} from {repo_id}: {str(e)}")
                return

        variants = ["bf16", "fp8", "fp16", None]  # None represents no variant
        for var in variants:
            try:
                if var:
                    logger.info(f"Attempting to download {repo_id} with {var} variant...")
                else:
                    logger.info(f"Attempting to download {repo_id} without variant...")

                await asyncio.to_thread(
                    DiffusionPipeline.download,
                    repo_id,
                    variant=var,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16,
                )

                logger.info(f"Model {repo_id} downloaded successfully with variant: {var if var else 'default'}")
                self.list()  # Refresh the cached list
                return

            except Exception as e:
                if var:
                    logger.error(f"Failed to download {var} variant for {repo_id}: {str(e)}. Trying next variant...")
                else:
                    logger.error(f"Failed to download default variant for {repo_id}: {str(e)}")

        logger.error(f"Failed to download model {repo_id} with any variant.")


    async def delete(self, repo_id: str) -> None:
        model_path = os.path.join(
            self.cache_dir, "models--" + repo_id.replace("/", "--")
        )
        if os.path.exists(model_path):
            await asyncio.to_thread(shutil.rmtree, model_path)
            logger.info(f"Model {repo_id} deleted successfully.")
        else:
            logger.warning(f"Model {repo_id} not found in cache.")

    # def load(self, gpu: Optional[int], repo_id: str) -> Optional[DiffusionPipeline]:

    #     if not self.is_downloaded(repo_id):
    #         logger.info(
    #             f" Model {repo_id} not downloaded. Please ensure the model is downloaded first."
    #         )
    #         return None

    #     if repo_id in self.loaded_models:
    #         logger.info(f"Model {repo_id} is already loaded.")
    #         return self.loaded_models[repo_id]

    #     # device = f"cuda:{gpu}" if gpu is not None else "cpu"
    #     try:
    #         pipeline = DiffusionPipeline.from_pretrained(
    #             repo_id,
    #             variant="fp16",
    #             torch_dtype=torch.float16,
    #             cache_dir=self.cache_dir,
    #             local_files_only=True,
    #         )
    #         # pipeline = pipeline.to(device)
    #         self.loaded_models[repo_id] = pipeline
    #         # logger.info(f"Model {repo_id} loaded to {device}.")
    #         return pipeline
    #     except Exception:
    #         logger.error("Attempting to load default variant...")
    #         try:
    #             pipeline = DiffusionPipeline.from_pretrained(
    #                 repo_id,
    #                 cache_dir=self.cache_dir,
    #                 local_files_only=True,
    #                 torch_dtype=torch.float16,
    #             )
    #             # pipeline = pipeline.to(device)
    #             self.loaded_models[repo_id] = pipeline
    #             # logger.info(f"Model {repo_id} loaded to {device}.")
    #             return pipeline
    #         except Exception as e:
    #             logger.error(f"Failed to load model {repo_id}: {str(e)}")
    #             logger.info(
    #                 f"\nFailed to load model {repo_id}. This might be due to partial download or the model is not downloaded."
    #                 " Try downloading (or redownloading) the model first by calling the download method. Use `manager.download(repo_id)`.\n"
    #             )
    #         return None

    # def unload(self, repo_id: str) -> None:
    #     if repo_id in self.loaded_models:
    #         del self.loaded_models[repo_id]

    #         # Clear cache for gpu were model was loaded

    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
    #         elif torch.backends.mps.is_available():
    #             torch.mps.empty_cache()
    #         else:
    #             logger.warning("No GPU available to clear cache.")

    #         logger.info(f"Model {repo_id} unloaded.")
    #     else:
    #         logger.warning(f"Model {repo_id} is not currently loaded.")

    # def list_loaded(self, gpu: Optional[int] = None) -> List[str]:
    #     if gpu is not None:
    #         return [
    #             repo_id
    #             for repo_id, pipeline in self.loaded_models.items()
    #             if pipeline.device.index == gpu
    #         ]
    #     return list(self.loaded_models.keys())

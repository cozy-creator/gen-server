import os
import shutil
import asyncio
from typing import Optional, List, Dict, Any
from huggingface_hub import snapshot_download, HfApi, hf_hub_download, scan_cache_dir
from huggingface_hub.file_download import repo_folder_name
from diffusers import DiffusionPipeline
import torch
import logging
from pathlib import Path
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
import json


logger = logging.getLogger(__name__)


class HFModelManager:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = HF_HUB_CACHE
        # self.cache_dir = None
        self.loaded_models: Dict[str, DiffusionPipeline] = {}
        self.hf_api = HfApi()

    # def is_downloaded(self, repo_id: str) -> bool:
    #     """
    #     Checks if a model is fully downloaded in the cache.
    #     Returns True only if all files for the latest revision of the model are present and complete.
    #     """
    #     try:
    #         # Get the storage folder for the repo
    #         storage_folder = os.path.join(self.cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model"))
            
    #         if not os.path.exists(storage_folder):
    #             return False

    #         # Get the latest commit hash
    #         refs_path = os.path.join(storage_folder, "refs", "main")
    #         if not os.path.exists(refs_path):
    #             return False
            
    #         with open(refs_path, "r") as f:
    #             commit_hash = f.read().strip()

    #         # Check the snapshot folder for this commit
    #         snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
    #         if not os.path.exists(snapshot_folder):
    #             return False

    #         # Check if there are any .incomplete files in the entire repo folder
    #         for root, _, files in os.walk(storage_folder):
    #             if any(file.endswith('.incomplete') for file in files):
    #                 return False

    #         # If we've made it this far, the model is fully downloaded
    #         return True

    #     except Exception as e:
    #         logger.error(f"Error checking download status for {repo_id}: {str(e)}")
    #         return False


    def is_downloaded(self, repo_id: str) -> bool:
        """
        Checks if a model is fully downloaded in the cache.
        Returns True if at least one variant is completely downloaded.
        """
        try:
            storage_folder = os.path.join(self.cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model"))
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
                with open(model_index_path, 'r') as f:
                    model_index = json.load(f)
                    required_folders = {
                        k for k, v in model_index.items() 
                        if isinstance(v, list) and len(v) == 2 and v[0] is not None and v[1] is not None
                    }

                # Remove known non-folder keys and ignored folders
                ignored_folders = {'_class_name', '_diffusers_version', 'scheduler', 'feature_extractor', 'tokenizer', 'tokenizer_2', 'tokenizer_3', 'safety_checker'}
                required_folders -= ignored_folders

                # Define variant hierarchy
                variants = ["bf16", "fp8", "fp16", ""]  # empty string for normal variant

                def check_folder_completeness(folder_path: str, variant: str) -> bool:
                    if not os.path.exists(folder_path):
                        return False
                    
                    # if repo_id == "black-forest-labs/FLUX.1-schnell":
                    #     print(required_folders)
                    #     print(variant)
                    #     print(f"{folder_path}:")
                    
                    for root, _, files in os.walk(folder_path):
                        if repo_id == "black-forest-labs/FLUX.1-schnell":
                            print(files)
                        for file in files:
                            if file.endswith('.incomplete'):
                                print("Here")
                                return False

                            
                            
                            if (file.endswith(f"{variant}.safetensors") or 
                                file.endswith(f"{variant}.bin") or
                                (variant == "" and (file.endswith('.safetensors') or file.endswith('.bin')))):
                                if repo_id == "black-forest-labs/FLUX.1-schnell":
                                    print(file)
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
                    if check_variant_completeness(variant):
                        return True

            else:
                # For repos without model_index.json, check the blob folder
                blob_folder = os.path.join(storage_folder, "blobs")
                if os.path.exists(blob_folder):
                    for root, _, files in os.walk(blob_folder):
                        if any(file.endswith('.incomplete') for file in files):
                            return False
                        
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking download status for {repo_id}: {str(e)}")
            return False



    def list(self) -> List[str]:
        cache_info = scan_cache_dir()
        # print(cache_info)
        return [repo.repo_id for repo in cache_info.repos if self.is_downloaded(repo.repo_id)]

    async def download(
        self,
        repo_id: str,
        variant: Optional[str] = "fp16",
        file_name: Optional[str] = None,
        sub_folder: Optional[str] = None,
    ) -> None:
        try:
            if file_name:
                await asyncio.to_thread(
                    hf_hub_download,
                    repo_id,
                    file_name,
                    cache_dir=self.cache_dir,
                    subfolder=sub_folder,
                )
            else:
                await asyncio.to_thread(
                    DiffusionPipeline.download,
                    repo_id,
                    variant=variant,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16,
                )

            # After successful download, update the list in memory
            self.list()  # Call the list method to refresh the cached list

            logger.info(f"Model {repo_id} downloaded successfully.")

        except Exception as e:
            logger.error(f"Failed to download fp16 variant: {str(e)}")
            logger.info("Attempting to download default variant...")
            try:
                await asyncio.to_thread(
                    DiffusionPipeline.download, repo_id, cache_dir=self.cache_dir
                )

                # After successful download, update the list in memory
                self.list()  # Call the list method to refresh the cached list

                logger.info(f"Model {repo_id} downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download model {repo_id}: {str(e)}")

    async def delete(self, repo_id: str) -> None:
        model_path = os.path.join(
            self.cache_dir, "models--" + repo_id.replace("/", "--")
        )
        if os.path.exists(model_path):
            await asyncio.to_thread(shutil.rmtree, model_path)
            logger.info(f"Model {repo_id} deleted successfully.")
        else:
            logger.warning(f"Model {repo_id} not found in cache.")

    def load(self, gpu: Optional[int], repo_id: str) -> Optional[DiffusionPipeline]:
        # print(self.list())

        if not self.is_downloaded(repo_id):
            logger.info(
                f" Model {repo_id} not downloaded. Please ensure the model is downloaded first."
            )
            return None

        if repo_id in self.loaded_models:
            logger.info(f"Model {repo_id} is already loaded.")
            return self.loaded_models[repo_id]

        # device = f"cuda:{gpu}" if gpu is not None else "cpu"
        try:
            pipeline = DiffusionPipeline.from_pretrained(
                repo_id,
                variant="fp16",
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir,
                local_files_only=True,
            )
            # pipeline = pipeline.to(device)
            self.loaded_models[repo_id] = pipeline
            # logger.info(f"Model {repo_id} loaded to {device}.")
            return pipeline
        except Exception as e:
            logger.error(f"Attempting to load default variant...")
            try:
                pipeline = DiffusionPipeline.from_pretrained(
                    repo_id,
                    cache_dir=self.cache_dir,
                    local_files_only=True,
                    torch_dtype=torch.float16,
                )
                # pipeline = pipeline.to(device)
                self.loaded_models[repo_id] = pipeline
                # logger.info(f"Model {repo_id} loaded to {device}.")
                return pipeline
            except Exception as e:
                logger.error(f"Failed to load model {repo_id}: {str(e)}")
                logger.info(
                    f"\nFailed to load model {repo_id}. This might be due to partial download or the model is not downloaded."
                    " Try downloading (or redownloading) the model first by calling the download method. Use `manager.download(repo_id)`.\n"
                )
            return None

    def unload(self, repo_id: str) -> None:
        if repo_id in self.loaded_models:
            del self.loaded_models[repo_id]

            # Clear cache for gpu were model was loaded

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            else:
                logger.warning("No GPU available to clear cache.")

            logger.info(f"Model {repo_id} unloaded.")
        else:
            logger.warning(f"Model {repo_id} is not currently loaded.")

    def list_loaded(self, gpu: Optional[int] = None) -> List[str]:
        if gpu is not None:
            return [
                repo_id
                for repo_id, pipeline in self.loaded_models.items()
                if pipeline.device.index == gpu
            ]
        return list(self.loaded_models.keys())

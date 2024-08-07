import os
import shutil
import asyncio
from typing import Optional, List, Dict
from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch
import logging


logger = logging.getLogger(__name__)


class HFModelManager:
    def __init__(self, cache_dir: Optional[str] = None):
        # self.cache_dir = get_models_dir() or  os.path.expanduser("~/.cache/huggingface/hub")
        self.cache_dir = None
        self.loaded_models: Dict[str, DiffusionPipeline] = {}
        self.hf_api = HfApi()

    def is_downloaded(self, repo_id: str) -> bool:
        try:
            hf_hub_download(
                repo_id,
                "model_index.json",
                cache_dir=self.cache_dir,
                local_files_only=True,
            )
            return True
        except Exception:
            return False

    def list(self) -> List[str]:
        cache_info = scan_cache_dir()
        return [
            repo.repo_id
            for repo in cache_info.repos
            if self.is_downloaded(repo.repo_id)
        ]

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
            torch.cuda.empty_cache()
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

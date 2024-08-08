import asyncio
import threading
from typing import Dict, Optional
from .hf_model_manager import HFModelManager
from ..base_types.pydantic_models import ModelConfig


class DownloadManager:
    def __init__(self, hf_manager: HFModelManager):
        self._hf_manager = hf_manager
        self._pending_downloads = {}

    @property
    def pending_downloads(self):
        return self._pending_downloads

    @pending_downloads.setter
    def pending_downloads(self):
        raise AttributeError("Cannot manually alter pending_downloads")

    def is_downloaded(self, repo_id: str) -> bool:
        return self._hf_manager.is_downloaded(repo_id)

    def is_pending_download(self, repo_id: str) -> bool:
        return repo_id in self._pending_downloads

    async def download_model(
        self,
        repo_id: str,
        variant: Optional[str] = None,
        file_name: Optional[str] = None,
        sub_folder: Optional[str] = None,
    ):
        try:
            await self._hf_manager.download(
                repo_id,
                variant=variant,
                file_name=file_name,
                sub_folder=sub_folder,
            )
        except Exception as e:
            print(f"Failed to download model {repo_id}: {str(e)}")

    async def download_models(self, models: Dict[str, ModelConfig]):
        try:
            self._pending_downloads = {
                repo_id: self.download_model(repo_id, config.variant)
                for repo_id, config in models.items()
                if not self._hf_manager.is_downloaded(repo_id)
            }

            for repo_id, download_model in self._pending_downloads.items():
                try:
                    await download_model
                    del self._pending_downloads[repo_id]
                except Exception as e:
                    print(f"Failed to download model {repo_id}: {str(e)}")
        except Exception as e:
            print(f"Failed to download models: {str(e)}")

    def download_with_event_loop(self, models: Dict[str, ModelConfig]):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(self.download_models(models))
            except Exception as e:
                print(f"Failed to download models: {str(e)}")
            finally:
                loop.close()
        except Exception as e:
            print(f"Failed to download models: {str(e)}")

    def download_models_silently(self, models: Dict[str, ModelConfig]):
        thread = threading.Thread(target=self.download_with_event_loop, args=(models,))
        thread.daemon = True
        thread.start()

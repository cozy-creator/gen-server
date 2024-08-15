import asyncio
import threading
import aiohttp
from urllib.parse import urlparse
import os
from .hf_model_manager import HFModelManager
from ..base_types.pydantic_models import ModelConfig
from ..utils.paths import get_models_dir


class DownloadManager:
    def __init__(self, hf_manager: HFModelManager):
        self._hf_manager = hf_manager
        self._pending_downloads = {}

    @property
    def pending_downloads(self):
        return self._pending_downloads.copy()

    def is_downloaded(self, repo_id: str) -> bool:
        return self._hf_manager.is_downloaded(repo_id)

    def is_pending_download(self, repo_id: str) -> bool:
        return repo_id in self._pending_downloads

    async def download_model(self, model_id: str, config: ModelConfig):
        try:
            if config.source.startswith("hf:"):
                await self._download_hf(config.source[3:])
            elif config.source.startswith("http"):
                await self._download_file(config.source, model_id)

            if hasattr(config, "components"):
                for component_name, component_config in config.components.items():
                    if component_config.source.startswith("hf:"):
                        await self._download_hf(component_config.source[3:])
                    elif component_config.source.startswith("http"):
                        await self._download_file(
                            component_config.source, f"{model_id}_{component_name}"
                        )
        except Exception as e:
            print(f"Failed to download model {model_id}: {str(e)}")

    async def _download_hf(self, hf_path: str):
        parts = hf_path.split("/")
        repo_id = "/".join(parts[:2])
        sub_folder = "/".join(parts[2:-1]) if len(parts) > 3 else None
        file_name = (
            parts[-1] if len(parts) > 2 and not parts[-1].endswith("/") else None
        )

        await self._hf_manager.download(
            repo_id=repo_id, file_name=file_name, sub_folder=sub_folder
        )

    async def _download_file(self, url: str, filename: str):
        # Extract file extension from URL
        parsed_url = urlparse(url)
        file_extension = os.path.splitext(parsed_url.path)[1]

        # If filename doesn't have an extension, add .safetensors if file_extension is unknown
        if "." not in filename:
            filename += ".safetensors" if not file_extension else file_extension
        # If filename already has an extension, keep it as is
        elif not os.path.splitext(filename)[1]:
            filename += file_extension if file_extension else ".safetensors"

        file_path = os.path.join(get_models_dir(), filename)

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(file_path, "wb") as f:
                        while True:
                            chunk = await response.content.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                else:
                    raise Exception(f"Failed to download file: HTTP {response.status}")

    async def download_models(self, models: dict[str, ModelConfig]):
        try:
            self._pending_downloads = {
                model_id: self.download_model(model_id, config)
                for model_id, config in models.items()
            }

            for model_id, download_task in self._pending_downloads.items():
                try:
                    await download_task
                except Exception as e:
                    print(f"Failed to download model {model_id}: {str(e)}")
                finally:
                    del self._pending_downloads[model_id]
        except Exception as e:
            print(f"Failed to download models: {str(e)}")

    # ... (rest of the methods remain unchanged)

    def download_with_event_loop(self, models: dict[str, ModelConfig]):
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

    def download_models_silently(self, models: dict[str, ModelConfig]):
        thread = threading.Thread(target=self.download_with_event_loop, args=(models,))
        thread.daemon = True
        thread.start()

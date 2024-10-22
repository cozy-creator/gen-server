import os
import shutil
import asyncio
import aiohttp
from typing import Optional, List, Tuple, Dict, Any
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir, snapshot_download
from huggingface_hub.file_download import repo_folder_name
import torch
import logging
from huggingface_hub.constants import HF_HUB_CACHE
import json
from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse
from ..config import get_config
from ..utils.utils import serialize_config
import hashlib
import time


logger = logging.getLogger(__name__)

class ModelSource:
    """Represents a model source with its type and details"""
    def __init__(self, source_str: str):
        self.original_string = source_str
        if source_str.startswith("hf:"):
            self.type = "huggingface"
            self.location = source_str[3:]
        elif "civitai.com" in source_str:
            self.type = "civitai"
            self.location = source_str
        elif source_str.startswith(("http://", "https://")):
            self.type = "direct"
            self.location = source_str
        else:
            raise ValueError(f"Unsupported model source: {source_str}")

class ModelManager:
    def __init__(self, cache_dir: Optional[str] = None):
        self.hf_api = HfApi()
        self.cache_dir = cache_dir or HF_HUB_CACHE
        self.base_cache_dir = cache_dir or os.path.expanduser("~/.cache")
        self.cozy_cache_dir = os.path.join(self.base_cache_dir, "cozy-creator", "models")
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            self.session = None

    def parse_hf_string(
        self, hf_string: str
    ) -> tuple[str, Optional[str], Optional[str]]:
        """
        Parses an HuggingFace string into its components.
        Returns a tuple of (repo_id, subfolder, filename)
        """
        # Remove 'hf:' prefix if present
        if hf_string.startswith("hf:"):
            hf_string = hf_string[3:]

        parts = hf_string.split("/")
        if len(parts) < 2:
            raise ValueError("Invalid HuggingFace string: repo_id is required")

        repo_id = "/".join(parts[:2])
        subfolder = None
        filename = None

        if len(parts) > 2:
            if not parts[-1].endswith("/"):
                filename = parts[-1]
                subfolder = "/".join(parts[2:-1]) if len(parts) > 3 else None
            else:
                subfolder = "/".join(parts[2:])

        return repo_id, subfolder, filename

    async def is_downloaded(self, model_id: str) -> tuple[bool, Optional[str]]:
        """Check if a model is downloaded, handling all source types"""
        try:
            config = serialize_config(get_config())
            model_info = config["enabled_models"].get(model_id)
            if not model_info:
                logger.error(f"Model {model_id} not found in configuration.")
                return False, None

            source = ModelSource(model_info["source"])

            # Check main source
            if source.type == "huggingface":
                is_downloaded, variant = self._check_repo_downloaded(source.location)
                print(f"Repo {source.location} is downloaded: {is_downloaded}, variant: {variant}")
                return is_downloaded, variant
            else:
                return self._check_file_downloaded(self._get_cache_path(model_id, source)), None

        except Exception as e:
            logger.error(f"Error checking download status for {model_id}: {e}")
            return False, None

    async def download_model(self, model_id: str, source: ModelSource):
        """Download a model from any source"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async with context.")

        if source.type == "huggingface":
            return await self.download(source.location)
        elif source.type == "civitai":
            return await self._download_civitai(model_id, source.location)
        else:
            return await self._download_direct(model_id, source.location)

    async def _download_civitai(self, model_id: str, url: str):
        """Handle Civitai-specific download logic"""
        # Convert to API URL if needed
        if "/api/download/" not in url:
            model_path = urlparse(url).path
            model_number = model_path.split("/models/")[1].split("/")[0]
            api_url = f"https://civitai.com/api/v1/models/{model_number}"
            
            async with self.session.get(api_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get Civitai model info: {response.status}")
                data = await response.json()
                # Extract download URL from the first version
                if "modelVersions" in data and len(data["modelVersions"]) > 0:
                    download_url = data["modelVersions"][0]["downloadUrl"]
                else:
                    raise Exception("No model versions found in Civitai response")
        else:
            download_url = url

        await self._download_direct(model_id, download_url)

    async def _download_direct(self, model_id: str, url: str):
        """Download from direct URL with progress bar"""
        dest_path = self._get_cache_path(model_id, ModelSource(url))
        temp_path = dest_path + '.tmp'
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        async with self.session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Download failed with status {response.status}")

            total_size = int(response.headers.get('content-length', 0))
            
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                try:
                    with open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            pbar.update(len(chunk))

                    # Verify download
                    if await self._verify_file(temp_path):
                        os.rename(temp_path, dest_path)
                    else:
                        raise Exception("File verification failed")

                except Exception as e:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise e

    async def _verify_file(self, path: str) -> bool:
        """Verify downloaded file integrity"""
        if not os.path.exists(path):
            print(f"File {path} does not exist")
            return False

        # Size check
        if os.path.getsize(path) < 1024 * 1024:  # 1MB minimum
            print(f"File {path} is too small")
            return False

        # Extension check
        valid_extensions = {'.safetensors', '.ckpt', '.pt', '.bin'}
        if not any(path.endswith(ext) for ext in valid_extensions):
            print(f"File {path} has invalid extension")
            return False

        print(f"File {path} is valid")
        return True

    def _get_cache_path(self, model_id: str, source: ModelSource) -> str:
        """Get the cache path for a model"""
        if source.type == "huggingface":
            return os.path.join(HF_HUB_CACHE, 
                              repo_folder_name(source.location, "model"))

        # For non-HF models
        safe_name = model_id.replace('/', '-')
        url_hash = hashlib.sha256(source.location.encode()).hexdigest()[:8]
        

        # Create model directory with hash
        model_dir = os.path.join(self.cozy_cache_dir, f"{safe_name}--{url_hash}")
        os.makedirs(model_dir, exist_ok=True)

        # Get filename from URL, fallback to safe name if none
        url_path = urlparse(source.location).path
        filename = os.path.basename(url_path) if url_path else f"{safe_name}.safetensors"
        
        # Use hash for the final filename to avoid duplicates
        base, ext = os.path.splitext(filename)
        final_filename = f"{base}_{url_hash}{ext}"
            
        return os.path.join(model_dir, final_filename)


    def _check_repo_downloaded(self, repo_id: str) -> bool:
        storage_folder = os.path.join(
            self.cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model")
        )

        print(f"Storage Folder: {storage_folder}")
        if not os.path.exists(storage_folder):
            return False, None

        # Get the latest commit hash
        refs_path = os.path.join(storage_folder, "refs", "main")
        if not os.path.exists(refs_path):
            return False, None

        with open(refs_path, "r") as f:
            commit_hash = f.read().strip()

        snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
        if not os.path.exists(snapshot_folder):
            return False, None

        # Check model_index.json for required folders
        model_index_path = os.path.join(snapshot_folder, "model_index.json")

        if os.path.exists(model_index_path):
            with open(model_index_path, "r") as f:
                model_index = json.load(f)
                print(model_index)
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
                        if file.endswith(".incomplete"):
                            print(f"Incomplete File: {file}")
                            return False

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
                    print(f"Variant {variant} is complete")
                    return True, variant

        else:
            # For repos without model_index.json, check the blob folder
            blob_folder = os.path.join(storage_folder, "blobs")
            if os.path.exists(blob_folder):
                for _root, _, files in os.walk(blob_folder):
                    if any(file.endswith(".incomplete") for file in files):
                        return False, None

                return True, None

        return False, None

    def _check_component_downloaded(self, repo_id: str, component_name: str) -> bool:
        storage_folder = os.path.join(
            self.cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model")
        )

        if not os.path.exists(storage_folder):
            return False

        refs_path = os.path.join(storage_folder, "refs", "main")
        if not os.path.exists(refs_path):
            return False

        with open(refs_path, "r") as f:
            commit_hash = f.read().strip()

        component_folder = os.path.join(
            storage_folder, "snapshots", commit_hash, component_name
        )

        if not os.path.exists(component_folder):
            return False

        # Check for any .bin, .safetensors, or .ckpt file in the component folder
        for _, _, files in os.walk(component_folder):
            for file in files:
                if file.endswith(
                    (".bin", ".safetensors", ".ckpt")
                ) and not file.endswith(".incomplete"):
                    return True

        return False

    # def _check_file_downloaded(self, file_path: str) -> bool:
    #     # Keep only the name between and after the first slash including the slash
    #     repo_folder = os.path.dirname(file_path)

    #     storage_folder = os.path.join(
    #         self.cache_dir, repo_folder_name(repo_id=repo_folder, repo_type="model")
    #     )

    #     # Get the safetensors file name by splitting the repo_id by '/' and getting the last element
    #     weights_name = file_path.split("/")[-1]

    #     if not os.path.exists(storage_folder):
    #         storage_folder = os.path.join(self.cache_dir, repo_folder)
    #         if not os.path.exists(storage_folder):
    #             return False
    #         else:
    #             full_path = os.path.join(storage_folder, weights_name)
    #             return os.path.exists(full_path) and not full_path.endswith(
    #                 ".incomplete"
    #             )

    #     refs_path = os.path.join(storage_folder, "refs", "main")
    #     if not os.path.exists(refs_path):
    #         return False

    #     with open(refs_path, "r") as f:
    #         commit_hash = f.read().strip()

    #     full_path = os.path.join(storage_folder, "snapshots", commit_hash, weights_name)
    #     return os.path.exists(full_path) and not full_path.endswith(".incomplete")

    def _check_file_downloaded(self, path: str) -> bool:
            """Check if a file exists and is complete in the cache"""
            if not os.path.exists(path):
                print(f"File {path} does not exist")
                return False

            # Check for temporary file
            if os.path.exists(f"{path}.tmp"):
                print(f"File {path}.tmp exists")
                return False

            # Check for .incomplete file (similar to HF's approach)
            if os.path.exists(f"{path}.incomplete"):
                print(f"File {path}.incomplete exists")
                return False

            print(f"File {path} is valid")
            return True


    def list(self) -> List[str]:
        cache_info = scan_cache_dir()
        return [
            repo.repo_id
            for repo in cache_info.repos
            if self.is_downloaded(repo.repo_id)[0]
        ]

    async def download(
        self,
        repo_id: str,
        file_name: Optional[str] = None,
        sub_folder: Optional[str] = None,
    ) -> None:
        if file_name or sub_folder:
            try:
                if sub_folder and not file_name:
                    await asyncio.to_thread(
                        snapshot_download,
                        repo_id,
                        allow_patterns=f"{sub_folder}/*",
                    )
                    logger.info(
                        f"{sub_folder} subfolder from {repo_id} downloaded successfully."
                    )
                else:
                    await asyncio.to_thread(
                        hf_hub_download,
                        repo_id,
                        file_name,
                        cache_dir=self.cache_dir,
                        subfolder=sub_folder,
                    )
                    logger.info(
                        f"File {file_name} from {repo_id} downloaded successfully."
                    )
                # self.list()  # Refresh the cached list
                return True
            except Exception as e:
                logger.error(f"Failed to download file {file_name} from {repo_id}: {e}")
                return False

        variants = ["bf16", "fp8", "fp16", None]  # None represents no variant
        for var in variants:
            try:
                if var:
                    logger.info(
                        f"Attempting to download {repo_id} with {var} variant..."
                    )
                else:
                    logger.info(f"Attempting to download {repo_id} without variant...")

                await asyncio.to_thread(
                    DiffusionPipeline.download,
                    repo_id,
                    variant=var,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16,
                )

                logger.info(
                    f"Model {repo_id} downloaded successfully with variant: {var if var else 'default'}"
                )
                # self.list()  # Refresh the cached list
                return True

            except Exception as e:
                if var:
                    logger.error(
                        f"Failed to download {var} variant for {repo_id}. Trying next variant..."
                    )
                else:
                    logger.error(
                        f"Failed to download default variant for {repo_id}: {e}"
                    )

        logger.error(f"Failed to download model {repo_id} with any variant.")
        return False

    async def delete(self, repo_id: str) -> None:
        model_path = os.path.join(
            self.cache_dir, "models--" + repo_id.replace("/", "--")
        )
        if os.path.exists(model_path):
            await asyncio.to_thread(shutil.rmtree, model_path)
            logger.info(f"Model {repo_id} deleted successfully.")
        else:
            logger.warning(f"Model {repo_id} not found in cache.")

    async def get_diffusers_multifolder_components(
        self, repo_id: str
    ) -> dict[str, str | tuple[str, str]] | None:
        """
        This is only meaningful if the repo is in diffusers-multifolder layout.
        This retrieves and parses the model_index.json file, and None otherwise.
        """
        try:
            model_index_path = await asyncio.to_thread(
                hf_hub_download,
                repo_id=repo_id,
                filename="model_index.json",
                cache_dir=self.cache_dir,
            )

            if model_index_path:
                with open(model_index_path, "r") as f:
                    data = json.load(f)
                    return {
                        k: tuple(v) if isinstance(v, list) else v
                        for k, v in data.items()
                    }
            else:
                return None
        except Exception as e:
            logger.error(f"Error retrieving model_index.json for {repo_id}: {e}")
            return None
        
    # Keep all your existing HF-specific methods:
    # - get_diffusers_multifolder_components
    # - _check_repo_downloaded
    # - download
    # - delete
    # etc.
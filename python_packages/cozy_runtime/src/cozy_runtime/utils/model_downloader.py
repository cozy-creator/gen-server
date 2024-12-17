import os
import shutil
import asyncio
import aiohttp
from typing import Optional, List
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from .paths import get_models_dir
from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir, snapshot_download
from huggingface_hub.file_download import repo_folder_name
import torch
import logging
from huggingface_hub.constants import HF_HUB_CACHE
import json
from tqdm import tqdm
from ..config import get_config
from ..utils.utils import serialize_config
import hashlib
import time
import re
from urllib.parse import urlparse, parse_qs, unquote
import backoff


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
        elif source_str.startswith("file:"):
            self.type = "file"
            self.location = source_str[5:]
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
        self.cozy_cache_dir = get_models_dir()
        self.session: Optional[aiohttp.ClientSession] = None

        # config = serialize_config(get_config())
        # self.civitai_api_key = config["civitai_api_key"]

        # check env for civitai api key
        self.civitai_api_key = os.getenv("CIVITAI_API_KEY")

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

    async def _get_civitai_filename(self, url: str) -> Optional[str]:
        """Extract original filename from Civitai redirect response"""
        try:
            headers = {}
            if self.civitai_api_key:
                headers["Authorization"] = f"Bearer {self.civitai_api_key}"

            need_cleanup = False
            if not self.session:
                self.session = aiohttp.ClientSession()
                need_cleanup = True

            try:
                async with self.session.get(
                    url, headers=headers, allow_redirects=False
                ) as response:
                    if response.status in (301, 302, 307):
                        location = response.headers.get("location")
                        if location:
                            # Parse the query parameters from the redirect URL
                            parsed = urlparse(location)
                            query_params = parse_qs(parsed.query)

                            # Look for response-content-disposition parameter
                            content_disp = query_params.get(
                                "response-content-disposition", [None]
                            )[0]
                            if content_disp:
                                # Extract filename from content disposition
                                match = re.search(r'filename="([^"]+)"', content_disp)
                                if match:
                                    return unquote(match.group(1))

                            # Fallback to path if no content disposition
                            path = parsed.path
                            if path:
                                return os.path.basename(path)

                return None
            finally:
                # Clean up the session if we created it
                if need_cleanup and self.session:
                    await self.session.close()
                    self.session = None

        except Exception as e:
            logger.error(f"Error getting Civitai filename: {e}")
            # Make sure to clean up session on error if we created it
            if "need_cleanup" in locals() and need_cleanup and self.session:
                await self.session.close()
                self.session = None
            return None

    async def is_downloaded(self, model_id: str, model_config: Optional[dict] = None) -> tuple[bool, Optional[str]]:
        """Check if a model is downloaded, handling all source types including Civitai filename variants"""
        try:
            config = serialize_config(get_config())
            model_info = config["pipeline_defs"].get(model_id)
            if not model_info:
                logger.error(f"Model {model_id} not found in configuration.")
                return False, None

            source = ModelSource(model_info["source"])

            # Get components from model_config
            if model_config:
                components = model_config.get("components", [])
                # add the component names to an array
                component_names = [component for component in components]
                print(f"Component names: {component_names}")

            # Handle local files - just check if they exist
            if source.type == "file":
                exists = os.path.exists(source.location)
                if not exists:
                    logger.error(f"Local file not found: {source.location}")
                return exists, None

            # Handle HuggingFace models as before
            if source.type == "huggingface":
                is_downloaded, variant = self._check_repo_downloaded(source.location, component_names)
                print(
                    f"Repo {source.location} is downloaded: {is_downloaded}, variant: {variant}"
                )
                return is_downloaded, variant

            # Special handling for Civitai models
            elif source.type == "civitai":
                # First check the default numeric ID path
                default_path = await self._get_cache_path(model_id, source)
                if self._check_file_downloaded(default_path):
                    return True, None

                # If not found, try to get the original filename
                if not self.session:
                    self.session = aiohttp.ClientSession()
                    need_cleanup = True
                else:
                    need_cleanup = False

                try:
                    original_filename = await self._get_civitai_filename(
                        source.location
                    )
                    if original_filename:
                        dir_path = os.path.dirname(default_path)
                        alternate_path = os.path.join(dir_path, original_filename)
                        if self._check_file_downloaded(alternate_path):
                            return True, None
                finally:
                    if need_cleanup and self.session:
                        await self.session.close()
                        self.session = None

                return False, None

            # Handle direct downloads
            else:
                cache_path = await self._get_cache_path(model_id, source)
                return self._check_file_downloaded(cache_path), None

        except Exception as e:
            logger.error(f"Error checking download status for {model_id}: {e}")
            return False, None

    def _get_model_directory(self, model_id: str, url_hash: str) -> str:
        """Get the directory path for a model"""
        safe_name = model_id.replace("/", "-")
        return os.path.join(self.cozy_cache_dir, f"{safe_name}--{url_hash}")

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
        """Handle Civitai-specific download logic with proper filename handling"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async with context.")

        # Convert to API URL if needed
        if "/api/download/" not in url:
            model_path = urlparse(url).path
            model_number = model_path.split("/models/")[1].split("/")[0]
            api_url = f"https://civitai.com/api/v1/models/{model_number}"

            headers = {}
            if self.civitai_api_key:
                headers["Authorization"] = f"Bearer {self.civitai_api_key}"

            async with self.session.get(api_url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(
                        f"Failed to get Civitai model info: {response.status}"
                    )
                data = await response.json()
                # Extract download URL from the first version
                if "modelVersions" in data and len(data["modelVersions"]) > 0:
                    download_url = data["modelVersions"][0]["downloadUrl"]
                else:
                    raise Exception("No model versions found in Civitai response")
        else:
            download_url = url

        # Get original filename from redirect
        original_filename = await self._get_civitai_filename(download_url)
        if original_filename:
            # Update the cache path with the original filename
            dest_path = await self._get_cache_path(model_id, ModelSource(download_url))
            if original_filename != os.path.basename(dest_path):
                dir_path = os.path.dirname(dest_path)
                dest_path = os.path.join(dir_path, original_filename)

        # Download with the correct filename
        await self._download_direct(model_id, download_url, dest_path)

    @backoff.on_exception(
        backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=3
    )
    async def _download_direct(
        self, model_id: str, url: str, dest_path: Optional[str] = None
    ):
        """Download from direct URL with progress bar, retry logic, and resume capability"""
        if dest_path is None:
            dest_path = await self._get_cache_path(model_id, ModelSource(url))

        temp_path = dest_path + ".tmp"

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        headers = {}
        if self.civitai_api_key:
            headers["Authorization"] = f"Bearer {self.civitai_api_key}"

        # Check if we have a partial download
        initial_size = 0
        if os.path.exists(temp_path):
            initial_size = os.path.getsize(temp_path)
            if initial_size > 0:
                headers["Range"] = f"bytes={initial_size}-"
                logger.info(f"Resuming download from byte {initial_size}")

        timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=60)

        try:
            async with self.session.get(
                url, headers=headers, timeout=timeout
            ) as response:
                # Handle resume responses
                if initial_size > 0:
                    if response.status == 206:  # Partial Content, resume successful
                        total_size = initial_size + int(
                            response.headers.get("content-length", 0)
                        )
                    elif response.status == 200:  # Server doesn't support resume
                        logger.warning(
                            "Server doesn't support resume, starting from beginning"
                        )
                        total_size = int(response.headers.get("content-length", 0))
                        initial_size = 0
                    else:
                        raise Exception(f"Resume failed with status {response.status}")
                else:
                    if response.status != 200:
                        raise Exception(
                            f"Download failed with status {response.status}"
                        )
                    total_size = int(response.headers.get("content-length", 0))

                # Open file in append mode if resuming, write mode if starting fresh
                mode = "ab" if initial_size > 0 else "wb"
                downloaded_size = initial_size
                last_progress_update = time.time()
                stall_timer = 0

                with tqdm(
                    total=total_size, initial=initial_size, unit="iB", unit_scale=True
                ) as pbar:
                    try:
                        with open(temp_path, mode) as f:
                            async for chunk in response.content.iter_chunked(8192):
                                if chunk:  # filter out keep-alive chunks
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    pbar.update(len(chunk))

                                    # Check for download stalls
                                    current_time = time.time()
                                    if (
                                        current_time - last_progress_update > 30
                                    ):  # 30 seconds without progress
                                        stall_timer += (
                                            current_time - last_progress_update
                                        )
                                        if (
                                            stall_timer > 120
                                        ):  # 2 minutes total stall time
                                            raise Exception(
                                                "Download stalled for too long"
                                            )
                                    else:
                                        stall_timer = 0
                                        last_progress_update = current_time

                        # Verify downloaded size
                        if total_size > 0 and downloaded_size != total_size:
                            raise Exception(
                                f"Download incomplete. Expected {total_size} bytes, got {downloaded_size} bytes"
                            )

                        # Verify file integrity
                        if await self._verify_file(temp_path):
                            os.rename(temp_path, dest_path)
                            logger.info(
                                f"Downloaded and saved as: {os.path.basename(dest_path)}"
                            )
                        else:
                            raise Exception(
                                "File verification failed - will attempt resume on next try"
                            )

                    except Exception as e:
                        logger.error(
                            f"Download error (temporary file kept for resume): {str(e)}"
                        )
                        raise

        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            raise

    async def _verify_file(self, path: str) -> bool:
        """Verify downloaded file integrity with more thorough checks"""
        try:
            if not os.path.exists(path):
                logger.error(f"File {path} does not exist")
                return False

            # Size check
            file_size = os.path.getsize(path)
            if file_size < 1024 * 1024:  # 1MB minimum
                logger.error(f"File {path} is too small: {file_size} bytes")
                return False

            # Extension check - we check the final intended path, not the .tmp path
            check_path = path[:-4] if path.endswith(".tmp") else path
            valid_extensions = {".safetensors", ".ckpt", ".pt", ".bin"}
            if not any(check_path.endswith(ext) for ext in valid_extensions):
                logger.error(f"File {check_path} has invalid extension")
                return False

            # Try to open the file to ensure it's not corrupted
            with open(path, "rb") as f:
                # Read first and last 1MB to check file accessibility
                f.read(1024 * 1024)
                f.seek(-1024 * 1024, 2)
                f.read(1024 * 1024)

            logger.info(f"File {path} passed all verification checks")
            return True

        except Exception as e:
            logger.error(f"File verification failed: {str(e)}")
            return False

    async def _get_cache_path(self, model_id: str, source: ModelSource) -> str:
        """Get the cache path for a model"""
        if source.type == "huggingface":
            return os.path.join(
                HF_HUB_CACHE, repo_folder_name(source.location, "model")
            )

        # For non-HF models
        safe_name = model_id.replace("/", "-")
        url_hash = hashlib.sha256(source.location.encode()).hexdigest()[:8]

        # Create model directory with hash
        model_dir = os.path.join(self.cozy_cache_dir, f"{safe_name}--{url_hash}")
        os.makedirs(model_dir, exist_ok=True)

        if source.type == "civitai":
            # Try to get original filename from Civitai
            print(f"Getting Civitai filename for {source.location}")
            original_filename = await self._get_civitai_filename(source.location)
            if original_filename:
                return os.path.join(model_dir, original_filename)

        # Fallback for direct downloads or if couldn't get Civitai filename
        url_path = urlparse(source.location).path
        filename = (
            os.path.basename(url_path) if url_path else f"{safe_name}.safetensors"
        )

        # Use hash for the final filename to avoid duplicates
        base, ext = os.path.splitext(filename)
        final_filename = f"{base}_{url_hash}{ext}"

        return os.path.join(model_dir, final_filename)

    def _check_repo_downloaded(self, repo_id: str, component_names: Optional[List[str]] = None) -> bool:
        storage_folder = os.path.join(
            self.cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model")
        )

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
            if component_names:
                required_folders -= set(component_names)

            print(f"Required folders: {required_folders}")

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


    def _check_file_downloaded(self, path: str) -> bool:
        """Check if a file exists and is complete in the cache"""
        # First check if the exact path exists
        if os.path.exists(path):
            # Check for temporary or incomplete markers
            if os.path.exists(f"{path}.tmp") or os.path.exists(f"{path}.incomplete"):
                print(f"Found incomplete markers for {path}")
                return False
            print(f"Found complete file at {path}")
            return True

        # If path doesn't exist, check the directory for any valid model files
        dir_path = os.path.dirname(path)
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if file.endswith((".safetensors", ".ckpt", ".pt", ".bin")):
                    if not os.path.exists(f"{file_path}.tmp") and not os.path.exists(
                        f"{file_path}.incomplete"
                    ):
                        print(f"Found alternative model file at {file_path}")
                        return True

        print(f"No valid model files found in {dir_path}")
        return False

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


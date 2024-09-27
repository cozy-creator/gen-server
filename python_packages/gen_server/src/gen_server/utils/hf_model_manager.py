import os
import shutil
import asyncio
from typing import Optional, List, Dict, Any
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir, snapshot_download
from huggingface_hub.file_download import repo_folder_name
import torch
import logging
from huggingface_hub.constants import HF_HUB_CACHE
import json
from ..config import get_config
from ..utils.utils import serialize_config

logger = logging.getLogger(__name__)


class HFModelManager:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or HF_HUB_CACHE
        self.loaded_models: dict[str, DiffusionPipeline] = {}
        self.hf_api = HfApi()

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
            logger.error(f"Error retrieving model_index.json for {repo_id}: {str(e)}")
            return None
        

    def is_downloaded(self, model_id: str) -> tuple[bool, Optional[str]]:
        """
        Checks if a model is fully downloaded in the cache.
        Returns True if at least one variant is completely downloaded.
        """
        try:
            # Get the repo_id from the YAML configuration
            config = get_config()

            config = serialize_config(config)
            # print(f"Config: {config}")
            model_info = config["enabled_models"].get(model_id)
            if not model_info:
                logger.error(f"Model {model_id} not found in configuration.")
                return False, None

            print(f"Model Config: {model_info}")

            if model_info["source"]:
                repo_id = model_info["source"].replace("hf:", "")
            else:
                repo_id = model_id

            print(f"Repo ID: {repo_id}")

            # Check main repo
            repo_downloaded, variant = self._check_repo_downloaded(repo_id)
            if not repo_downloaded:
                return False, None

            # Check components if model_index is present
            if "components" in model_info and model_info["components"] is not None:
                for component_name, source in model_info["components"].items():
                    if isinstance(source, list):
                        component_repo = source[0]
                        if not self._check_component_downloaded(
                            component_repo, component_name
                        ):
                            print(
                                f"Component {component_name} from {component_repo} is not downloaded."
                            )
                            logger.info(
                                f"Component {component_name} from {component_repo} is not downloaded."
                            )
                            return False, None
                    elif isinstance(source, str) and source.endswith(
                        (".safetensors", ".bin", ".ckpt")
                    ):
                        if not self._check_file_downloaded(source):
                            print(f"Custom component file {source} is not downloaded.")
                            logger.info(
                                f"Custom component file {source} is not downloaded."
                            )
                            return False, None

            return True, variant

        except Exception as e:
            logger.error(f"Error checking download status for {model_id}: {str(e)}")
            return False, None

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
        for root, _, files in os.walk(component_folder):
            for file in files:
                if file.endswith(
                    (".bin", ".safetensors", ".ckpt")
                ) and not file.endswith(".incomplete"):
                    return True

        return False

    def _check_file_downloaded(self, file_path: str) -> bool:
        # Keep only the name between and after the first slash including the slash
        repo_folder = os.path.dirname(file_path)

        storage_folder = os.path.join(
            self.cache_dir, repo_folder_name(repo_id=repo_folder, repo_type="model")
        )

        # Get the safetensors file name by splitting the repo_id by '/' and getting the last element
        weights_name = file_path.split("/")[-1]

        if not os.path.exists(storage_folder):
            storage_folder = os.path.join(self.cache_dir, repo_folder)
            if not os.path.exists(storage_folder):
                return False
            else:
                full_path = os.path.join(storage_folder, weights_name)
                return os.path.exists(full_path) and not full_path.endswith(
                    ".incomplete"
                )

        refs_path = os.path.join(storage_folder, "refs", "main")
        if not os.path.exists(refs_path):
            return False

        with open(refs_path, "r") as f:
            commit_hash = f.read().strip()

        full_path = os.path.join(storage_folder, "snapshots", commit_hash, weights_name)
        return os.path.exists(full_path) and not full_path.endswith(".incomplete")

    def list(self) -> List[str]:
        cache_info = scan_cache_dir()
        return [
            repo.repo_id
            for repo in cache_info.repos
            if self.is_downloaded(repo.repo_id)[0]
        ]

    # def get_model_index(self, repo_id: str) -> Dict[str, Any]:
    #     storage_folder = os.path.join(
    #         self.cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model")
    #     )

    #     if not os.path.exists(storage_folder):
    #         raise FileNotFoundError(f"Cache folder for {repo_id} not found")

    #     # Get the latest commit hash
    #     refs_path = os.path.join(storage_folder, "refs", "main")
    #     if not os.path.exists(refs_path):
    #         raise FileNotFoundError(f"refs/main not found for {repo_id}")

    #     with open(refs_path, "r") as f:
    #         commit_hash = f.read().strip()

    #     # Construct the path to model_index.json
    #     model_index_path = os.path.join(
    #         storage_folder, "snapshots", commit_hash, "model_index.json"
    #     )

    #     if not os.path.exists(model_index_path):
    #         raise FileNotFoundError(f"model_index.json not found for {repo_id}")

    #     with open(model_index_path, "r") as f:
    #         return json.load(f)

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
                logger.error(
                    f"Failed to download file {file_name} from {repo_id}: {str(e)}"
                )
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
                        f"Failed to download default variant for {repo_id}: {str(e)}"
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



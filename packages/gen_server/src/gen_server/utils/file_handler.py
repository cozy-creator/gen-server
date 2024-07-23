import io
import os
import asyncio

import aiofiles
import aioshutil
from typing import Union, Optional, TypedDict, AsyncGenerator, Any
import logging
import aioboto3
import blake3
from PIL import Image, PngImagePlugin
from abc import ABC, abstractmethod

from ..base_types.pydantic_models import FilesystemTypeEnum, RunCommandConfig
from ..config import get_config, is_runpod_available, get_runpod_url
from .paths import get_assets_dir, get_s3_public_url

logger = logging.getLogger(__name__)


class FileURL(TypedDict):
    url: str
    is_temp: bool


class FileHandler(ABC):
    @abstractmethod
    async def upload_file(
        self,
        file_content: bytes,
        file_extension: str,
        file_basename: str | None = None,
        is_temp: bool = False,
    ) -> FileURL:
        pass

    async def upload_png_files(
        self,
        images: list[Image.Image],
        metadata: Optional[PngImagePlugin.PngInfo] = None,
        is_temp: bool = False,
    ) -> AsyncGenerator[FileURL, None]:
        """
        This is a convenient method for uploading PNG files, which is a common use-case.
        It wraps the built-in `upload_file` method.
        """
        image_dict: dict[str, bytes] = {}

        for img in images:
            # Convert image to bytes without metadata and compute Blake3 hash
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_hash = blake3.blake3(img_byte_arr.getvalue()).hexdigest()

            # Now add metadata and convert to bytes again
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG", pnginfo=metadata)
            img_bytes = img_byte_arr.getvalue()

            # Store in dictionary
            image_dict[img_hash] = img_bytes

        tasks = [
            self.upload_file(
                file_content=image,
                file_extension="png",
                file_basename=name,
                is_temp=is_temp,
            )
            for name, image in image_dict.items()
        ]

        print(f"Created {len(tasks)} tasks in upload_png_files")

        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            yield result

        print("finished uploading")

    @abstractmethod
    async def delete_files(
        self,
        folder_name: Optional[str] = None,
        file_names: Optional[list[str]] = None,
    ) -> None:
        pass

    @abstractmethod
    async def list_files(self) -> list[str]:
        pass

    @abstractmethod
    async def download_file(self, filename: str) -> Union[bytes, None]:
        pass


class LocalFileHandler(FileHandler):
    def __init__(self, config: RunCommandConfig):
        self.server_url = (
            get_runpod_url(config.port)
            if is_runpod_available()
            else f"http://{config.host}:{config.port}"
        )

        self.assets_path = get_assets_dir()
        if not os.path.exists(self.assets_path):
            os.makedirs(self.assets_path)

    async def upload_file(
        self,
        file_content: bytes,
        file_extension: str,
        file_basename: str | None = None,
        is_temp: bool = False,
    ) -> FileURL:
        if file_basename is None:
            file_basename = blake3.blake3(file_content).hexdigest()

        filename = f"{file_basename}.{file_extension}"
        base_path = (
            os.path.join(self.assets_path, "temp") if is_temp else self.assets_path
        )
        filepath = os.path.join(base_path, filename)

        async with aiofiles.open(filepath, "wb") as f:
            await f.write(file_content)

        return {
            "url": f"{self.server_url}/media/{filename}",
            "is_temp": is_temp,
        }

    async def delete_files(
        self,
        folder_name: Optional[str] = None,
        file_names: Optional[list[str]] = None,
    ):
        """
        Deletes files from the assets directory.

        Parameters:
        - folder_name (Optional[str]): The name of the folder to delete.
        - file_names (Optional[list[str]]): A list of file names to delete.
        """

        if folder_name is None and file_names is None:
            raise ValueError("Either folder_name or file_names must be provided")

        def _delete_files(folder_path: str, files: list[str]):
            for file in files:
                file_path = os.path.join(folder_path, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error while deleting {file_path}: {str(e)}")

        async def _delete_dirs(folder_path: str, dirs: list[str]):
            for dir in dirs:
                dir_path = os.path.join(folder_path, dir)
                try:
                    await aioshutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Error while deleting {dir_path}: {str(e)}")

        async def _empty_folder(folder_path: str):
            for _, dirs, files in os.walk(folder_path, topdown=False):
                _delete_files(folder_path, files)
                await _delete_dirs(folder_path, dirs)

        if file_names is not None:
            folder_path = (
                os.path.join(self.assets_path, folder_name)
                if folder_name
                else self.assets_path
            )
            _delete_files(folder_path, file_names)
        else:
            await _empty_folder(os.path.join(self.assets_path, folder_name))

    async def list_files(self) -> list[str]:
        """
        Lists all uploaded files.
        """

        if not os.path.exists(self.assets_path):
            return []

        def _make_url(file: str):
            return f"{self.server_url}/files/{file}"

        return [_make_url(file) for file in os.listdir(self.assets_path)]

    async def download_file(self, filename: str) -> Union[bytes, None]:
        full_filepath = os.path.join(self.assets_path, filename)

        if not os.path.exists(full_filepath):
            return None

        async with aiofiles.open(full_filepath, mode="rb") as f:
            return await f.read()


class S3FileHandler(FileHandler):
    def __init__(self, config: RunCommandConfig):
        if config.s3 is None:
            raise ValueError("S3 configuration is required")

        self.config = config

        self.session = aioboto3.Session()

    async def _get_client(self):
        return self.session.client(
            self.config.filesystem_type,
            region_name=self.config.s3.region_name,
            endpoint_url=self.config.s3.endpoint_url,
            aws_access_key_id=self.config.s3.access_key,
            aws_secret_access_key=self.config.s3.secret_key,
        )

    async def upload_file(
        self,
        file_content: bytes,
        file_extension: str,
        file_basename: str | None = None,
        is_temp: bool = False,
    ) -> FileURL:
        if file_basename is None:
            file_basename = blake3.blake3(file_content).hexdigest()

        folder_prefix = f"{self.config.s3.folder}/" if self.config.s3.folder else ""
        temp_prefix = "temp/" if is_temp else ""
        key = f"{folder_prefix}{temp_prefix}{file_basename}.{file_extension}"

        try:
            async with await self._get_client() as client:
                await client.put_object(
                    Bucket=self.config.s3.bucket_name,
                    Key=key,
                    Body=file_content,
                    ACL="public-read",
                )

        except Exception as e:
            logger.error(f"Failed to upload file {key}: {e}")
            raise

        return {
            "url": f"{get_s3_public_url()}/{key}",
            "is_temp": is_temp,
        }

    async def apply_temp_config(self, **kwargs):
        """
        Applies a temporary configuration to the bucket, such as setting an expiration rule for temporary files.

        Parameters:
        - id (str): The ID of the rule.
        - days (int): The number of days before the files expire.
        - date (str): The date when the files expire.
        """

        expiration = {}
        if kwargs.get("days") is not None:
            expiration["Days"] = kwargs.get("days")
        elif kwargs.get("date") is not None:
            expiration["Date"] = kwargs.get("date")
        else:
            raise ValueError("Invalid expiration configuration")

        try:
            client = await self._get_client()
            lifecycle_configuration = {
                "Rules": [
                    {
                        "ID": kwargs["id"]
                        if kwargs.get("id")
                        else "TempFilesExpirationRule",
                        "Prefix": "/temp",
                        "Status": "Enabled",
                        "Expiration": expiration,
                    }
                ]
            }

            # Apply the lifecycle configuration to the bucket
            await client.put_bucket_lifecycle_configuration(
                Bucket=self.config.bucket_name,
                LifecycleConfiguration=lifecycle_configuration,
            )

        except Exception as e:
            logger.error(f"Failed to upload file apply temp config: {e}")
            raise

    async def delete_files(
        self,
        folder_name: Optional[str] = None,
        file_names: Optional[list[str]] = None,
    ):
        """
        Delete all objects within a folder in an S3 bucket.

        Parameters:
        - folder_name (Optional[str]): The name of the folder to delete files from.
        - file_names (Optional[list[str]]): A list of file names to delete.
        """

        if folder_name is None and file_names is None:
            raise ValueError("Either folder_name or file_names must be provided")

        print(f"Deleting files in {folder_name}")
        print(f"Files: {file_names}")
        async with await self._get_client() as client:

            async def _delete_objects_in_chunks(objects):
                chunk_size = 1000  # S3 allows up to 1000 objects per delete request
                for i in range(0, len(objects), chunk_size):
                    chunk = objects[i : i + chunk_size]
                    await client.delete_objects(
                        Bucket=self.config.s3.bucket_name,
                        Delete={"Objects": chunk},
                    )

            try:
                continuation_token = None
                folder_name = (
                    f"{folder_name}/"
                    if folder_name is not None and folder_name.endswith("/")
                    else folder_name
                )

                if file_names is not None:
                    objects_to_delete = [
                        {"Key": f"{folder_name}/{name}" if folder_name else name}
                        for name in file_names
                    ]

                    print(f"Deleting {(objects_to_delete)} files")
                    await _delete_objects_in_chunks(objects_to_delete)
                else:
                    while True:
                        list_params = {
                            "Bucket": self.config.s3.bucket_name,
                            "Prefix": folder_name,
                        }
                        if continuation_token:
                            list_params["ContinuationToken"] = continuation_token

                        response = await client.list_objects_v2(**list_params)

                        if "Contents" in response:
                            objects_to_delete = [
                                {"Key": obj["Key"]} for obj in response["Contents"]
                            ]
                            await _delete_objects_in_chunks(objects_to_delete)

                        if response.get("IsTruncated"):
                            continuation_token = response.get("NextContinuationToken")
                        else:
                            break
            except Exception as e:
                print(f"Failed to delete files: {e}")

    async def list_files(self) -> list[str]:
        """
        Lists all uploaded files.
        """

        async with await self._get_client() as client:
            response = await client.list_objects_v2(
                Bucket=self.config.s3.bucket_name,
                Prefix=self.config.s3.folder,
            )

        def _make_url(obj: dict[str, Any]) -> str:
            return f"{self.config.s3.endpoint_url}/{obj['Key']}"

        return [_make_url(obj) for obj in response.get("Contents", [])]

    async def download_file(self, filename: str) -> Union[bytes, None]:
        key = f"{self.config.s3.folder}/{filename}"

        bytesio = io.BytesIO()
        async with await self._get_client() as client:
            await client.download_fileobj(self.config.s3.bucket_name, key, bytesio)
        return bytesio.getvalue()


_file_handler_cache = None
"""
    Implements a singleton pattern so that every call to get_file_handler doesn't create a new instance
    of a FileHandler class.
"""


def get_file_handler(config: Optional[RunCommandConfig] = None) -> FileHandler:
    global _file_handler_cache

    if _file_handler_cache is not None:
        return _file_handler_cache

    if config is None:
        config = get_config()

    if config.filesystem_type == FilesystemTypeEnum.S3:
        _file_handler_cache = S3FileHandler(config)
    elif config.filesystem_type == FilesystemTypeEnum.LOCAL:
        _file_handler_cache = LocalFileHandler(config)
    else:
        raise ValueError("Invalid Filesystem-type")

    return _file_handler_cache


def get_mime_type(file_path: str) -> str:
    """
    Determines the content type based on the file extension.
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    content_types = {
        ".html": "text/html",
        ".css": "text/css",
        ".js": "application/javascript",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".avif": "image/avif",
        ".webp": "image/webp",
        ".ico": "image/x-icon",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".wmv": "video/x-ms-wmv",
        ".flv": "video/x-flv",
        ".mkv": "video/x-matroska",
    }

    return content_types.get(file_extension, "application/octet-stream")


def calculate_blake3_hash(file_bytes: bytes) -> str:
    """
    Calculates the BLAKE3 hash of the given file bytes and returns it as a hexadecimal string.
    """
    hasher = blake3.blake3()
    hasher.update(file_bytes)
    return hasher.hexdigest()

import io
import os
import time
import asyncio
import aiofiles
from typing import Union, Optional, TypedDict, AsyncGenerator, Any
import logging
from boto3.session import Session
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
    def upload_files(
        self,
        content: list[bytes] | dict[str, bytes],
        file_extension: str,
        is_temp: bool = False,
    ) -> AsyncGenerator[FileURL, None]:
        pass

    async def upload_png_files(
        self,
        images: list[Image.Image],
        metadata: Optional[PngImagePlugin.PngInfo] = None,
        is_temp: bool = False,
    ) -> AsyncGenerator[FileURL, None]:
        """
        This is a convenient method for uploading PNG files, which is a common use-case.
        It wraps the built-in `upload_files` method.
        """
        image_dict = {}

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

        results = self.upload_files(
            content=image_dict,
            file_extension="png",
            is_temp=is_temp,
        )

        async for result in results:
            print("result", result)
            yield result

    @abstractmethod
    def list_files(self) -> list[str]:
        pass

    @abstractmethod
    def download_file(self, filename: str) -> Union[bytes, None]:
        pass


class LocalFileHandler(FileHandler):
    def __init__(self, config: RunCommandConfig):
        self.server_url = (
            get_runpod_url(config.port)
            if is_runpod_available()
            else f"http://{config.host}:{config.port}"
        )

        self.assets_path = get_assets_dir()

    async def upload_files(
        self,
        content: list[bytes] | dict[str, bytes],
        file_extension: str,
        is_temp: bool = False,
    ) -> AsyncGenerator[FileURL, None]:
        if not os.path.exists(self.assets_path):
            os.makedirs(self.assets_path)

        if isinstance(content, list):
            content_dict = {blake3.blake3(item).hexdigest(): item for item in content}
        else:
            content_dict = content

        async def _upload_file(basename: str, file_content: bytes) -> str:
            filename = f"{basename}.{file_extension}"
            base_path = (
                os.path.join(self.assets_path, "temp") if is_temp else self.assets_path
            )
            filepath = os.path.join(base_path, filename)
            async with aiofiles.open(filepath, "wb") as f:
                await f.write(file_content)

            return f"{self.server_url}/media/{filename}"

        tasks = {
            asyncio.create_task(_upload_file(basename, file_content)): basename
            for basename, file_content in content_dict.items()
        }

        for completed_task in asyncio.as_completed(tasks):
            url = await completed_task
            yield {"url": url, "is_temp": is_temp}

    def list_files(self) -> list[str]:
        """
        Lists all uploaded files.
        """

        if not os.path.exists(self.assets_path):
            return []

        def _make_url(file: str):
            return f"{self.server_url}/files/{file}"

        return [_make_url(file) for file in os.listdir(self.assets_path)]

    def download_file(self, filename: str) -> Union[bytes, None]:
        full_filepath = os.path.join(self.assets_path, filename)

        if not os.path.exists(full_filepath):
            return None

        with open(full_filepath, mode="rb") as f:
            return f.read()


class S3FileHandler(FileHandler):
    def __init__(self, config: RunCommandConfig):
        if config.s3 is None:
            raise ValueError("S3 configuration is required")

        self.config = config.s3

        session = Session()
        self.client = session.client(
            config.filesystem_type,
            region_name=config.s3.region_name,
            endpoint_url=config.s3.endpoint_url,
            aws_access_key_id=config.s3.access_key,
            aws_secret_access_key=config.s3.secret_key,
        )

    async def upload_files(
        self,
        content: list[bytes] | dict[str, bytes],
        file_extension: str,
        is_temp: bool = False,
    ) -> AsyncGenerator[FileURL, None]:
        if isinstance(content, list):
            content_dict = {blake3.blake3(item).hexdigest(): item for item in content}
        else:
            content_dict = content

        async def _upload_file(basename: str, file_content: bytes) -> str:
            folder_prefix = f"{self.config.folder}/" if self.config.folder else ""
            temp_prefix = "temp/" if is_temp else ""
            key = f"{folder_prefix}{temp_prefix}{basename}.{file_extension}"

            try:
                self.client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    Body=file_content,
                    ACL="public-read",
                )

            except Exception as e:
                logger.error(f"Failed to upload file {key}: {e}")
                raise

            return f"{get_s3_public_url()}/{key}"

        tasks = [
            asyncio.create_task(_upload_file(basename, file_content))
            for basename, file_content in content_dict.items()
        ]

        for completed_task in asyncio.as_completed(tasks):
            url = await completed_task
            print("url", url)
            yield {"url": url, "is_temp": is_temp}

    def list_files(self) -> list[str]:
        """
        Lists all uploaded files.
        """

        response = self.client.list_objects_v2(
            Bucket=self.config.bucket_name,
            Prefix=self.config.folder,
        )

        def _make_url(obj: dict[str, Any]) -> str:
            return f"{self.config.endpoint_url}/{obj['Key']}"

        return [_make_url(url) for url in response.get("Contents", [])]

    def download_file(self, filename: str) -> Union[bytes, None]:
        key = f"{self.config.folder}/{filename}"

        bytesio = io.BytesIO()
        self.client.download_fileobj(self.config.bucket_name, key, bytesio)
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

import io
import os
from typing import Protocol, Union, Optional, runtime_checkable, TypedDict, AsyncGenerator
import logging
from boto3.session import Session
import blake3
from ..globals import FilesystemTypeEnum, RunCommandConfig
from ..config import get_config

logger = logging.getLogger(__name__)


class FileMetadata(TypedDict):
    url: str
    is_temp: bool


@runtime_checkable
class FileHandler(Protocol):
    async def upload_files(
        self, content: Union[list[bytes], bytes], file_type: str, metadata: Optional[bytes] = None
    ) -> AsyncGenerator[FileMetadata, None]:
        ...

    def list_files(self) -> list[str]:
        ...

    def download_file(self, filename: str) -> Union[bytes, None]:
        ...


class LocalFileHandler(FileHandler):
    def __init__(self, config: RunCommandConfig):
        if config.filesystem_type != FilesystemTypeEnum.LOCAL:
            raise ValueError("Invalid storage configuration")

        self.server_url = f"http://{config.host}:{config.port}"
        if config.assets_path is None:
            raise ValueError("Assets path is required")

        self.assets_path = config.assets_path

    def upload_files(self, content: Union[list[bytes], bytes], filename: Optional[str] = None) -> list[str]:
        if not os.path.exists(self.assets_path):
            os.makedirs(self.assets_path)

        filepath = os.path.join(self.assets_path, filename)
        with open(filepath, "wb") as f:
            f.write(content)

        return f"{self.server_url}/files/{filename}"

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
        if config.filesystem_type != FilesystemTypeEnum.S3:
            raise ValueError("Invalid storage configuration")

        self.config = config.s3

        session = Session()
        self.client = session.client(
            config.filesystem_type,
            region_name=config.s3.region_name,
            endpoint_url=config.s3.endpoint_url,
            aws_access_key_id=config.s3.access_key,
            aws_secret_access_key=config.s3.secret_key,
        )

    def upload_files(self, content: Union[list[bytes], bytes], filename: Optional[str] = None) -> list[str]:
        key = f"{self.config.folder}/{filename}" if self.config.folder else filename

        self.client.put_object(
            Bucket=self.config.bucket_name,
            Key=key,
            Body=content,
            ACL="public-read",
        )

        return f"{self.config.endpoint_url}/{key}"

    def list_files(self) -> list[str]:
        """
        Lists all uploaded files.
        """

        response = self.client.list_objects_v2(
            Bucket=self.config.bucket_name,
            Prefix=self.config.folder,
        )

        def _make_url(obj):
            return f"{self.config.endpoint_url}/{obj['Key']}"

        return [_make_url(url) for url in response.get("Contents", [])]

    def download_file(self, filename: str) -> Union[bytes, None]:
        key = f"{self.config.folder}/{filename}"

        bytesio = io.BytesIO()
        self.client.download_fileobj(self.config.bucket_name, key, bytesio)
        return bytesio.getvalue()


def get_file_handler() -> FileHandler:
    config = get_config()
    
    if config.filesystem_type == FilesystemTypeEnum.S3:
        return S3FileHandler(config)
    elif config.filesystem_type == FilesystemTypeEnum.LOCAL:
        return LocalFileHandler(config)
    else:
        raise ValueError("Invalid Filesystem-type")


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
    Calculates the BLAKE3 hash of the given file bytes and returns it as a string.
    """
    hasher = blake3.blake3()
    hasher.update(file_bytes)
    return hasher.hexdigest()

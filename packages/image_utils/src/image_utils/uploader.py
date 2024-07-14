import io
import os

from boto3.session import Session

import logging

from gen_server.globals import FilesystemTypeEnum, RunCommandConfig

logger = logging.getLogger(__name__)


def get_uploader(config):
    return (
        S3FileUploader(config)
        if config.filesystem_type == FilesystemTypeEnum.S3
        else LocalFileUploader(config)
    )


class LocalFileUploader:
    def __init__(self, config: RunCommandConfig):
        if config.filesystem_type != FilesystemTypeEnum.LOCAL:
            raise ValueError("Invalid storage configuration")

        self.server_url = f"http://{config.host}:{config.port}"
        self.assets_path = config.assets_path

    def upload_file(self, content: bytes | str, filename: str):
        if isinstance(content, str):
            content = content.encode('utf-8')

        if not os.path.exists(self.assets_path):
            os.makedirs(self.assets_path)

        filepath = os.path.join(self.assets_path, filename)
        with open(filepath, "wb") as f:
            f.write(content)

        return f"{self.server_url}/files/{filename}"

    def list_files(self):
        """
        Lists all uploaded files.
        """

        if not os.path.exists(self.assets_path):
            return []

        def _make_url(file):
            return f"{self.server_url}/files/{file}"

        return [_make_url(file) for file in os.listdir(self.assets_path)]

    def download_file(self, filename: str):
        full_filepath = os.path.join(self.assets_path, filename)

        if not os.path.exists(full_filepath):
            return None

        with open(full_filepath, mode="rb") as f:
            return f.read()


class S3FileUploader:
    def __init__(self, config):
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

    def upload_file(self, content: bytes | str, filename: str):
        key = f"{self.config.folder}/{filename}" if self.config.folder else filename

        self.client.put_object(
            Bucket=self.config.bucket_name,
            Key=key,
            Body=content,
            ACL="public-read",
        )

        return f"{self.config.endpoint_url}/{key}"

    def list_files(self):
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

    def download_file(
        self,
        filename: str,
    ):
        key = f"{self.config.folder}/{filename}"

        bytesio = io.BytesIO()
        self.client.download_fileobj(self.config.bucket_name, key, bytesio)
        return bytesio.getvalue()


def get_content_type(file_path):
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
        ".webp": "image/webp",
        ".ico": "image/x-icon",
    }
    return content_types.get(file_extension, "application/octet-stream")

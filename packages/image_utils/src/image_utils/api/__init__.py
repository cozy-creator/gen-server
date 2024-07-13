import io
import os
import traceback
from typing import List

import blake3
from aiohttp import web
from boto3.session import Session

from gen_server.config import get_config
import logging

from gen_server.globals import FilesystemTypeEnum, RunCommandConfig

logger = logging.getLogger(__name__)


class LocalFileHandler:
    def __init__(self, config: RunCommandConfig):
        if config.filesystem_type != FilesystemTypeEnum.LOCAL:
            raise ValueError("Invalid storage configuration")

        self.assets_path = config.assets_path

    def upload_file(self, content: bytes | str, filename: str):
        if not os.path.exists(self.assets_path):
            os.makedirs(self.assets_path)

        filepath = os.path.join(self.assets_path, filename)
        with open(filepath, "wb") as f:
            f.write(content)

        return f"http://127.0.0.1:8881/files/{filename}"

    def list_files(self):
        """
        Lists all files in the specified folder.
        """

        folder_path = self.assets_path
        if not os.path.exists(folder_path):
            return []

        def _make_url(file):
            return f"http://127.0.0.1:8881/files/{file}"

        return [_make_url(file) for file in os.listdir(folder_path)]

    def download_file(self, filename: str):
        full_filepath = os.path.join(self.assets_path, filename)

        if not os.path.exists(full_filepath):
            return None

        with open(full_filepath, mode="rb") as f:
            return f.read()


class S3FileHandler:
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
        Lists all files in the specified folder.
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


# Create an AioHttp handler class
class FileHandler:
    """
    AioHttp handler for uploading images.
    """

    def __init__(self):
        config = get_config()
        self.handler = (
            S3FileHandler(config)
            if config.filesystem_type == FilesystemTypeEnum.S3
            else LocalFileHandler(config)
        )

    async def handle_upload(self, request: web.Request) -> web.Response:
        """
        Handles image upload requests.
        """

        reader = await request.multipart()
        content = None
        filename = None
        try:
            while True:
                part = await reader.next()
                if not part:
                    break
                if part.name == "file":
                    content = await part.read()
                    filehash = blake3.blake3(content).hexdigest()
                    extension = os.path.splitext(part.filename)[1]
                    filename = f"{filehash}{extension}"

            url = self.handler.upload_file(content, filename)
            return web.json_response({"success": True, "url": url}, status=201)
        except Exception as e:
            traceback.print_exc()
            print(f"Error uploading file: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def list_files(self, _request: web.Request) -> web.Response:
        try:
            files = self.handler.list_files()
            return web.json_response({"success": True, "files": files}, status=200)
        except Exception as e:
            traceback.print_exc()
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def download_file(self, request: web.Request) -> web.Response:
        filename = request.match_info["filename"]
        try:
            body = self.handler.download_file(filename)
            if body is None:
                return web.json_response(
                    {"success": False, "error": "File not found"},
                    status=404,
                )

            headers = {
                "Content-Type": get_content_type(filename),
                "Content-Disposition": f'attachment; filename="{filename}"',
            }

            return web.Response(body=body, headers=headers, status=200)
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def get_file(self, request: web.Request) -> web.Response:
        """
        Serves a static file from the local filesystem.
        """

        filename = request.match_info.get("filename")
        if not isinstance(self.handler, LocalFileHandler):
            return web.json_response(
                {"success": False, "error": "Operation not supported"},
                status=400,
            )

        body = self.handler.download_file(filename)
        if body is None:
            return web.json_response(
                {"success": False, "error": "File not found"},
                status=404,
            )

        headers = {"Content-Type": get_content_type(filename)}
        return web.Response(body=body, headers=headers, status=200)


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


handler = FileHandler()

routes: List[web.RouteDef] = [
    web.post("/upload", handler.handle_upload),
    web.get("/files", handler.list_files),
    web.get("/files/{filename}", handler.get_file),
    web.get("/download/{filename}", handler.download_file),
]

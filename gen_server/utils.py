import mimetypes
import os
from pathlib import Path

from blake3 import blake3


def get_project_root() -> Path:
    return Path(__file__).parent.parent


async def get_file_blake3_hash(file_path: str):
    hasher = blake3()

    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(50 * 1024)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def get_file_mimetype(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)

    # If the MIME type cannot be determined, default to application/octet-stream
    if mime_type is None:
        mime_type = 'application/octet-stream'

    return mime_type


def get_file_extension(file_path):
    _, ext = os.path.splitext(file_path)
    return ext

import os.path
from typing import Union

from aiohttp import web, MultipartReader, BodyPartReader

from core_library.uploader import S3Uploader
from gen_server.settings import settings
from gen_server.utils import get_project_root, get_file_blake3_hash, get_file_mimetype, get_file_extension
import os
import json
from importlib import import_module


def discover_extensions():
    extensions_dir = "extensions"
    for item in os.listdir(extensions_dir):
        item_path = os.path.join(extensions_dir, item)
        if os.path.isdir(item_path):
            yield item_path


def register_extension_endpoints(app):
    for extension_dir in discover_extensions():
        manifest_path = os.path.join(extension_dir, "manifest.json")
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            api_module = import_module(f"{extension_dir}.api")
            for endpoint_info in manifest.get("api", []):
                app.router.add_route(endpoint_info["method"], endpoint_info["path"], getattr(api_module, endpoint_info["function"]))
        except (ImportError, FileNotFoundError):
            pass


app = web.Application()
register_extension_endpoints(app)


def setup_routes(app: web.Application):
    routes = web.RouteTableDef()

    @routes.post('/upload')
    async def upload(request: web.Request):
        reader = await request.multipart()

        field = await reader.next()
        assert field.name == 'file'

        filename = field.filename
        if not filename:
            return web.Response(status=400)

        file = await handle_file_upload(field)
        return web.json_response({"status": "success", "data": {"file": file}})

    @routes.get('/view')
    async def view(request: web.Request):
        file_name = request.query.get('file_name')

        if not file_name:
            return web.Response(status=400)

        root = get_project_root()
        file_path = os.path.abspath(os.path.normpath(root.joinpath("input", file_name)))
        return web.FileResponse(file_path)

    app.add_routes(routes)


async def handle_file_upload(file: Union[MultipartReader, BodyPartReader, None]):
    fpath, _ = await save_temp_file(file)

    if settings.is_production():
        upload_path = await upload_file_to_s3(fpath)
        os.unlink(fpath)

        return upload_path
    else:
        return await persist_temp_file(fpath)
    

async def persist_temp_file(file_path: str):
    root = get_project_root()
    file_key = await get_file_key(file_path)
    new_path = os.path.abspath(os.path.normpath(root.joinpath("input", file_key)))

    os.rename(file_path, new_path)
    return new_path


async def upload_file_to_s3(file_path: str):
    uploader = S3Uploader(settings.s3)
    key = await get_file_key(file_path)

    uploader.upload(file_path, key, get_file_mimetype(file_path))
    return uploader.get_uploaded_file_url(key)


async def get_file_key(file_path: str):
    file_hash = await get_file_blake3_hash(file_path)
    ext = get_file_extension(file_path)

    return f"{file_hash}{ext}"


async def save_temp_file(file: Union[MultipartReader, BodyPartReader, None]):
    root = get_project_root()
    file_path = os.path.abspath(os.path.normpath(root.joinpath("temp", file.filename)))

    size = 0
    with open(file_path, 'wb') as f:
        while True:
            chunk = await file.read_chunk(settings.read_chunk_size)
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)

    return file_path, size


def serve_static(app: web.Application):
    dist_path = get_project_root().joinpath('web/dist')
    if not os.path.exists(dist_path):
        print(f"Web dist directory not found: {dist_path}")
        return

    app.add_routes([
        web.static('/', get_project_root().joinpath('web/dist'), follow_symlinks=True)
    ])


def start_server(host: str, port: int):
    setup_routes(app)
    serve_static(app)
    web.run_app(app, host=host, port=port)

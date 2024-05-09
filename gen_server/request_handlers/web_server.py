import os.path
from typing import Union

from aiohttp import web, MultipartReader, BodyPartReader

from gen_server.settings import settings
from gen_server.uploader import get_client as get_uploader_client, create_multipart_upload, upload_chunk, \
    complete_multipart_upload, upload_raw_bytes, get_uploaded_file_url
from gen_server.utils import get_project_root, get_file_blake3_hash, get_file_mimetype, get_file_extension

app = web.Application()


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
    fpath, size = await save_temp_file(file)

    if settings.is_production():
        uploaded_path = await upload_file_to_s3(fpath, size)
        os.unlink(fpath)
        return uploaded_path
    else:
        return await persist_temp_file(fpath)


async def persist_temp_file(file_path: str):
    root = get_project_root()
    file_key = await get_file_key(file_path)
    new_path = os.path.abspath(os.path.normpath(root.joinpath("input", file_key)))

    os.rename(file_path, new_path)
    return new_path


async def upload_file_to_s3(file_path: str, file_size: int):
    client = get_uploader_client()

    mime_type = get_file_mimetype(file_path)
    key = await get_file_key(file_path)

    if file_size <= settings.upload_chunk_size:
        with open(file_path, 'rb') as file:
            _response = upload_raw_bytes(client=client, key=key, data=file.read(), mime_type=mime_type)

    else:
        upload = create_multipart_upload(client=client, key=key, mime_type=mime_type)

        parts, part_number = [], 1
        with open(file_path, 'rb') as file:
            while True:
                chunk = file.read(settings.upload_chunk_size)

                if not chunk:
                    break

                resp = upload_chunk(
                    chunk=chunk,
                    client=client,
                    key=key,
                    part_number=part_number,
                    upload_id=upload['UploadId']
                )

                if resp is not None:
                    parts.append({'PartNumber': part_number, 'ETag': resp['ETag']})

                part_number += 1

            _response = complete_multipart_upload(client=client, upload_id=upload['UploadId'], key=key, parts=parts)
    return get_uploaded_file_url(key)


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
    dist_path = get_project_root().joinpath('web/dist');
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

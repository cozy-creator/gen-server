import json
from typing import List, Tuple, Dict, Optional, Any
from aiohttp import web, BodyPartReader, MultipartReader
from aiohttp_middlewares import cors_middleware
import asyncio
import logging
import blake3

from ..config import get_config
from ..utils.file_handler import (
    LocalFileHandler,
    get_mime_type,
    get_file_handler,
)
from ..globals import CHECKPOINT_FILES, API_ENDPOINTS
from ..executor import generate_images, generate_images_from_repo
from typing import Iterable
import os
from ..utils.paths import get_web_root

routes = web.RouteTableDef()


@routes.get('/{filename:.*}')
async def serve_spa(request):
    filename = request.match_info['filename']
    if not filename:
        filename = 'index.html'
    
    file_path = os.path.join(get_web_root(), filename)
    
    if os.path.exists(file_path) and not os.path.isdir(file_path):
        return web.FileResponse(file_path)
    else:
        # Throw a 404 error if not found
        raise web.HTTPNotFound(text="File not found")


@routes.get("/checkpoints")
async def get_checkpoints(_req: web.Request) -> web.Response:
    serialized_checkpoints = {
        key: value.serialize() for key, value in CHECKPOINT_FILES.items()
    }

    print(f"here you go: {json.dumps(serialized_checkpoints)}")

    return web.Response(
        text=json.dumps(serialized_checkpoints), content_type="application/json"
    )


@routes.post("/generate")
async def handle_post(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200, reason="OK", headers={"Content-Type": "application/json"}
    )
    await response.prepare(request)

    try:
        # TO DO: validate these types using something like pydantic
        data = await request.json()
        models: Dict[str, int] = data["models"]
        positive_prompt: str = data["positive_prompt"]
        negative_prompt: str = data["negative_prompt"]
        random_seed: Optional[int] = data.get("random_seed", None)
        aspect_ratio: str = data["aspect_ratio"]

        async for urls in generate_images(
            models, positive_prompt, negative_prompt, random_seed, aspect_ratio
        ):
            json_response = json.dumps({"output": urls})
            await response.write((json_response).encode("utf-8") + b"\n")

    except Exception as e:
        # Handle the exception, you might want to log it or send an error response
        error_message = json.dumps({"error": str(e)})
        await response.write((error_message + "\n").encode("utf-8"))

    await response.write_eof()

    return response


# This does inference using an arbitrary hugging face repo
@routes.post("/generate-from-repo")
async def generate_from_repo(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200, reason="OK", headers={"Content-Type": "application/json"}
    )
    await response.prepare(request)

    # TO DO: validate these types using something like pydantic
    try:
        data = await request.json()
        repo_id: str = data["repo_id"]
        components: List[str] = data["components"]
        positive_prompt: str = data["positive_prompt"]
        negative_prompt: str = data["negative_prompt"]
        random_seed: Optional[int] = data.get("random_seed", None)
        aspect_ratio: Tuple[int, int] = data["aspect_ratio"]

        # Start streaming image URLs
        async for urls in generate_images_from_repo(
            repo_id,
            components,
            positive_prompt,
            negative_prompt,
            random_seed,
            aspect_ratio,
        ):
            await response.write(json.dumps({"output": urls}).encode("utf-8") + b"\n")

    except Exception as e:
        logging.error(f"Error during image generation: {str(e)}")
        await response.write(json.dumps({"error": str(e)}).encode("utf-8"))

    await response.write_eof()

    return response


@routes.post("/upload")
async def handle_upload(request: web.Request) -> web.Response:
    """
    Handles image upload requests.
    """

    reader = await request.multipart()
    content = None
    filename = None
    try:
        async def process_part(part):
            if part.name == "file":
                content = await part.read()
                filehash = blake3.blake3(content).hexdigest()
                if part.filename is None:
                    raise ValueError("File name is required")
                extension = os.path.splitext(part.filename)[1]
                return content, f"{filehash}{extension}"
            return None, None

        while True:
            part = await reader.next()
            if not part:
                break
            elif isinstance(part, BodyPartReader):
                content, filename = await process_part(part)
            elif isinstance(part, MultipartReader):
                async for subpart in part:
                    if isinstance(subpart, BodyPartReader):
                        content, filename = await process_part(subpart)
            else:
                raise ValueError(f"Unexpected part type: {type(part)}")

            if content and filename:
                break

        uploader = get_file_handler()
        url = uploader.upload_files(content, filename)
        return web.json_response({"success": True, "url": url}, status=201)
    except Exception as e:
        print(f"Error uploading file: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


@routes.get("/media")
async def list_files(_request: web.Request) -> web.Response:
    try:
        uploader = get_file_handler()
        files = uploader.list_files()
        return web.json_response({"success": True, "files": files}, status=200)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


@routes.get("/download/{filename}")
async def download_file(request: web.Request) -> web.Response:
    filename = request.match_info["filename"]
    try:
        uploader = get_file_handler()
        body = uploader.download_file(filename)
        if body is None:
            return web.json_response(
                {"success": False, "error": "File not found"},
                status=404,
            )

        headers = {
            "Content-Type": get_mime_type(filename),
            "Content-Disposition": f'attachment; filename="{filename}"',
        }

        return web.Response(body=body, headers=headers, status=200)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


@routes.get("/media/{filename}")
async def get_file(request: web.Request) -> web.Response:
    """
    Serves a static file from the local filesystem.
    """

    uploader = get_file_handler()
    filename = request.match_info.get("filename")
    
    if filename is None:
        return web.Response(body=b"", status=200)
    
    if not isinstance(uploader, LocalFileHandler):
        return web.json_response(
            {"success": False, "error": "Operation not supported"},
            status=400,
        )

    body = uploader.download_file(filename)
    if body is None:
        return web.json_response(
            {"success": False, "error": "File not found"},
            status=404,
        )

    headers = { "Content-Type": get_mime_type(filename) }
    return web.Response(body=body, headers=headers, status=200)


async def start_server(host: str = "localhost", port: int = 8881):
    """
    Starts the web server with API endpoints from extensions
    """
    app = web.Application(middlewares=[
        cors_middleware(allow_all=True)  # Enable CORS for all routes
    ])
    global routes

    # Make the entire /web/dist folder accessible at the root URL
    # routes.static("/", get_web_root())
    
    # Register built-in routes
    app.add_routes(routes)

    # Register all API endpoints added by extensions
    for name, routes in API_ENDPOINTS.items():
        # TO DO: consider adding prefixes to the routes based on extension-name?
        # How can we overwrite built-in routes
        app.router.add_routes(routes)

    runner = web.AppRunner(app)
    await runner.setup()

    # for route in app.router.routes():
    #     print(f"Route: {route}")

    # Try to bind to the desired port
    # try:
    site = web.TCPSite(runner, host, port)
    await site.start()
    # except socket.error:
    #     # If the desired port is in use, bind to a random available port
    #     site = web.TCPSite(runner, host, 0)
    #     await site.start()
    #     # _, port = site._server.sockets[0].getsockname()

    print(f"Server running on {site.name} (click to open)", flush=True)

    try:
        await asyncio.Future()  # Keep the server running
    except asyncio.CancelledError:
        print("Server is shutting down...")
        await runner.cleanup()


def api_routes_validator(plugin: Any) -> bool:
    if isinstance(plugin, web.RouteTableDef):
        return True

    if isinstance(plugin, Iterable):
        return all(isinstance(route, web.RouteDef) for route in plugin)

    return False


if __name__ == "__main__":
    asyncio.run(start_server())

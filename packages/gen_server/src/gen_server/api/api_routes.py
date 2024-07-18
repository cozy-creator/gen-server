import os
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from queue import Queue
from typing import List, Tuple, Dict, Optional, Any, Iterable
from aiohttp import web, BodyPartReader
from aiohttp_middlewares.cors import cors_middleware
import aiohttp_cors
import asyncio
import logging
import blake3

from ..utils.file_handler import (
    get_mime_type,
    get_file_handler,
)
from ..globals import RouteDefinition, CheckpointMetadata
# from ..executor import generate_images_from_repo
from ..utils.paths import get_web_root, get_assets_dir

# TO DO: get rid of this; use a more standard location
script_dir = os.path.dirname(os.path.abspath(__file__))
react_components = os.path.abspath(os.path.join(script_dir, "..", "react_components"))


# TO DO: eventually replace checkpoint_files with a database query instead
def create_aiohttp_app(
    job_queue: multiprocessing.Queue,
    checkpoint_files: Optional[dict[str, CheckpointMetadata]] = None,
    extra_routes: Optional[dict[str, RouteDefinition]] = None
) -> web.Application:
    """
    Starts the web server with API endpoints from extensions
    """
    app = web.Application(
        middlewares=[
            cors_middleware(allow_all=True)  # Enable CORS for all routes
        ]
    )

    routes = web.RouteTableDef()

    @routes.get("/checkpoints")
    async def get_checkpoints(_req: web.Request) -> web.Response:
        serialized_checkpoints = {
            key: value.serialize() for key, value in checkpoint_files.items()
        }

        print(f"here you go: {json.dumps(serialized_checkpoints)}")

        return web.Response(
            text=json.dumps(serialized_checkpoints), content_type="application/json"
        )


    @routes.post("/generate")
    async def handle_generate(request: web.Request) -> web.StreamResponse:
        """
        Submits generation requests from the user to the queue and streams the results.
        """
        response = web.StreamResponse(
            status=200, reason="OK", headers={"Content-Type": "application/json"}
        )
        await response.prepare(request)

        try:
            # TO DO: validate these types using something like pydantic
            data = await request.json()
            # models: Dict[str, int] = data["models"]
            # positive_prompt: str = data["positive_prompt"]
            # negative_prompt: str = data["negative_prompt"]
            # random_seed: Optional[int] = data.get("random_seed", None)
            # aspect_ratio: str = data["aspect_ratio"]

            with multiprocessing.Manager() as manager:
                response_queue = manager.Queue()
                # job_queue: multiprocessing.Queue = request.app['job_queue']
                job_queue.put((data, response_queue))

                while response_queue.empty():
                    await asyncio.sleep(0)

                while not response_queue.empty():
                    urls = response_queue.get()
                    await response.write(
                        json.dumps({"output": urls}).encode("utf-8") + b"\n"
                    )
            # async for urls in generate_images(
            #     models, positive_prompt, negative_prompt, random_seed, aspect_ratio
            # ):
            #     print(urls)
            #     json_response = json.dumps({"output": urls})
            #     await response.write((json_response).encode("utf-8") + b"\n")

        except Exception as e:
            # Handle the exception, you might want to log it or send an error response
            error_message = json.dumps({"error": str(e)})
            await response.write((error_message + "\n").encode("utf-8"))

        await response.write_eof()

        return response


    # This does inference using an arbitrary hugging face repo
    # @routes.post("/generate-from-repo")
    # async def generate_from_repo(request: web.Request) -> web.StreamResponse:
    #     response = web.StreamResponse(
    #         status=200, reason="OK", headers={"Content-Type": "application/json"}
    #     )
    #     await response.prepare(request)

    #     # TO DO: validate these types using something like pydantic
    #     try:
    #         data = await request.json()
    #         repo_id: str = data["repo_id"]
    #         components: List[str] = data["components"]
    #         positive_prompt: str = data["positive_prompt"]
    #         negative_prompt: str = data["negative_prompt"]
    #         random_seed: Optional[int] = data.get("random_seed", None)
    #         aspect_ratio: Tuple[int, int] = data["aspect_ratio"]

    #         # Start streaming image URLs
    #         async for urls in generate_images_from_repo(
    #             repo_id,
    #             components,
    #             positive_prompt,
    #             negative_prompt,
    #             random_seed,
    #             aspect_ratio,
    #         ):
    #             await response.write(json.dumps({"output": urls}).encode("utf-8") + b"\n")

    #     except Exception as e:
    #         logging.error(f"Error during image generation: {str(e)}")
    #         await response.write(json.dumps({"error": str(e)}).encode("utf-8"))

    #     await response.write_eof()

    #     return response


    @routes.post("/upload")
    async def handle_upload(request: web.Request) -> web.Response:
        """
        Receives files from users and saves them to the file system (S3 bucket or local).
        """

        reader = await request.multipart()
        content = None
        filename = None
        try:

            async def _process_part(
                part: BodyPartReader,
            ) -> Tuple[bytes | None, str | None]:
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
                    content, filename = await _process_part(part)
                else:
                    async for subpart in part:
                        content, filename = await _process_part(subpart)

                if content and filename:
                    break

            if content and filename:
                uploader = get_file_handler()
                url = uploader.upload_files([content], filename)
                return web.json_response({"success": True, "url": url}, status=201)
            else:
                return web.json_response(
                    {"success": False, "error": "No file uploaded"}, status=400
                )
        except Exception as e:
            print(f"Error uploading file: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)


    # TO DO: replace this with a database, so that the client can query for available assets
    @routes.get("/media")
    async def list_files(_request: web.Request) -> web.Response:
        """Lists files available in the the /assets directory"""
        try:
            uploader = get_file_handler()
            files = uploader.list_files()
            return web.json_response({"success": True, "files": files}, status=200)
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)


    @routes.get("/react-components")
    async def get_react_components(_req: web.Request) -> web.Response:
        """
        Returns additional react components and node definitions used by the custom-
        node factory on the client.
        """
        components = {}
        for filename in os.listdir(react_components):
            if filename.endswith(".js"):
                # Read the file
                with open(os.path.join(react_components, filename), "r") as f:
                    content = f.read()
                # Store the file name and content in the dictionary
                components[filename.replace(".js", "")] = content.strip()

        return web.Response(text=json.dumps(components), content_type="application/json")


    @routes.get("/{filename:.*}")
    async def serve_file(request: web.Request) -> web.Response:
        """
        Serves a static file from either the /assets folder or the /web/dist folder.
        This is a catch-all route, defined last so that all other routes can be matched
        first.
        """
        filename = request.match_info["filename"]
        if not filename:
            filename = "index.html"

        # serve from the assets folder
        if filename.startswith("media/"):
            # Serve from assets directory
            safe_filename = os.path.basename(filename[6:])  # Remove 'media/' prefix
            file_path = os.path.join(get_assets_dir(), safe_filename)
            print(f"Attempting to serve file from assets: {file_path}", flush=True)
        else:
            # Serve from web root
            file_path = os.path.join(get_web_root(), filename)
            print(f"Attempting to serve file from web root: {file_path}", flush=True)

        # serve from the static web-dist folder
        if not os.path.exists(file_path) or not os.path.isdir(file_path):
            if os.path.exists(file_path) and not os.path.isdir(file_path):
                with open(file_path, "rb") as file:
                    body = file.read()
                headers = {"Content-Type": get_mime_type(os.path.basename(file_path))}
                return web.Response(body=body, headers=headers, status=200)
            else:
                raise web.HTTPNotFound(text="File not found")

        # If we reach here, it means the path exists but is a directory
        raise web.HTTPNotFound(text="File not found")

    # Make the entire /web/dist folder accessible at the root URL
    # routes.static("/static", get_web_root())

    # Register built-in routes
    app.add_routes(routes)

    # Register all API endpoints added by extensions
    for _name, api_routes in extra_routes.items():
        # TO DO: consider adding prefixes to the routes based on extension-name?
        # How can we overwrite built-in routes
        app.router.add_routes(api_routes)

    # Store a reference to the queue in the aiohttp-application state
    # app['job_queue'] = queue
    
    return app

    # runner = web.AppRunner(app)
    # await runner.setup()

    # for route in app.router.routes():
    #     print(f"Route: {route}")

    # Try to bind to the desired port
    # try:
    # site = web.TCPSite(runner, host, port)
    # await site.start()
    # except socket.error:
    #     # If the desired port is in use, bind to a random available port
    #     site = web.TCPSite(runner, host, 0)
    #     await site.start()
    #     # _, port = site._server.sockets[0].getsockname()

    # print(f"Server running on {site.name} (click to open)", flush=True)

    # try:
    #     await asyncio.Future()  # Keep the server running
    # except asyncio.CancelledError:
    #     print("Server is shutting down...")
    #     await runner.cleanup()


# def run_server(host: str, port: int, queue: multiprocessing.Queue):
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(start_server(host, port, queue))


def api_routes_validator(plugin: Any) -> bool:
    if isinstance(plugin, web.RouteTableDef):
        return True

    if isinstance(plugin, Iterable):
        return all(isinstance(route, web.AbstractRouteDef) for route in plugin)

    return False

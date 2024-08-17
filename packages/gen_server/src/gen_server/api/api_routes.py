from functools import wraps
import os
import json
import multiprocessing
from threading import Event
import traceback
from typing import Callable, Optional, Tuple, Any, Iterable, Type
from uuid import uuid4
from aiohttp import web, BodyPartReader
from aiohttp_middlewares.cors import cors_middleware
import asyncio
import logging
import blake3
from huggingface_hub import HfApi
from pydantic import BaseModel

from ..base_types import ApiAuthenticator
from ..utils.file_handler import get_mime_type, get_file_handler
from ..globals import get_api_endpoints, get_checkpoint_files, get_hf_model_manager
from ..config import get_config, is_model_enabled
from ..base_types.authenticator import AuthenticationError

# from ..executor import generate_images_from_repo
from ..utils.paths import get_web_root, get_assets_dir

# TO DO: get rid of this; use a more standard location
script_dir = os.path.dirname(os.path.abspath(__file__))
react_components = os.path.abspath(os.path.join(script_dir, "..", "react_components"))


# Huggingface API
hf_api = HfApi()


class GenerateData(BaseModel):
    random_seed: int
    aspect_ratio: str
    positive_prompt: str
    negative_prompt: str
    models: dict[str, int]
    webhook_url: Optional[str] = None


# TO DO: eventually replace checkpoint_files with a database query instead
def create_aiohttp_app(
    job_queue: multiprocessing.Queue,
    job_registry: dict[str, Event],
    node_defs: dict[str, Any],
    api_authenticator: Optional[Type[ApiAuthenticator]] = None,
) -> web.Application:
    """
    Starts the web server with API endpoints from extensions
    """

    logging.basicConfig(
        level=logging.INFO,  # Set the minimum level to INFO
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    app = web.Application(
        middlewares=[
            cors_middleware(allow_all=True)  # Enable CORS for all routes
        ]
    )

    config = get_config()
    routes = web.RouteTableDef()
    authenticator = api_authenticator() if api_authenticator else None
    hf_model_manager = get_hf_model_manager()

    def auth_middleware(request: web.Request, handler: Callable):
        if authenticator:
            try:
                authenticator.authenticate(request)
            except AuthenticationError as e:
                logger.error("An error occured during authentication", e)
                return web.json_response(
                    {"status": "error", "message": e.message}, status=401
                )
            except Exception as e:
                logger.error("An unknown error occured", e)
                return web.json_response(
                    {
                        "status": "error",
                        "message": "An error occured, please try again",
                    },
                    status=500,
                )

        return handler(request)

    def validate_models(handler: Callable):
        @wraps(handler)
        async def wrapped(request: web.Request):
            data = await request.json()
            if "models" not in data:
                return web.json_response(
                    {"error": "models field is required"}, status=400
                )

            for model_id in data["models"].keys():
                if config.environment == "production" and not is_model_enabled(
                    model_id
                ):
                    return web.json_response(
                        {"error": f"Model {model_id} is not enabled"},
                        status=400,
                    )
                if config.environment != "production":
                    is_downloaded, variant = hf_model_manager.is_downloaded(model_id)
                    if not is_downloaded:
                        return web.json_response(
                            {"error": f"Model {model_id} is not available"},
                            status=404,
                        )
            return await handler(request)

        return wrapped

    def with_auth(handler: Callable) -> Callable:
        @wraps(handler)
        def wrapped(request: web.Request):
            return auth_middleware(request, handler)

        return wrapped

    @routes.get("/checkpoints")
    async def get_checkpoints(_req: web.Request) -> web.Response:
        checkpoint_files = get_checkpoint_files()
        serialized_checkpoints = {
            key: value.serialize() for key, value in checkpoint_files.items()
        }

        print(f"here you go: {json.dumps(serialized_checkpoints)}")

        return web.Response(
            text=json.dumps(serialized_checkpoints), content_type="application/json"
        )

    @routes.get("/node-defs")
    async def get_node_defs(_req: web.Request) -> web.Response:
        return web.Response(text=json.dumps(node_defs), content_type="application/json")

    @routes.post("/generate")
    @with_auth
    @validate_models
    async def handle_generate(request: web.Request) -> web.StreamResponse:
        """
        Submits generation requests from the user to the queue and streams the results.
        """

        data = GenerateData(**(await request.json()))

        if data.webhook_url is not None:
            return web.json_response(
                {"error": "webhook_url is not allowed for this endpoint"},
                status=400,
            )

        response = web.StreamResponse(
            status=200, reason="OK", headers={"Content-Type": "application/json"}
        )
        await response.prepare(request)

        try:
            # Create a pair of connections for inter-process communication
            parent_conn, child_conn = multiprocessing.Pipe()

            # Submit the job to the queue
            job_queue.put((data.dict(), child_conn, None))

            print(f"job queue put {data}", flush=True)

            # Set a timeout for the entire operation
            total_timeout = 300  # 5 minutes, adjust as needed
            start_time = asyncio.get_event_loop().time()

            while True:
                try:
                    # Use asyncio to make the blocking call non-blocking
                    # Check if data is available
                    has_data = await asyncio.get_event_loop().run_in_executor(
                        None,
                        parent_conn.poll,
                        1,  # 1 second timeout
                    )

                    if has_data:
                        # Data is available, so let's receive it
                        file_url = parent_conn.recv()

                        if file_url is None:  # Signal for completion
                            break
                        else:  # We have a valid file URL
                            await response.write(
                                json.dumps({"output": file_url}).encode("utf-8") + b"\n"
                            )
                            await (
                                response.drain()
                            )  # Ensure the data is sent immediately

                    # Check if we've exceeded the total timeout
                    if asyncio.get_event_loop().time() - start_time > total_timeout:
                        raise TimeoutError("Operation timed out")

                except EOFError:
                    # Connection was closed
                    break

            await response.write(
                json.dumps({"status": "finished"}).encode("utf-8") + b"\n"
            )

        except TimeoutError as e:
            logger.error(f"Generation timed out: {str(e)}")
            await response.write(
                json.dumps({"error": "Operation timed out"}).encode("utf-8") + b"\n"
            )
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            await response.write(json.dumps({"error": str(e)}).encode("utf-8") + b"\n")
        finally:
            if "parent_conn" in locals():
                parent_conn.close()

        await response.write_eof()
        return response

    @routes.get("/get_diffusers_models")
    async def get_diffusers_models(request: web.Request) -> web.StreamResponse:
        """
        Retrieves diffusers models and streams the results.
        """
        response = web.StreamResponse(
            status=200, reason="OK", headers={"Content-Type": "application/json"}
        )
        await response.prepare(request)

        try:
            page = int(request.query.get("page", 1))
            limit = int(request.query.get("limit", 10))
            offset = (page - 1) * limit

            # Set a timeout for the entire operation
            total_timeout = 60  # 1 minute, adjust as needed
            start_time = asyncio.get_event_loop().time()

            # Use asyncio to make the potentially blocking API calls non-blocking
            models = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: hf_api.list_models(
                    library="diffusers",
                    sort="downloads",
                    task=["text-to-image"],
                    limit=limit,
                ),
            )

            # Stream the models as they are retrieved
            for model in models:
                model_data = {
                    "id": model.id,
                    "downloads": model.downloads,
                    "tags": model.tags,
                }
                await response.write(
                    json.dumps({"model": model_data}).encode("utf-8") + b"\n"
                )
                await response.drain()

                # Check if we've exceeded the total timeout
                if asyncio.get_event_loop().time() - start_time > total_timeout:
                    raise TimeoutError("Operation timed out")

            # Get total number of models (this might be a heavy operation, maybe cache the value?)
            total_models = await asyncio.get_event_loop().run_in_executor(
                None, lambda: len(hf_api.list_models(library="diffusers"))
            )

            # Send pagination information
            pagination_data = {
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total_models,
                    "total_pages": (total_models // limit)
                    + (1 if total_models % limit > 0 else 0),
                }
            }
            await response.write(json.dumps(pagination_data).encode("utf-8") + b"\n")

            # Signal completion
            await response.write(
                json.dumps({"status": "finished"}).encode("utf-8") + b"\n"
            )

        except TimeoutError as e:
            logging.error(f"Operation timed out: {str(e)}")
            await response.write(
                json.dumps({"error": "Operation timed out"}).encode("utf-8") + b"\n"
            )
        except Exception as e:
            logging.error(f"Error in get_diffusers_models: {str(e)}")
            await response.write(json.dumps({"error": str(e)}).encode("utf-8") + b"\n")

        await response.write_eof()
        return response

    @routes.post("/generate_async")
    @with_auth
    @validate_models
    async def handle_generate_async(request: web.Request) -> web.Response:
        """
        Submits generation requests from the user to the queue and returns an id.
        """

        job_id = str(uuid4())
        try:
            data = GenerateData(**(await request.json()))
            if data.webhook_url is None:
                return web.json_response(
                    {"error": "webhook_url is required"}, status=400
                )

            # Submit the job to the queue
            job_queue.put((data.dict(), None, job_id))
            print(f"job queue put {data}", flush=True)

            return web.json_response(
                {"status": "pending", "job_id": job_id},
                status=201,
            )
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            return web.json_response({"error": str(e)})

    @routes.get("/cancel/{job_id}")
    async def handle_cancel(request: web.Request) -> web.Response:
        """
        Cancels a pending generation job.
        """

        job_id = request.match_info["job_id"]
        try:
            if job_id not in job_registry:
                return web.json_response(
                    {"error": f"Job ID {job_id} not found"}, status=404
                )

            cancel_event = job_registry[job_id]
            print(f"Cancelling job {job_id}", flush=True)
            print(cancel_event)
            print(f"Cancelled event?: {cancel_event.is_set()}")
            cancel_event.set()
            print(f"Cancelled event?: {cancel_event.is_set()}")

            return web.json_response({"status": "cancelled"}, status=200)
        except Exception as e:
            logger.error(f"Error cancelling job: {str(e)}")
            return web.json_response({"error": str(e)})

    @routes.post("/generate_async")
    @with_auth
    async def handle_generate_async(request: web.Request) -> web.Response:
        """
        Submits generation requests from the user to the queue and returns an id.
        """

        job_id = str(uuid4())
        try:
            data = GenerateData(**(await request.json()))
            if data.webhook_url is None:
                return web.json_response(
                    {"error": "webhook_url is required"}, status=400
                )

            # Submit the job to the queue
            job_queue.put((data.dict(), None, job_id))
            print(f"job queue put {data}", flush=True)

            return web.json_response(
                {"status": "pending", "job_id": job_id},
                status=201,
            )
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in generation: {str(e)}")
            return web.json_response({"error": str(e)})

    @routes.get("/cancel/{job_id}")
    async def handle_cancel(request: web.Request) -> web.Response:
        """
        Cancels a pending generation job.
        """

        job_id = request.match_info["job_id"]
        try:
            if job_id not in job_registry:
                return web.json_response(
                    {"error": f"Job ID {job_id} not found"}, status=404
                )

            cancel_event = job_registry[job_id]
            print(f"Cancelling job {job_id}", flush=True)
            print(cancel_event)
            print(f"Cancelled event?: {cancel_event.is_set()}")
            cancel_event.set()
            print(f"Cancelled event?: {cancel_event.is_set()}")

            return web.json_response({"status": "cancelled"}, status=200)
        except Exception as e:
            logger.error(f"Error cancelling job: {str(e)}")
            return web.json_response({"error": str(e)})

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

        return web.Response(
            text=json.dumps(components), content_type="application/json"
        )

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
    api_routes = get_api_endpoints()
    for _name, api_routes in api_routes.items():
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

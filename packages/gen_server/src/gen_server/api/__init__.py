import json
from typing import List, Tuple, Dict, Optional, AsyncGenerator, Any
from aiohttp import web
import asyncio
import logging
from ..globals import CHECKPOINT_FILES, API_ENDPOINTS, RouteDefinition
from ..executor import generate_images, generate_images_from_repo
from typing import Iterable
import os

routes = web.RouteTableDef()

script_path = '../../../../web/buildPlugins.sh'

script_dir = os.path.dirname(os.path.abspath(__file__))
web_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..', '..', 'web', 'dist'))


@routes.get("/checkpoints")
async def get_checkpoints(_req: web.Request) -> web.Response:
    serialized_checkpoints = {
        key: value.serialize() for key, value in CHECKPOINT_FILES.items()
    }

    return web.Response(
        text=json.dumps(serialized_checkpoints), content_type="application/json"
    )


@routes.post("/generate")
async def handle_post(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={'Content-Type': 'application/json'}
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
        
        async for urls in generate_images(models, positive_prompt, negative_prompt, random_seed, aspect_ratio):
            json_response = json.dumps({ "output": urls })
            await response.write((json_response).encode('utf-8') + b"\n")
            
    except Exception as e:
        # Handle the exception, you might want to log it or send an error response
        error_message = json.dumps({"error": str(e)})
        await response.write((error_message + "\n").encode('utf-8'))

    await response.write_eof()
    
    return response

@routes.post("/generate-from-repo")
async def generate_from_repo(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(status=200, reason='OK', headers={'Content-Type': 'application/json'})
    await response.prepare(request)
    
    # TO DO: validate these types using something like pydantic
    try:
        data = await request.json()
        repo_id: str = data['repo_id']
        components: List[str] = data['components']
        positive_prompt: str = data['positive_prompt']
        negative_prompt: str = data['negative_prompt']
        random_seed: Optional[int] = data.get('random_seed', None)
        aspect_ratio: Tuple[int, int] = data['aspect_ratio']

        # Start streaming image URLs
        async for urls in generate_images_from_repo(repo_id, components, positive_prompt, negative_prompt, random_seed, aspect_ratio):
            await response.write(json.dumps({"output": urls}).encode('utf-8') + b'\n')

    except Exception as e:
        logging.error(f"Error during image generation: {str(e)}")
        await response.write(json.dumps({"error": str(e)}).encode('utf-8'))

    await response.write_eof()
    
    return response


async def run_shell_script():
    process = await asyncio.create_subprocess_shell(
        script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        print(f'Error in script {script_path}:')
        print(f'Stdout: {stdout.decode()}')
        print(f'Stderr: {stderr.decode()}')
        raise RuntimeError('Shell script failed')

    print(f'Script {script_path} completed successfully')


@routes.get("/")
async def home(request: web.Request):
    return web.FileResponse(os.path.join(web_root, "index.html"))


# @routes.post("/generate")
# async def handle_post(request: web.Request) -> web.StreamResponse:
#     response = web.StreamResponse(
#         status=200, reason="OK", headers={"Content-Type": "application/json"}
#     )
#     await response.prepare(request)

#     # TO DO: validate these types using something like pydantic
#     try:
#         data = await request.json()
#         models: Dict[str, int] = data["models"]
#         positive_prompt: str = data["positive_prompt"]
#         negative_prompt: str = data["negative_prompt"]
#         random_seed: Optional[int] = data.get("random_seed", None)
#         aspect_ratio: str = data["aspect_ratio"]
        
#         print('request received')

#         # Start streaming image URLs
#         async for urls in generate_images(
#             models,
#             positive_prompt,
#             negative_prompt,
#             random_seed,
#             aspect_ratio,
#         ):
#             print(f"urls received!: {urls}")
#             json_response = json.dumps({ "output": urls })
#             await response.write(json_response.encode('utf-8') + b"\n")
#             # await response.write(json.dumps({"output": urls}).encode("utf-8") + b"\n")

#     except Exception as e:
#         logging.error(f"Error during image generation: {str(e)}")
#         await response.write(json.dumps({"error": str(e)}).encode("utf-8"))

#     await response.write_eof()
    
#     return response


async def start_server(host: str = 'localhost', port: int = 8188):
    """
    Starts the web server with API endpoints from extensions
    """
    app = web.Application()
    global routes
    app.add_routes(routes)

    # run shell script
    await run_shell_script()

    app.router.add_static('/', web_root)

    # Register all API endpoints from extensions
    # Iterate over API_ENDPOINTS and add routes
    for name, routes in API_ENDPOINTS.items():
        # TO DO: consider adding prefixes to the routes based on extension-name?
        app.router.add_routes(routes)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"Server running on {site.name} (click to open)")
    
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
    # web.run_app(app, port=8080)

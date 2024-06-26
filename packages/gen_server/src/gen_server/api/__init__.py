

import json
from typing import List, Tuple, Dict, Optional
from aiohttp import web
import asyncio
import logging
from ..globals import CHECKPOINT_FILES, API_ENDPOINTS
from ..executor import generate_images, generate_images_from_repo
from typing import Iterable
routes = web.RouteTableDef()


@routes.get("/checkpoints")
async def get_checkpoints(_req: web.Request) -> web.Response:
    serialized_checkpoints = { 
        key: value.serialize() for key, value in CHECKPOINT_FILES.items()
    }
    
    return web.Response(
        text=json.dumps(serialized_checkpoints), 
        content_type='application/json'
    )

@routes.post("/generate")
async def handle_post(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(status=200, reason='OK', headers={'Content-Type': 'application/json'})
    await response.prepare(request)
    
    # TO DO: validate these types using something like pydantic
    try:
        data = await request.json()
        models: Dict[str, int] = data['models']
        positive_prompt: str = data['positive_prompt']
        negative_prompt: str = data['negative_prompt']
        random_seed: Optional[int] = data.get('random_seed', None)
        aspect_ratio: Tuple[int, int] = data['aspect_ratio']

        # Start streaming image URLs
        async for urls in generate_images(models, positive_prompt, negative_prompt, random_seed, aspect_ratio):
            await response.write(json.dumps({"output": urls}).encode('utf-8') + b'\n')

    except Exception as e:
        logging.error(f"Error during image generation: {str(e)}")
        await response.write(json.dumps({"error": str(e)}).encode('utf-8'))

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


async def start_server():
    """
    Starts the web server with API endpoints from extensions
    """
    app = web.Application()
    global routes
    app.add_routes(routes)

    # Register all API endpoints from extensions
    # Iterate over API_ENDPOINTS and add routes
    for name, routes in API_ENDPOINTS.items():
        # TO DO: consider adding prefixes to the routes based on extension-name?
        app.router.add_routes(routes)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8080)  # Host and port
    await site.start()
    print(f"Server running on http://localhost:8080/")
    await asyncio.Future()  # Keep the server running

if __name__ == '__main__':
    asyncio.run(start_server())
    # web.run_app(app, port=8080)

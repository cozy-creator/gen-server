

import json
from typing import List, Tuple, Dict, Optional
from aiohttp import web
from ..globals import CHECKPOINT_FILES
from ..executor import generate_images
routes = web.RouteTableDef()


@routes.get("/checkpoints")
async def get_checkpoints(request):
    serialized_checkpoints = { 
        key: value.serialize() for key, value in CHECKPOINT_FILES.items()
    }
    
    return web.Response(
        text=json.dumps(serialized_checkpoints), 
        content_type='application/json'
    )

@routes.post("/generate")
async def handle_post(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        models: Dict[str, int] = data['models']  # Tuple of model-names and number of images per model
        positive_prompt: str = data['positive_prompt']
        negative_prompt: str = data['negative_prompt']
        random_seed: Optional[int] = data.get('random_seed', None)
        aspect_ratio: Tuple[int, int] = data['aspect_ratio'] # tuple (width, height)

        # Placeholder for image generation logic
        urls: List[str] = await generate_images(models, positive_prompt, negative_prompt, random_seed, aspect_ratio)

        return web.Response(text=json.dumps({"urls": urls}), content_type='application/json')
    except Exception as e:
        return web.Response(text=str(e), status=400)


app = web.Application()
app.add_routes(routes)

if __name__ == '__main__':
    web.run_app(app, port=8080)



import json
from typing import List, Tuple, Dict, Optional
from aiohttp import web
routes = web.RouteTableDef()


# TO DO: add endpoint for getting the model-registry
@routes.get("/model-registry")
async def get_model_registry(request):
    return web.Response(text=json.dumps({"models": {}}))

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

# This is a fixed prebuilt workflow; it's a placeholder for now
async def generate_images(models: Dict[str, int], positive_prompt: str, negative_prompt: str, random_seed: Optional[int], aspect_ratio: Tuple[int, int]) -> List[str]:
    # Simulate image generation and return URLs
    urls = []
    for model_name, num_images in models.items():
        for _ in range(num_images):
            urls.append(f"http://{model_name}.example.com/image.png")
    return urls


app = web.Application()
app.add_routes(routes)

if __name__ == '__main__':
    web.run_app(app, port=8080)

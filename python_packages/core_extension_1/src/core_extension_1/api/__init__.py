from huggingface_hub import model_info, repo_exists, hf_hub_download, HfApi
import logging
from aiohttp import web
import json
from typing import List


# Configure the logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

hf_api = HfApi()

async def get_components(request: web.Request):
    try:
        data = await request.json()
        repo_id = data.get('repo_id')
        if repo_exists(repo_id=repo_id):
            model = model_info(repo_id)
            file_to_check = "model_index.json"
            is_present = any(sibling.rfilename == file_to_check for sibling in model.siblings)
            if is_present:
                path = hf_hub_download(repo_id, "model_index.json")
                with open(path, 'r') as file:
                    data = json.load(file)
                keys = {key: value for key, value in data.items()}
                return web.json_response({"keys": keys})
            else:
                logging.error(f"The repository {repo_id} does not contain 'model_index.json' or is not im the valid diffusers repo format.")
                return web.json_response({'error': 'Invalid diffusers repo format'}, status=400)
        else:
            return web.json_response({'error': 'Repository not found'}, status=404)
    except Exception as e:
        logging.error(f"Error in get_components: {e}")
        return web.json_response({'error': str(e)}, status=500)
    

async def get_diffusers_models(request: web.Request):
    try:
        page = int(request.query.get('page', 1))
        limit = int(request.query.get('limit', 10))
        offset = (page - 1) * limit

        models = hf_api.list_models(library="diffusers", sort="trending", limit=limit, offset=offset)
        total_models = len(hf_api.list_models(library="diffusers"))

        response_data = {
            "models": models,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total_models,
                "total_pages": (total_models // limit) + (1 if total_models % limit > 0 else 0)
            }
        }
        return web.json_response(response_data)
    except Exception as e:
        logging.error(f"Error in get_diffusers_models: {e}")
        return web.json_response({'error': str(e)}, status=500)
    

routes: List[web.RouteDef] = [
    web.post('/get_components', get_components),
]

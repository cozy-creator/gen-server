import time
from gen_server.globals import get_custom_nodes
import torch
import logging
from typing import Optional, Tuple, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TO DO: remove this
async def generate_images_from_repo(
    repo_id: str,
    components: List[str],
    positive_prompt: str,
    negative_prompt: str,
    random_seed: Optional[int],
    aspect_ratio: Tuple[int, int],
):
    start = time.time()
    custom_nodes = get_custom_nodes()

    LoadComponents = custom_nodes["core_extension_1.load_components"]
    load_components = LoadComponents()

    components = load_components(repo_id, components)

    # runwayml/stable-diffusion-v1-5

    CreatePipe = custom_nodes["core_extension_1.create_pipe"]
    create_pipe = CreatePipe()

    pipe = create_pipe(loaded_components=components)

    run_pipe = custom_nodes["core_extension_1.run_pipe"]()

    pil_images = run_pipe(
        pipe,
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        width=aspect_ratio[0],
        height=aspect_ratio[1],
        num_images=1,
        generator=torch.Generator().manual_seed(random_seed)
        if random_seed is not None
        else None,
    )

    SaveNode = custom_nodes["image_utils.save_file"]
    save_node = SaveNode()
    urls: List[dict[str, Any]] = save_node(images=pil_images, temp=False)

    yield urls

    print(f"Image generated in {time.time() - start} seconds")

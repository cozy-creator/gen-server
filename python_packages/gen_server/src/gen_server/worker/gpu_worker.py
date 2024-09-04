import time
import logging
import traceback
from typing import Any, AsyncGenerator, TypeVar

import torch
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore


from ..globals import get_custom_nodes


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)


async def generate_images_non_io(
    task_data: dict[str, Any],
) -> AsyncGenerator[torch.Tensor, None]:
    """Generates images based on the provided task data."""
    start = time.time()

    custom_nodes = get_custom_nodes()

    try:
        models = task_data.get("models", {})
        positive_prompt = task_data.get("positive_prompt")
        negative_prompt = task_data.get("negative_prompt", "")
        random_seed = task_data.get("random_seed")
        aspect_ratio: str = task_data.get("aspect_ratio", "1/1")

        # Get the ImageGenNode
        image_gen_node = custom_nodes["core_extension_1.image_gen_node"]()

        for checkpoint_id, num_images in models.items():
            try:
                # Run the ImageGenNode
                result: torch.Tensor = await image_gen_node(
                    model_id=checkpoint_id,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    num_images=num_images,
                    random_seed=random_seed,
                )

                images = result["images"]

                tensor_images = images.to("cpu")
                # TO DO: could this problematic if the gpu-worker terminates as this tensor is
                # still in use?
                tensor_images.share_memory_()
                logger.info("Placed generation result on queue")
                logger.info(f"Tensor dimensions: {tensor_images.shape}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                yield tensor_images
            except Exception as e:
                traceback.print_exc()
                logger.error(
                    f"Error generating images for model '{checkpoint_id}': {e}"
                )
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in image generation workflow: {e}")

    logger.info(f"Image generated in {time.time() - start} seconds")

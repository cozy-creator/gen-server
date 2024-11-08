import time
import logging
import traceback
from typing import Any, AsyncGenerator, Tuple, TypeVar

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
) -> AsyncGenerator[Tuple[str, torch.Tensor], None]:
    """Generates images based on the provided task data."""
    logger.info(f"Generating images with task data: {task_data}")
    start = time.time()

    custom_nodes = get_custom_nodes()

    try:
        model_id = task_data.get("model")
        positive_prompt = task_data.get("positive_prompt")
        negative_prompt = task_data.get("negative_prompt", "")
        random_seed = task_data.get("random_seed")
        aspect_ratio: str = task_data.get("aspect_ratio", "1/1")
        num_outputs = task_data.get("num_outputs", 1)

        # Determine which node to use based on presence of source_image
        source_image = task_data.get("source_image")
        strength = task_data.get("strength", 0.8)

        if source_image:
            # Use image-to-image node
            image_gen_node = custom_nodes["core_extension_1.image_to_image_node"]()
        else:
            # Use regular image generation node
            image_gen_node = custom_nodes["core_extension_1.image_gen_node"]()

        try:
            # Run the ImageGenNode
            params = {
                "model_id": model_id,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "num_images": num_outputs,
                "random_seed": random_seed,
            }

            if source_image:
                # Add image-to-image specific parameters
                params["source_image"] = source_image
                params["strength"] = strength
            else:
                # Add regular generation specific parameters
                params["aspect_ratio"] = aspect_ratio

            # Run the appropriate node
            result = await image_gen_node(**params)

            images = result["images"]

            tensor_images = images.to("cpu")
            # TO DO: could this problematic if the gpu-worker terminates as this tensor is
            # still in use?
            tensor_images.share_memory_()
            logger.info("Placed generation result on queue")
            logger.info(f"Tensor dimensions: {tensor_images.shape}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            yield (model_id, tensor_images)
        except Exception as e:
            traceback.print_exc()
            logger.error(
                f"Error generating images for model '{model_id}': {e}"
            )
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in image generation workflow: {e}")

    logger.info(f"Image generated in {time.time() - start} seconds")

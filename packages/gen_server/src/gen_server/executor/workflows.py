import time
import traceback
from gen_server.utils.device import get_torch_device

import torch
import logging
from typing import Any, Optional
from multiprocessing.connection import Connection

from queue import Queue

from ..globals import (
    get_architectures,
    get_checkpoint_files,
    get_custom_nodes,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Simulating the executor code ===


# This is a fixed prebuilt workflow; it's a placeholder for now
def generate_images(
    task_data: dict[str, Any],
    tensor_queue: Queue,
    response_conn: Connection,
    start_time: float,
) -> None:
    """Generates images based on the provided task data."""
    start = time.time()

    custom_nodes = get_custom_nodes()
    architectures = get_architectures()
    checkpoint_files = get_checkpoint_files()

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
                tensor_images: torch.Tensor = image_gen_node(
                    checkpoint_id=checkpoint_id,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    num_images=num_images,
                    random_seed=random_seed,
                    checkpoint_files=checkpoint_files,
                    architectures=architectures,
                    device=get_torch_device(),
                )["images"]

                # Process the generated images (convert to tensors and put on the queue)
                # tensor_batch = []
                # for image in generated_images:
                #     tensor_image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0
                #     tensor_batch.append(tensor_image)

                # Stack individual tensors into a single tensor
                # tensor_images = torch.stack(tensor_batch)

                tensor_images = tensor_images.to("cpu")
                # TO DO: could this problematic if the gpu-worker terminates as this tensor is
                # still in use?
                tensor_images.share_memory_()

                tensor_queue.put((tensor_images, response_conn, start_time))

                print("Placed generation result on queue")
                print(f"Tensor dimensions: {tensor_images.shape}")
                print(f"Response connection: {response_conn}")

                # Free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                # Log the error and send an error message to the API server
                traceback.print_exc()
                logger.error(
                    f"Error generating images for model '{checkpoint_id}': {e}"
                )

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in image generation workflow: {e}")
        # tensor_queue.put((None, None)) # Tell the io-worker that we're done

    print(f"Image generated in {time.time() - start} seconds")

    # Signal end of generation to IO process
    # tensor_queue.put((None, None))


async def generate_images_non_io(
    task_data: dict[str, Any],
) -> Optional[torch.Tensor]:
    """Generates images based on the provided task data."""
    start = time.time()

    custom_nodes = get_custom_nodes()
    architectures = get_architectures()
    checkpoint_files = get_checkpoint_files()

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
                tensor_images: torch.Tensor = image_gen_node(
                    checkpoint_id=checkpoint_id,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    num_images=num_images,
                    random_seed=random_seed,
                    checkpoint_files=checkpoint_files,
                    architectures=architectures,
                    device=get_torch_device(),
                )["images"]

                # Process the generated images (convert to tensors and put on the queue)
                # tensor_batch = []
                # for image in generated_images:
                #     tensor_image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0
                #     tensor_batch.append(tensor_image)

                # Stack individual tensors into a single tensor
                # tensor_images = torch.stack(tensor_batch)

                tensor_images = tensor_images.to("cpu")
                # TO DO: could this problematic if the gpu-worker terminates as this tensor is
                # still in use?
                tensor_images.share_memory_()
                print("Placed generation result on queue")
                print(f"Tensor dimensions: {tensor_images.shape}")

                # Free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return tensor_images

            except Exception as e:
                # Log the error and send an error message to the API server
                traceback.print_exc()
                logger.error(
                    f"Error generating images for model '{checkpoint_id}': {e}"
                )

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in image generation workflow: {e}")

    print(f"Image generated in {time.time() - start} seconds")

    # Signal end of generation to IO process
    # tensor_queue.put((None, None))

import asyncio
from threading import Event
import time
import traceback


import torch
import logging
from typing import Any, Dict, Generator, Optional
from multiprocessing.connection import Connection
from diffusers.callbacks import PipelineCallback
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

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
    _architectures = get_architectures()
    _checkpoint_files = get_checkpoint_files()

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
                    repo_id=checkpoint_id,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    num_images=num_images,
                    random_seed=random_seed,
                    # checkpoint_files=checkpoint_files,
                    # architectures=architectures,
                    # device=get_torch_device(),
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


class CancelCallback(PipelineCallback):
    tensor_inputs = []  # type: ignore

    def __init__(
        self,
        cancel_event: Optional[Event] = None,
        cutoff_step_ratio: float = 1.0,
        cutoff_step_index: Optional[int] = None,
    ):
        super().__init__(cutoff_step_ratio, cutoff_step_index)
        self._cancel_event = cancel_event

    def callback_fn(
        self,
        pipeline: DiffusionPipeline,
        step_index: int,
        timesteps: int,
        callback_kwargs: Dict,
    ) -> Dict[str, Any]:
        if self._cancel_event and self._cancel_event.is_set():
            raise StopIteration("Inference was cancelled.")
        return callback_kwargs


def generate_images_non_io(
    task_data: dict[str, Any],
    cancel_event: Optional[Event],
) -> Generator[torch.Tensor, None, None]:
    """Generates images based on the provided task data."""
    start = time.time()

    custom_nodes = get_custom_nodes()
    _architectures = get_architectures()
    _checkpoint_files = get_checkpoint_files()

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
                if cancel_event is not None and cancel_event.is_set():
                    raise asyncio.CancelledError("Operation was cancelled.")

                # Run the ImageGenNode
                tensor_images: torch.Tensor = image_gen_node(
                    repo_id=checkpoint_id,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    num_images=num_images,
                    random_seed=random_seed,
                    callback=CancelCallback(cancel_event),
                    # checkpoint_files=checkpoint_files,
                    # architectures=architectures,
                    # device=get_torch_device(),
                )["images"]

                if cancel_event is not None and cancel_event.is_set():
                    raise asyncio.CancelledError("Operation was cancelled.")

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
                logger.info("Placed generation result on queue")
                logger.info(f"Tensor dimensions: {tensor_images.shape}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                yield tensor_images
            except StopIteration:
                logger.info("Task was cancelled during image generation.")
                raise asyncio.CancelledError("Operation was cancelled.")
            except asyncio.CancelledError:
                logger.info("Task was cancelled during image generation.")
                raise

            except Exception as e:
                traceback.print_exc()
                logger.error(
                    f"Error generating images for model '{checkpoint_id}': {e}"
                )

    except asyncio.CancelledError:
        logger.info("Task was cancelled.")
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in image generation workflow: {e}")

    logger.info(f"Image generated in {time.time() - start} seconds")

    # Signal end of generation to IO process
    # tensor_queue.put((None, None))

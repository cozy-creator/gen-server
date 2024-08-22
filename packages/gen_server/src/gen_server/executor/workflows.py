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


async def generate_images_with_lora(
    task_data: dict[str, Any],
    cancel_event: Optional[Event],
) -> Generator[torch.Tensor, None, None]:
    """Generates images based on the provided task data, with LoRA support."""
    start = time.time()

    custom_nodes = get_custom_nodes()

    try:
        models = task_data.get("models", {})
        positive_prompt = task_data.get("positive_prompt")
        negative_prompt = task_data.get("negative_prompt", "")
        random_seed = task_data.get("random_seed")
        aspect_ratio: str = task_data.get("aspect_ratio", "1/1")
        lora_path = task_data.get("lora_path")
        model_scale = task_data.get("model_scale", 1.0)
        text_encoder_scale = task_data.get("text_encoder_scale", 1.0)
        text_encoder_2_scale = task_data.get("text_encoder_2_scale", 1.0)
        adapter_name = task_data.get("adapter_name", None)

        # Get the LoraPrepNode, ControlNetPrepNode and ImageGenNode
        lora_prep_node = custom_nodes["core_extension_1.load_lora_node"]()
        image_gen_node = custom_nodes["core_extension_1.image_gen_node"]()
        controlnet_preprocessor_node = custom_nodes["core_extension_1.controlnet_preprocessor_node"]()

        # Prepare LoRA information
        lora_info = None
        if lora_path:
            lora_info = lora_prep_node(
                lora_path=lora_path,
                model_scale=model_scale,
                text_encoder_scale=text_encoder_scale,
                text_encoder_2_scale=text_encoder_2_scale,
                adapter_name=adapter_name
            )

        # Prepare ControlNet input
        controlnet_info = None
        if task_data.get("controlnet_preprocessor"):
            control_image = controlnet_preprocessor_node(
                image=task_data["input_image"],
                preprocessor=task_data["controlnet_preprocessor"],
                threshold1=task_data.get("canny_threshold1", 100),
                threshold2=task_data.get("canny_threshold2", 200)
            )["control_image"]
            
            controlnet_info = {
                "model_id": task_data["controlnet_model_id"],
                "control_image": control_image,
                "conditioning_scale": task_data.get("controlnet_conditioning_scale", 1.0),
                "guess_mode": task_data.get("controlnet_guess_mode", False)
            }

        for checkpoint_id, num_images in models.items():
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError("Operation was cancelled.")

            try:
                # # Get the actual repo_id from the model config
                # model_config_entry = model_config['models'].get(model_id)
                # if not model_config_entry:
                #     raise ValueError(f"Model {model_id} not found in configuration.")
                
                # repo_id = model_config_entry['repo'].replace('hf:', '')

                # Run the ImageGenNode with LoRA information
                result = await image_gen_node(
                    repo_id=checkpoint_id,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    num_images=num_images,
                    random_seed=random_seed,
                    callback=CancelCallback(cancel_event),
                    lora_info=lora_info,
                    controlnet_info=controlnet_info
                )

                tensor_images: torch.Tensor = result["images"]

                if cancel_event is not None and cancel_event.is_set():
                    raise asyncio.CancelledError("Operation was cancelled.")

                tensor_images = tensor_images.to("cpu")
                tensor_images.share_memory_()
                logger.info(f"Generated images for model '{checkpoint_id}' with LoRA. Tensor dimensions: {tensor_images.shape}")

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
                logger.error(f"Error generating images for model '{checkpoint_id}' with LoRA: {str(e)}")
                raise

    except asyncio.CancelledError:
        logger.info("Task was cancelled.")
        raise
    except Exception as e:
        logger.error(f"Error in LoRA-enhanced image generation workflow: {str(e)}")
        raise

    logger.info(f"LoRA-enhanced image generation completed in {time.time() - start} seconds")

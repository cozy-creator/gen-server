import asyncio
import queue
import time
from typing import Type
from multiprocessing.connection import Connection

from PIL import PngImagePlugin
import torch


from ..base_types.pydantic_models import RunCommandConfig
from ..globals import (
    CustomNode,
    CheckpointMetadata,
    update_architectures,
    update_checkpoint_files,
    update_custom_nodes,
)
from .workflows import generate_images_non_io
from ..config import set_config
from ..utils.file_handler import FileHandler, get_file_handler
from ..utils.image import tensor_to_pil

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def upload_batch(
    file_handler: FileHandler, tensor_batch: torch.Tensor, response_conn: Connection
):
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Description", "Generated by gen_server")
    metadata.add_text("Author", "gen_server")

    pil_images = tensor_to_pil(tensor_batch)

    print("Starting to upload PNG files")
    async for file_url in file_handler.upload_png_files(pil_images, metadata):
        print(f"File uploaded successfully. URL: {file_url}")
        response_conn.send(file_url)


async def start_gpu_worker_non_io(
    task_queue: queue.Queue,
    cozy_config: RunCommandConfig,
    custom_nodes: dict[str, Type[CustomNode]],
    checkpoint_files: dict[str, CheckpointMetadata],
    architectures: dict,
):
    set_config(cozy_config)
    update_custom_nodes(custom_nodes)
    update_architectures(architectures)
    update_checkpoint_files(checkpoint_files)

    logger = logging.getLogger(__name__)
    file_handler = get_file_handler(cozy_config)

    print("GPU worker started", flush=True)

    while True:
        try:
            data, response_conn = task_queue.get(timeout=1.0)

            if data is None:  # Use None as a signal to stop the worker
                logger.info("Received stop signal. GPU-worker shutting down.")
                break

            if not isinstance(data, dict):
                logger.error(f"Invalid data received: {data}")
                response_conn.send(None)  # Signal error to API server
                response_conn.close()
                continue

            start_time = time.time()
            try:
                # Generate images and upload the images in the current process
                tensor_images = generate_images_non_io(data)
                if tensor_images is not None:
                    try:
                        await upload_batch(file_handler, tensor_images, response_conn)
                    except Exception as e:
                        logger.error(f"Error uploading images: {str(e)}")
                        response_conn.send(None)

            except Exception as e:
                logger.error(f"Error in image generation: {str(e)}")
                response_conn.send(None)  # Signal error to API server
            finally:
                end_time = time.time()
                # Signal end of generation to IO process, s it can close out connection
                execution_time = end_time - start_time
                logger.info(
                    f"Time taken to generate images: {execution_time:.2f} seconds"
                )

            # We don't need to wait for the future here, as sync_response handles the communication

        except queue.Empty:
            # No new job, continue the loop
            continue

        except Exception as e:
            logger.error(f"Unexpected error in gpu-worker: {str(e)}")

    logger.info("GPU-worker shut down complete")


def start_gpu_worker(
    task_queue: queue.Queue,
    cozy_config: RunCommandConfig,
    custom_nodes: dict[str, Type[CustomNode]],
    checkpoint_files: dict[str, CheckpointMetadata],
    architectures: dict,
):
    asyncio.run(
        start_gpu_worker_non_io(
            task_queue, cozy_config, custom_nodes, checkpoint_files, architectures
        )
    )


# async def run_worker(request_queue: multiprocessing.Queue):
#     with ThreadPoolExecutor() as executor:
#         while True:
#             if not request_queue.empty():
#                 (data, response_queue) = request_queue.get()
#                 if not isinstance(data, dict):
#                     print("Invalid data received")
#                     continue

#                 try:
#                     future = executor.submit(generate_images, **data)
#                     async for image_metadata in iterate_results(future):
#                         response_queue.put(image_metadata)
#                 except StopIteration:
#                     print("Images generated")

#             else:
#                 time.sleep(1)


# async def iterate_results(future: Future[AsyncGenerator[FileURL, None]]):
#     while not future.done():
#         await asyncio.sleep(0.1)

#     async for result in future.result():
#         yield result

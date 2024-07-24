from platform import node
import queue
from typing import Optional, Type, Any

from gen_server.config import set_config

from ..base_types.pydantic_models import RunCommandConfig
from ..globals import (
    CustomNode,
    CheckpointMetadata,
    update_architectures,
    update_checkpoint_files,
    update_custom_nodes,
)
from .workflows import generate_images

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_gpu_worker(
    task_queue: queue.Queue,
    tensor_queue: queue.Queue,
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

            try:
                # Generate images in the current process
                generate_images(
                    task_data=data,
                    tensor_queue=tensor_queue,
                    response_conn=response_conn,
                )
            except Exception as e:
                logger.error(f"Error in image generation: {str(e)}")
                response_conn.send(None)  # Signal error to API server
            finally:
                # Signal end of generation to IO process, s it can close out connection
                tensor_queue.put((None, response_conn))

            # We don't need to wait for the future here, as sync_response handles the communication

        except queue.Empty:
            # No new job, continue the loop
            continue

        except Exception as e:
            logger.error(f"Unexpected error in gpu-worker: {str(e)}")

    logger.info("GPU-worker shut down complete")


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

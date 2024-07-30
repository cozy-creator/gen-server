import asyncio
import queue
from threading import Event
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Type,
    TypeVar,
    Dict,
)

from multiprocessing import managers
from gen_server.api.api_routes import GenerateData
from requests.packages.urllib3.util.retry import Retry
from PIL import PngImagePlugin
import requests
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
from ..utils.file_handler import FileHandler, get_file_handler, FileURL
from ..utils.image import tensor_to_pil
from requests.adapters import HTTPAdapter

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)


async def upload_batch(
    file_handler: FileHandler, tensor_batch: torch.Tensor
) -> AsyncGenerator[FileURL, None]:
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Description", "Generated by gen_server")
    metadata.add_text("Author", "gen_server")

    pil_images = tensor_to_pil(tensor_batch)

    logger.info("Starting to upload PNG files")
    async for file_url in file_handler.upload_png_files(pil_images, metadata):
        yield file_url


def invoke_webhook(webhook_url: str, file_url: FileURL):
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        headers = {"Content-Type": "application/json"}
        response = session.post(
            webhook_url, json={"file_url": file_url}, headers=headers, timeout=10
        )
        response.raise_for_status()
        logger.info("Webhook invoked successfully! Response: %s", response.text)
    except requests.exceptions.RequestException as e:
        logger.error("Request error occurred: %s", e)


async def check_cancellation(cancel_event: Event):
    while not cancel_event.is_set():
        await asyncio.sleep(0.1)
    raise asyncio.CancelledError("Operation was cancelled.")


async def cancellable(
    event: Event, task: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
) -> T:
    task_future = asyncio.ensure_future(task(*args, **kwargs))
    cancel_future = asyncio.ensure_future(check_cancellation(event))

    done, pending = await asyncio.wait(
        [task_future, cancel_future], return_when=asyncio.FIRST_COMPLETED
    )

    for future in pending:
        future.cancel()

    if task_future in done:
        return task_future.result()
    else:
        raise asyncio.CancelledError("Operation was cancelled.")


async def start_gpu_worker_non_io(
    task_queue: queue.Queue,
    cancel_registry: managers.DictProxy,
    cozy_config: RunCommandConfig,
    custom_nodes: Dict[str, Type[CustomNode]],
    checkpoint_files: Dict[str, CheckpointMetadata],
    architectures: Dict[str, Any],
):
    set_config(cozy_config)
    update_custom_nodes(custom_nodes)
    update_architectures(architectures)
    update_checkpoint_files(checkpoint_files)

    file_handler = get_file_handler(cozy_config)

    logger.info("GPU worker started")

    while True:
        try:
            data, response_conn, request_id = task_queue.get(timeout=1.0)

            if data is None:
                logger.info("Received stop signal. GPU-worker shutting down.")
                break

            if not isinstance(data, GenerateData):
                logger.error(f"Invalid data received: {data.dict()}")
                if response_conn is not None:
                    response_conn.send(None)
                    response_conn.close()
                continue

            cancel_event = None
            if request_id is not None:
                cancel_event = Event()
                cancel_registry[request_id] = cancel_event

                if cancel_event.is_set():
                    logger.info("Task was cancelled. Skipping task.")
                    continue

            async def _generate_images():
                if cancel_event is None:
                    return await generate_images_non_io(data.model_dump())
                return await cancellable(
                    cancel_event, generate_images_non_io, data.model_dump()
                )

            async def _upload_images(images: torch.Tensor):
                if cancel_event is None:
                    return upload_batch(file_handler, images)

                return await cancellable(
                    cancel_event,
                    upload_batch,
                    file_handler,
                    images,
                )

            try:
                images = await _generate_images()
                if images is not None:
                    async for file_url in await _upload_images(images):
                        if response_conn is not None:
                            response_conn.send(file_url)
                        if data.webhook_url is not None:
                            invoke_webhook(data.webhook_url, file_url)
            except asyncio.CancelledError:
                logger.info("Task was cancelled.")
                if response_conn is not None:
                    response_conn.send(None)
            except Exception as e:
                logger.error(f"Error in image generation: {str(e)}")
                if response_conn is not None:
                    response_conn.send(None)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Unexpected error in gpu-worker: {str(e)}")

    logger.info("GPU-worker shut down complete")


def start_gpu_worker(
    task_queue: queue.Queue,
    cancel_registry: managers.DictProxy,
    cozy_config: RunCommandConfig,
    custom_nodes: Dict[str, Type[CustomNode]],
    checkpoint_files: Dict[str, CheckpointMetadata],
    architectures: Dict[str, Any],
):
    asyncio.run(
        start_gpu_worker_non_io(
            task_queue,
            cancel_registry,
            cozy_config,
            custom_nodes,
            checkpoint_files,
            architectures,
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

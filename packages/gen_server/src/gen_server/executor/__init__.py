from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing
import concurrent.futures
from multiprocessing import process
from multiprocessing.queues import Queue
from multiprocessing.connection import Connection
import queue
import time
import torch
import asyncio
import logging
from typing import Optional, Tuple, List, Any, AsyncGenerator, Type, Dict
from PIL import PngImagePlugin
from PIL import Image
from ..globals import RunCommandConfig, get_custom_nodes, get_checkpoint_files
from ..utils.file_handler import get_file_handler, FileURL
from .io_worker import run_io_worker, run_io_worker_sync
from ..globals import CheckpointMetadata, CustomNode
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Simulating the executor code ===

# This is a fixed prebuilt workflow; it's a placeholder for now
def generate_images(
    task_data: Dict[str, Any],
    tensor_queue: Queue, 
    custom_nodes: Dict[str, Type[CustomNode]],
    checkpoint_files: Dict[str, CheckpointMetadata],
) -> None:
    """Generates images based on the provided task data."""
    start = time.time()


    try:
        models = task_data.get("models", {})
        positive_prompt = task_data.get("positive_prompt")
        negative_prompt = task_data.get("negative_prompt", "")
        random_seed = task_data.get("random_seed")
        aspect_ratio = task_data.get("aspect_ratio") 

        # Get the ImageGenNode
        image_gen_node = custom_nodes["core_extension_1.image_gen_node"]()

        for checkpoint_id, num_images in models.items():
            try:
                # Determine width and height from aspect ratio
                width, height = aspect_ratio_to_dimensions(aspect_ratio, checkpoint_files[checkpoint_id].category)

                # Run the ImageGenNode 
                generated_images = image_gen_node(
                    checkpoint_id=checkpoint_id,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_images=num_images,
                    random_seed=random_seed,
                )["images"]

                # Process the generated images (convert to tensors and put on the queue)
                tensor_batch = []
                for image in generated_images:
                    tensor_image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0  
                    tensor_batch.append(tensor_image)
                
                tensor_queue.put(tensor_batch)  # Put the entire batch in the queue

            except Exception as e:
                # Log the error and send an error message to the API server
                logger.error(f"Error generating images for model '{checkpoint_id}': {e}")
                error_message = f"Error generating images for model '{checkpoint_id}': {str(e)}"
                tensor_queue.put({"error": error_message}) # Send the error through the tensor queue

    except Exception as e:
        logger.error(f"Error in image generation workflow: {e}")
        tensor_queue.put({"error": str(e)}) # Send a general error



    print(f"Image generated in {time.time() - start} seconds")

    # for idx, img in enumerate(images):
    #     img.save(os.path.join(output_folder, f"generated_image_{idx}.png"))

    # print(f"Image generated in {time.time() - start} seconds")

    # if args.run_web_server:
    #     from request_handlers.web_server import start_server

    # if args.run_web_server:
    #     from request_handlers.web_server import start_server

    #     start_server(args.host, args.web_server_port)

    # if args.run_grpc:
    #     from request_handlers.grpc_server import start_server

    #     start_server(args.host, args.grpc_port)

    # Signal end of generation to IO process
    tensor_queue.put(None)

    # free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # tensor_queue.close() # Close the queue from the generator side

    # await asyncio.sleep(0)  # yield control back to the caller

    


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


def aspect_ratio_to_dimensions(
    aspect_ratio: str, model_category: str
) -> Tuple[int, int]:
    aspect_ratio_map = {
        "21/9": {"large": (1536, 640), "default": (896, 384)},
        "16/9": {"large": (1344, 768), "default": (768, 448)},
        "4/3": {"large": (1152, 896), "default": (704, 512)},
        "1/1": {"large": (1024, 1024), "default": (512, 512)},
        "3/4": {"large": (896, 1152), "default": (512, 704)},
        "9/16": {"large": (768, 1344), "default": (448, 768)},
        "9/21": {"large": (640, 1536), "default": (384, 896)},
    }

    if aspect_ratio not in aspect_ratio_map:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")

    size = (
        "large" if (model_category == "SDXL" or model_category == "SD3") else "default"
    )

    return aspect_ratio_map[aspect_ratio][size]


# type alias for the queue item
QueueItem = tuple[dict[str, Any], Connection]

def run_worker(
    task_queue: Queue,
    executor: ProcessPoolExecutor,
    cozy_config: RunCommandConfig,
    custom_nodes: dict[str, Type[CustomNode]],
    checkpoint_files: dict[str, CheckpointMetadata],
):
    logger = logging.getLogger(__name__)
    active_io_processes = []

    while True:
        try:
            data, response_conn = task_queue.get(timeout=1.0)

            if data is None:  # Use None as a signal to stop the worker
                logger.info("Received stop signal. Worker shutting down.")
                break

            if not isinstance(data, dict):
                logger.error(f"Invalid data received: {data}")
                response_conn.send(None)  # Signal error to API server
                response_conn.close()
                continue

            io_process = run_gen_worker(data, response_conn, cozy_config, custom_nodes, checkpoint_files)
            active_io_processes.append(io_process)

            # Clean up completed IO processes
            for process in active_io_processes:
                if not process.is_alive():
                    process.join()

            # We don't need to wait for the future here, as sync_response handles the communication

        except queue.Empty:
            # No new job, continue the loop
            continue
            
        except Exception as e:
            logger.error(f"Unexpected error in worker: {str(e)}")

    logger.info("Worker shut down complete")


def run_gen_worker(
    data: dict[str, Any],
    response_conn: Connection,
    cozy_config: RunCommandConfig,
    custom_nodes: dict[str, Type[CustomNode]],
    checkpoint_files: dict[str, CheckpointMetadata],
):
    file_handler = get_file_handler(cozy_config)

    # Create a tensor queue so that the gen-worker process can push its finished
    # files into the io-worker process.
    tensor_queue = multiprocessing.Queue()

    # Start the IO worker process
    io_process = multiprocessing.Process(
        target=run_io_worker_sync,
        args=(tensor_queue, response_conn, file_handler)
    )
    io_process.start()

    # print(data)

    try:
        # Generate images in the current process
        generate_images(
            task_data=data,
            tensor_queue=tensor_queue,
            custom_nodes=custom_nodes,
            checkpoint_files=checkpoint_files
        )
    except Exception as e:
        logger.error(f"Error in image generation: {str(e)}")
        tensor_queue.put({"error": str(e)})  # Send error through tensor queue
    finally:
        # Wait for the IO process to complete
        tensor_queue.put(None)
        io_process.join()
        logger.info("Signaled end of generation to IO process")

    # We don't need to wait for the IO process to complete
    # It will handle closing the response connection and tensor-queue
    # when it's done

    # For clean up later
    return io_process


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

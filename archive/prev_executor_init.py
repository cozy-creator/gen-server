from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing.queues import Queue
from multiprocessing.connection import Connection
import queue
import time
import torch
import logging
from typing import Optional, Tuple, List, Any, Type

from gen_server.utils.device import get_torch_device
from ..globals import RunCommandConfig, get_custom_nodes
from ..utils.file_handler import get_file_handler
from .io_worker import run_io_worker
from ..globals import CheckpointMetadata, CustomNode
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Simulating the executor code ===


# This is a fixed prebuilt workflow; it's a placeholder for now
def generate_images(
    models: dict[str, int],
    positive_prompt: str,
    negative_prompt: str,
    random_seed: Optional[int],
    aspect_ratio: str,
    tensor_queue: multiprocessing.Queue,
    custom_nodes: dict[str, Type[CustomNode]],
    checkpoint_files: dict[str, CheckpointMetadata],
):
    start = time.time()

    print(custom_nodes)

    # custom_nodes = get_custom_nodes()
    # checkpoint_files = get_checkpoint_files()

    # ) -> Generator[list[PilImagePlugin.PngImageFile], None, None]:
    # Simulate image generation and yield URLs
    for checkpoint_id, num_images in models.items():
        # Yield a list of placeholder URLs and then sleep
        # yield [{"url": f"placeholder_url_{i}.jpg"} for i in range(num_images)]
        # await asyncio.sleep(1)  # Sleep for 1 second

        # === Node 1: Load Checkpoint ===

        checkpoint_metadata = checkpoint_files.get(checkpoint_id, None)
        print(checkpoint_metadata)
        if checkpoint_metadata is None:
            print(f"No checkpoint file found for model: {checkpoint_id}")
            continue  # skip to next model

        LoadCheckpoint = custom_nodes["core_extension_1.load_checkpoint"]
        load_checkpoint = LoadCheckpoint()

        # figure out what outputs we need from this node
        output_keys = {}
        components = load_checkpoint(
            checkpoint_metadata.file_path,
            output_keys=output_keys,
            device=get_torch_device(),
        )

        # components = load_checkpoint("SDXL VAE", "fp16")

        # print("Number of items loaded:", len(components))
        # for key in components.keys():
        #     print(f"Model key: {key}")

        # === Node 2: Create Pipe ===
        CreatePipe = custom_nodes["core_extension_1.create_pipe"]
        create_pipe = CreatePipe()

        # ???
        # pipe_type = create_pipe.determine_output()
        # print(pipe_type)

        # signature = inspect.signature(create_pipe.__call__)
        # print(signature)

        # Detailed parameter analysis
        # for name, param in signature.parameters.items():
        #     print(f"Parameter Name: {name}")
        #     print(f"  Kind: {param.kind}")
        #     print(f"  Default: {param.default if param.default is not inspect.Parameter.empty else 'No default'}")
        #     print(f"  Annotation: {param.annotation if param.annotation is not inspect.Parameter.empty else 'No annotation'}")

        match checkpoint_metadata.category:
            case "SD1":
                vae = components["core_extension_1.vae"].model
                unet = components["core_extension_1.sd1_unet"].model
                text_encoder_1 = components["core_extension_1.sd1_text_encoder"].model

                pipe = create_pipe(vae=vae, text_encoder=text_encoder_1, unet=unet)
                cfg = 7.0
                num_inference_steps = 25

            case "SDXL":
                vae = components["core_extension_1.vae"].model
                unet = components["core_extension_1.sdxl_unet"].model
                text_encoder_1 = components[
                    "core_extension_1.sdxl_text_encoder_1"
                ].model
                text_encoder_2 = components["core_extension_1.text_encoder_2"].model

                sdxl_type = checkpoint_metadata.components["core_extension_1.vae"][
                    "input_space"
                ].lower()

                pipe = create_pipe(
                    vae=vae,
                    text_encoder=text_encoder_1,
                    text_encoder_2=text_encoder_2,
                    unet=unet,
                    model_type=sdxl_type,
                )
                cfg = 7.0
                num_inference_steps = 20

            case "SD3":
                vae = components["core_extension_1.vae"].model
                unet = components["core_extension_1.sd3_unet"].model
                text_encoder_1 = components["core_extension_1.sd3_text_encoder_1"].model
                text_encoder_2 = components["core_extension_1.text_encoder_2"].model
                text_encoder_3 = components["core_extension_1.sd3_text_encoder_3"].model

                pipe = create_pipe(
                    vae=vae,
                    text_encoder=text_encoder_1,
                    text_encoder_2=text_encoder_2,
                    text_encoder_3=text_encoder_3,
                    unet=unet,
                )
                cfg = 7.0
                num_inference_steps = 28

            case _:
                raise ValueError(f"Unknown category: {checkpoint_metadata.category}")

        # Presumably we'd figure this out from the edges?
        # SD3
        # vae = components["core_extension_1.sd1_vae"].model
        # unet = components["core_extension_1.sd3_unet"].model
        # text_encoder_1 = components["core_extension_1.sd3_text_encoder_1"].model
        # text_encoder_2 = components["core_extension_1.sd3_text_encoder_2"].model
        # text_encoder_3 = components["core_extension_1.sd3_text_encoder_3"].model

        # pipe = create_pipe(
        #     vae=vae,
        #     text_encoder=text_encoder_1,
        #     text_encoder_2=text_encoder_2,
        #     text_encoder_3=text_encoder_3,
        #     unet=unet
        # )
        # pipe.to('cuda')

        # SDXL
        # vae = models["core_extension_1.sd1_vae"].model
        # unet = models["core_extension_1.sdxl_unet"].model
        # text_encoder_1 = models["core_extension_1.sdxl_text_encoder_1"].model
        # text_encoder_2 = models["core_extension_1.sdxl_text_encoder_2"].model

        # pipe = create_pipe(
        #     vae=vae,
        #     text_encoder=text_encoder_1,
        #     text_encoder_2=text_encoder_2,
        #     unet=unet
        # )

        # SD1.5
        # vae = components["core_extension_1.sd1_vae"].model
        # unet = components["core_extension_1.sd1_unet"].model
        # text_encoder_1 = components["core_extension_1.sd1_text_encoder"].model

        # pipe = create_pipe(
        #     vae=vae,
        #     text_encoder=text_encoder_1,
        #     unet=unet
        # )

        # === Node 3: Run Pipe ===
        RunPipe = custom_nodes["core_extension_1.run_pipe"]
        run_pipe = RunPipe()

        # Determine the width and height based on the aspect ratio and base model
        width, height = aspect_ratio_to_dimensions(
            aspect_ratio, checkpoint_metadata.category
        )

        # output_type = run_pipe.determine_output()
        # print(output_type)

        # execute the 3rd node
        # prompt = "Beautiful anime woman with dark-skin"
        # negative_prompt = "poor quality, worst quality, watermark, blurry"

        pil_images = run_pipe(
            pipe,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_images=num_images,
            guidance_scale=cfg,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator().manual_seed(random_seed)
            if random_seed is not None
            else None,
        )

        # TO DO: this is temporary until we fix the run_pipe output
        # Convert PIL images to torch tensors
        tensor_images = []
        for pil_image in pil_images:
            # Convert PIL image to numpy array
            np_image = np.array(pil_image)
            # Convert numpy array to torch tensor and normalize to range [0, 1]
            tensor_image = torch.from_numpy(np_image).float() / 255.0
            # Rearrange dimensions from (H, W, C) to (C, H, W)
            tensor_image = tensor_image.permute(2, 0, 1)
            tensor_images.append(tensor_image)

        # Stack individual tensors into a single tensor
        tensor_images = torch.stack(tensor_images)
        del pipe

        tensor_queue.put(pil_images)

    # loop = asyncio.get_event_loop()
    # with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
    #     try:
    #         pil_images = await loop.run_in_executor(
    #             pool,
    #             generate_images_internal,
    #             models,
    #             positive_prompt,
    #             negative_prompt,
    #             random_seed,
    #             aspect_ratio,
    #         )
    #     except Exception as e:
    #         logger.error(f"Error in ProcessPoolExecutor: {e}")
    #         return

    # Save Images
    # images[0].save("output.png")

    # diffusion_pytorch_model.fp16.safetensors
    # playground-v2.5-1024px-aesthetic.fp16.safetensors
    # diffusion_pytorch_model.safetensors
    # darkSushi25D25D_v40.safetensors
    # sd3_medium_incl_clips_t5xxlfp8.safetensors
    # sd_xl_base_1.0.safetensors

    # Save the images
    # === Node 4: Save Files ===

    # test metadata
    # metadata = PngImagePlugin.PngInfo()
    # metadata.add_text("Description", "Generated by gen_server")
    # metadata.add_text("Author", "gen_server")

    # file_handler = get_file_handler()

    # if pil_images is None:
    #     return

    # async for file_metadata in file_handler.upload_png_files(pil_images, metadata):
    #     yield file_metadata

    # SaveNode = custom_nodes["image_utils.save_file"]
    # save_node = SaveNode()
    # urls: dict[str, Any] = save_node(images=pil_images, temp=False)

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

    # free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

            io_process = run_gen_worker(
                data, response_conn, cozy_config, custom_nodes, checkpoint_files
            )
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
        target=run_io_worker, args=(tensor_queue, response_conn, file_handler)
    )
    io_process.start()

    try:
        # Generate images in the current process
        generate_images(
            **data,
            tensor_queue=tensor_queue,
            custom_nodes=custom_nodes,
            checkpoint_files=checkpoint_files,
        )
    except Exception as e:
        logger.error(f"Error in image generation: {str(e)}")
        response_conn.send(None)  # Signal error to API server
    finally:
        # Signal end of generation to IO process
        tensor_queue.put(None)

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

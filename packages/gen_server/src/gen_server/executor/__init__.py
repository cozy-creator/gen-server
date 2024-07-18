from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing
import concurrent.futures
from multiprocessing import process
from multiprocessing.connection import Connection
import queue
import time
import torch
import asyncio
import logging
from typing import Optional, Tuple, List, Any, AsyncGenerator
from PIL import PngImagePlugin
from PIL import Image
from ..globals import CUSTOM_NODES, CHECKPOINT_FILES
from ..utils.file_handler import get_file_handler, FileURL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Simulating the executor code ===


def generate_images_internal(
    models: dict[str, int],
    positive_prompt: str,
    negative_prompt: str,
    random_seed: Optional[int],
    aspect_ratio: str,
) -> Optional[list[Image.Image]]:
    # ) -> Generator[list[PilImagePlugin.PngImageFile], None, None]:
    # Simulate image generation and yield URLs
    for checkpoint_id, num_images in models.items():
        # Yield a list of placeholder URLs and then sleep
        # yield [{"url": f"placeholder_url_{i}.jpg"} for i in range(num_images)]
        # await asyncio.sleep(1)  # Sleep for 1 second

        # === Node 1: Load Checkpoint ===

        checkpoint_metadata = CHECKPOINT_FILES.get(checkpoint_id, None)
        print(checkpoint_metadata)
        if checkpoint_metadata is None:
            print(f"No checkpoint file found for model: {checkpoint_id}")
            continue  # skip to next model

        LoadCheckpoint = CUSTOM_NODES["core_extension_1.load_checkpoint"]
        load_checkpoint = LoadCheckpoint()

        # figure out what outputs we need from this node
        output_keys = {}
        components = load_checkpoint(
            checkpoint_metadata.file_path,
            output_keys=output_keys,
            device="cuda",
        )

        # components = load_checkpoint("SDXL VAE", "fp16")

        # print("Number of items loaded:", len(components))
        # for key in components.keys():
        #     print(f"Model key: {key}")

        # === Node 2: Create Pipe ===
        CreatePipe = CUSTOM_NODES["core_extension_1.create_pipe"]
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
        run_pipe = CUSTOM_NODES["core_extension_1.run_pipe"]()

        # Determine the width and height based on the aspect ratio and base model
        width, height = aspect_ratio_to_dimensions(
            aspect_ratio, checkpoint_metadata.category
        )

        # output_type = run_pipe.determine_output()
        # print(output_type)

        # execute the 3rd node
        # prompt = "Beautiful anime woman with dark-skin"
        # negative_prompt = "poor quality, worst quality, watermark, blurry"

        images = run_pipe(
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
        del pipe

        return images


# This is a fixed prebuilt workflow; it's a placeholder for now
async def generate_images(
    models: dict[str, int],
    positive_prompt: str,
    negative_prompt: str,
    random_seed: Optional[int],
    aspect_ratio: str,
) -> AsyncGenerator[FileURL, None]:
    start = time.time()

    pil_images = generate_images_internal(
        models,
        positive_prompt,
        negative_prompt,
        random_seed,
        aspect_ratio,
    )

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
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Description", "Generated by gen_server")
    metadata.add_text("Author", "gen_server")

    file_handler = get_file_handler()

    if pil_images is None:
        return

    async for file_metadata in file_handler.upload_png_files(pil_images, metadata):
        yield file_metadata

    # SaveNode = CUSTOM_NODES["image_utils.save_file"]
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

    await asyncio.sleep(0)  # yield control back to the caller


async def generate_images_from_repo(
    repo_id: str,
    components: List[str],
    positive_prompt: str,
    negative_prompt: str,
    random_seed: Optional[int],
    aspect_ratio: Tuple[int, int],
):
    start = time.time()

    LoadComponents = CUSTOM_NODES["core_extension_1.load_components"]
    load_components = LoadComponents()

    components = load_components(repo_id, components)

    # runwayml/stable-diffusion-v1-5

    CreatePipe = CUSTOM_NODES["core_extension_1.create_pipe"]
    create_pipe = CreatePipe()

    pipe = create_pipe(loaded_components=components)

    run_pipe = CUSTOM_NODES["core_extension_1.run_pipe"]()

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

    SaveNode = CUSTOM_NODES["image_utils.save_file"]
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


def run_worker(task_queue: multiprocessing.Queue, executor: ProcessPoolExecutor):
    logger = logging.getLogger(__name__)
    
    while True:
        try:
            # Check for new jobs
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

                # Submit the job to the process pool
                _future = executor.submit(run_sync_response, data, response_conn)
                
                # We don't need to wait for the future here, as sync_response handles the communication

            except queue.Empty:
                # No new job, continue the loop
                continue

        except Exception as e:
            logger.error(f"Unexpected error in worker: {str(e)}")

    logger.info("Worker shut down complete")


def run_sync_response(data: dict[str, Any], response_conn: Connection):
    asyncio.run(sync_response(data, response_conn))

async def sync_response(data: dict[str, Any], response_conn: Connection):
    try:
        async for file_url in generate_images(**data):
            response_conn.send(file_url)
    except Exception as e:
        logger.error(f"Error in generate_images: {str(e)}")
        response_conn.send(None)  # Signal error to API server
    finally:
        response_conn.send(StopIteration)  # Signal end of stream
        response_conn.close()

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

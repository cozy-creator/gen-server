import time
import torch
import logging
from typing import Optional, AsyncGenerator, Type, Dict, Any
from PIL import PngImagePlugin
from PIL import Image
import numpy as np
from multiprocessing.managers import SyncManager
from multiprocessing.connection import Connection

from queue import Queue

from ..globals import CheckpointMetadata, CustomNode, Architecture

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Simulating the executor code ===

# This is a fixed prebuilt workflow; it's a placeholder for now
def generate_images(
    task_data: dict[str, Any],
    tensor_queue: Queue, 
    response_conn: Connection,
    custom_nodes: dict[str, Type[CustomNode]],
    checkpoint_files: dict[str, CheckpointMetadata],
    architectures: dict[str, Architecture]
) -> None:
    """Generates images based on the provided task data."""
    start = time.time()

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
                    architectures=architectures
                )["images"]

                # Process the generated images (convert to tensors and put on the queue)
                # tensor_batch = []
                # for image in generated_images:
                #     tensor_image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0  
                #     tensor_batch.append(tensor_image)
                
                # Stack individual tensors into a single tensor
                # tensor_images = torch.stack(tensor_batch)
                
                tensor_images = tensor_images.to("cpu")
                tensor_queue.put((tensor_images, response_conn))
                
                print('Placed generation result on queue')
                print(f"Tensor dimensions: {tensor_images.shape}")
                print(f"Response connection: {response_conn}")
                
                # Free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                # Log the error and send an error message to the API server
                logger.error(f"Error generating images for model '{checkpoint_id}': {e}")

    except Exception as e:
        logger.error(f"Error in image generation workflow: {e}")
        # tensor_queue.put((None, None)) # Tell the io-worker that we're done

    print(f"Image generated in {time.time() - start} seconds")

    # Signal end of generation to IO process
    # tensor_queue.put((None, None))


# === Simulating the executor code ===

# This is a fixed prebuilt workflow; it's a placeholder for now
# def generate_images(
#     models: dict[str, int],
#     positive_prompt: str,
#     negative_prompt: str,
#     random_seed: Optional[int],
#     aspect_ratio: str,
#     tensor_queue: SyncManager.Queue,
#     response_conn: Connection,
#     custom_nodes: dict[str, Type[CustomNode]],
#     checkpoint_files: dict[str, CheckpointMetadata],
# ) -> None:
#     start = time.time()

#     # custom_nodes = get_custom_nodes()
#     # checkpoint_files = get_checkpoint_files()

#     # ) -> Generator[list[PilImagePlugin.PngImageFile], None, None]:
#     # Simulate image generation and yield URLs
#     for checkpoint_id, num_images in models.items():
#         # Yield a list of placeholder URLs and then sleep
#         # yield [{"url": f"placeholder_url_{i}.jpg"} for i in range(num_images)]
#         # await asyncio.sleep(1)  # Sleep for 1 second

#         # === Node 1: Load Checkpoint ===

#         checkpoint_metadata = checkpoint_files.get(checkpoint_id, None)
#         print(checkpoint_metadata)
#         if checkpoint_metadata is None:
#             print(f"No checkpoint file found for model: {checkpoint_id}")
#             continue  # skip to next model

#         LoadCheckpoint = custom_nodes["core_extension_1.load_checkpoint"]
#         load_checkpoint = LoadCheckpoint()

#         # figure out what outputs we need from this node
#         output_keys = {}
#         components = load_checkpoint(
#             checkpoint_metadata.file_path,
#             output_keys=output_keys,
#             device="cuda",
#         )

#         # components = load_checkpoint("SDXL VAE", "fp16")

#         # print("Number of items loaded:", len(components))
#         # for key in components.keys():
#         #     print(f"Model key: {key}")

#         # === Node 2: Create Pipe ===
#         CreatePipe = custom_nodes["core_extension_1.create_pipe"]
#         create_pipe = CreatePipe()


#         match checkpoint_metadata.category:
#             case "SD1":
#                 vae = components["core_extension_1.vae"].model
#                 unet = components["core_extension_1.sd1_unet"].model
#                 text_encoder_1 = components["core_extension_1.sd1_text_encoder"].model

#                 pipe = create_pipe(vae=vae, text_encoder=text_encoder_1, unet=unet)
#                 cfg = 7.0
#                 num_inference_steps = 25

#             case "SDXL":
#                 vae = components["core_extension_1.vae"].model
#                 unet = components["core_extension_1.sdxl_unet"].model
#                 text_encoder_1 = components[
#                     "core_extension_1.sdxl_text_encoder_1"
#                 ].model
#                 text_encoder_2 = components["core_extension_1.text_encoder_2"].model

#                 sdxl_type = checkpoint_metadata.components["core_extension_1.vae"][
#                     "input_space"
#                 ].lower()

#                 pipe = create_pipe(
#                     vae=vae,
#                     text_encoder=text_encoder_1,
#                     text_encoder_2=text_encoder_2,
#                     unet=unet,
#                     model_type=sdxl_type,
#                 )
#                 cfg = 7.0
#                 num_inference_steps = 20

#             case "SD3":
#                 vae = components["core_extension_1.vae"].model
#                 unet = components["core_extension_1.sd3_unet"].model
#                 text_encoder_1 = components["core_extension_1.sd3_text_encoder_1"].model
#                 text_encoder_2 = components["core_extension_1.text_encoder_2"].model
#                 text_encoder_3 = components["core_extension_1.sd3_text_encoder_3"].model

#                 pipe = create_pipe(
#                     vae=vae,
#                     text_encoder=text_encoder_1,
#                     text_encoder_2=text_encoder_2,
#                     text_encoder_3=text_encoder_3,
#                     unet=unet,
#                 )
#                 cfg = 7.0
#                 num_inference_steps = 28

#             case _:
#                 raise ValueError(f"Unknown category: {checkpoint_metadata.category}")



#         # === Node 3: Run Pipe ===
#         RunPipe = custom_nodes["core_extension_1.run_pipe"]
#         run_pipe = RunPipe()

#         # Determine the width and height based on the aspect ratio and base model
#         width, height = aspect_ratio_to_dimensions(
#             aspect_ratio, checkpoint_metadata.category
#         )


#         pil_images = run_pipe(
#             pipe,
#             prompt=positive_prompt,
#             negative_prompt=negative_prompt,
#             width=width,
#             height=height,
#             num_images=num_images,
#             guidance_scale=cfg,
#             num_inference_steps=num_inference_steps,
#             generator=torch.Generator().manual_seed(random_seed)
#             if random_seed is not None
#             else None,
#         )
        
#         # TO DO: this is temporary until we fix the run_pipe output
#         # Convert PIL images to torch tensors
#         tensor_images = []
#         for pil_image in pil_images:
#             # Convert PIL image to numpy array
#             np_image = np.array(pil_image)
#             # Convert numpy array to torch tensor and normalize to range [0, 1]
#             tensor_image = torch.from_numpy(np_image).float() / 255.0
#             # Rearrange dimensions from (H, W, C) to (C, H, W)
#             tensor_image = tensor_image.permute(2, 0, 1)
#             tensor_images.append(tensor_image)
        
#         # Stack individual tensors into a single tensor
#         tensor_images = torch.stack(tensor_images)
#         del pipe
    
#         tensor_queue.put((tensor_images, response_conn))


#     print(f"Image generated in {time.time() - start} seconds")


#     # free up memory
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     # await asyncio.sleep(0)  # yield control back to the caller
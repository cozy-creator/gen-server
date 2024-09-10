import asyncio
from typing import Dict, Any, List
from gen_server.executor.workflows import flux_train_workflow, image_regen_workflow, generate_images_with_lora, generate_images_unified
# from gen_server.base_types.custom_node import get_custom_nodes
from gen_server.config import init_config
from gen_server.globals import update_custom_nodes
from gen_server.utils.extension_loader import load_extensions
import time
from gen_server.base_types.custom_node import custom_node_validator
from gen_server.utils.cli_helpers import parse_known_args_wrapper
import torchvision.transforms as T
from gen_server.utils.image import tensor_to_pil
from gen_server.utils.paths import get_assets_dir
import signal
import os
import sys
import json
from PIL import Image
import hashlib
import io

# cancel_flag = asyncio.Event()

# def signal_handler(signum, frame):
#     print("\nCtrl+C pressed. Cancelling training...")
#     cancel_flag.set()

# signal.signal(signal.SIGINT, signal_handler)



async def test_flux_train(
    image_directory: str,
    lora_name: str,
    image_paths: List[str],
    initial_captions: Dict[str, str] = {},
    use_auto_captioning: bool = False,
    flux_version: str = "schnell",
    training_steps: int = 2000,
    resolution: List[int] = [1024],
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    trigger_word: str = None,
    low_vram: bool = True,
    seed: int = None,
    walk_seed: bool = True
) -> None:
    task_data = {
        "image_directory": image_directory,
        "lora_name": lora_name,
        "image_paths": image_paths,
        "initial_captions": initial_captions,
        "use_auto_captioning": use_auto_captioning,
        "flux_version": flux_version,
        "training_steps": training_steps,
        "resolution": resolution,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "trigger_word": trigger_word,
        "low_vram": low_vram,
        "random_seed": seed,
        "walk_seed": walk_seed
    }

    cancel_event = asyncio.Event()

    def signal_handler(signum, frame):
        print("\nCtrl+C pressed. Cancelling training...")
        cancel_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        async for update in flux_train_workflow(task_data, cancel_event):
            print(update)
            if cancel_event.is_set():
                print("Training cancelled")
                break
            if update.get('type') == 'finished' or update.get('type') == 'cancelled':
                break
    except asyncio.CancelledError:
        print("Training was cancelled")
    finally:
        print("Training process finished")

async def test_image_regen(
    image_path: str,
    mask_path: str = None,
    mask_prompt: str = None,
    prompt: str = "",
    model_id: str = "",
    negative_prompt: str = "",
    num_inference_steps: int = 25,
    strength: float = 0.7,
    feather_radius: int = 0
) -> None:
    task_data = {
        "image": image_path,
        "prompt": prompt,
        "model_id": model_id,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "strength": strength,
        "feather_radius": feather_radius
    }

    if mask_path:
        task_data["mask"] = mask_path
    elif mask_prompt:
        task_data["mask_prompt"] = mask_prompt
    else:
        raise ValueError("Either mask_path or mask_prompt must be provided")

    result = await image_regen_workflow(task_data)
    
    # Convert the result tensor to a PIL Image using the provided function
    output_images = tensor_to_pil(result['regenerated_image'])
    
    # Save each image in the batch
    for i, img in enumerate(output_images):
        img.save(f"regenerated_image_{i}.png")
    
    print(f"Regenerated {len(output_images)} image(s) saved as 'regenerated_image_X.png'")


async def test_generate_images_with_lora(
    positive_prompt: str,
    negative_prompt: str = "",
    models: Dict[str, int] = {"stabilityai/stable-diffusion-xl-base-1.0": 1},
    random_seed: int = None,
    aspect_ratio: str = "1/1",
    lora_path: str = None,
    model_scale: float = 1.0,
    text_encoder_scale: float = 1.0,
    text_encoder_2_scale: float = 1.0,
    adapter_name: str = None,
    input_image: str = None,
    controlnet_preprocessor: str = None,
    controlnet_model_id: str = None,
    canny_threshold1: int = 100,
    canny_threshold2: int = 200,
    controlnet_conditioning_scale: float = 1.0,
    controlnet_guess_mode: bool = False
) -> None:
    task_data = {
        "models": models,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "random_seed": random_seed,
        "aspect_ratio": aspect_ratio,
        "lora_path": lora_path,
        "model_scale": model_scale,
        "text_encoder_scale": text_encoder_scale,
        "text_encoder_2_scale": text_encoder_2_scale,
        "adapter_name": adapter_name,
    }

    if input_image and controlnet_preprocessor and controlnet_model_id:
        task_data.update({
            "input_image": input_image,
            "controlnet_preprocessor": controlnet_preprocessor,
            "controlnet_model_id": controlnet_model_id,
            "canny_threshold1": canny_threshold1,
            "canny_threshold2": canny_threshold2,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "controlnet_guess_mode": controlnet_guess_mode
        })

    cancel_event = asyncio.Event()

    def signal_handler(signum, frame):
        print("\nCtrl+C pressed. Cancelling image generation...")
        cancel_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        async for tensor_images in generate_images_with_lora(task_data, cancel_event):
            print(f"Generated images tensor shape: {tensor_images.shape}")
            
            # Convert tensor images to PIL images and save them
            pil_images = tensor_to_pil(tensor_images)
            for i, img in enumerate(pil_images):
                img.save(f"generated_image_lora_{i}.png")
                print(f"Saved image: generated_image_lora_{i}.png")

            if cancel_event.is_set():
                print("Image generation was cancelled.")
                break
    except asyncio.CancelledError:
        print("Image generation was cancelled.")
    finally:
        print("Image generation process finished.")


async def test_generate_images_unified(
    positive_prompt: str,
    negative_prompt: str = "",
    models: Dict[str, int] = {"stabilityai/stable-diffusion-xl-base-1.0": 1},
    random_seed: int = None,
    aspect_ratio: str = "1/1",
    lora_path: str = None,
    model_scale: float = 1.0,
    text_encoder_scale: float = 1.0,
    text_encoder_2_scale: float = 1.0,
    adapter_name: str = None,
    controlnet_configs: List[Dict[str, Any]] = None,
    ip_adapter_image_path: str = None,
    output_dir: str = "generated_images"
) -> None:
    task_data = {
        "models": models,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "random_seed": random_seed,
        "aspect_ratio": aspect_ratio,
    }

    if lora_path:
        task_data["lora_path"] = lora_path
        task_data["model_scale"] = model_scale
        task_data["text_encoder_scale"] = text_encoder_scale
        task_data["text_encoder_2_scale"] = text_encoder_2_scale
        task_data["adapter_name"] = adapter_name

    if controlnet_configs:
        task_data["controlnet_info_list"] = []
        for config in controlnet_configs:
            image_path = f"{get_assets_dir()}/{config['image_path']}"
            control_image = Image.open(image_path).convert("RGB")
            task_data["controlnet_info_list"].append({
                "model_id": config["model_id"],
                "control_image": control_image,
                "feature_type": config["feature_type"],
                "threshold1": config.get("threshold1", 100),
                "threshold2": config.get("threshold2", 200)
            })

    if ip_adapter_image_path:
        ip_image = Image.open(ip_adapter_image_path).convert("RGB")
        ip_image_tensor = T.ToTensor()(ip_image).unsqueeze(0)
        task_data["ip_adapter_embeds"] = ip_image_tensor  # Note: In a real scenario, you'd process this through the IP-Adapter first

    cancel_event = asyncio.Event()

    def signal_handler(signum, frame):
        print("\nCtrl+C pressed. Cancelling image generation...")
        cancel_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    os.makedirs(output_dir, exist_ok=True)

    try:
        async for tensor_images in generate_images_unified(task_data, cancel_event):
            print(f"Generated images tensor shape: {tensor_images.shape}")
            
            pil_images = tensor_to_pil(tensor_images)
            for i, img in enumerate(pil_images):
                # Generate a hash of the image content
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                img_hash = hashlib.sha256(img_byte_arr).hexdigest()[:16]  # Use first 16 characters of the hash

                # Create a filename with the hash
                filename = f"generated_image_{img_hash}.png"
                filepath = os.path.join(output_dir, filename)

                # Save the image
                img.save(filepath)
                print(f"Saved image: {filepath}")

            if cancel_event.is_set():
                print("Image generation was cancelled.")
                break

    except asyncio.CancelledError:
        print("Image generation was cancelled.")
    finally:
        print("Image generation process finished.")


# async def test_generate_image(prompt: str, num_images: int = 1, other_params: Dict[str, Any] = {}) -> None:
#     custom_nodes = get_custom_nodes()
#     generate_node = custom_nodes["core_extension_1.generate_image_node"]()

#     result = await generate_node(prompt=prompt, num_images=num_images, **other_params)
#     print(result)

def startup_extensions():
    start_time_custom_nodes = time.time()

    update_custom_nodes(
        load_extensions("cozy_creator.custom_nodes", validator=custom_node_validator)
    )

    print(
        f"CUSTOM_NODES loading time: {time.time() - start_time_custom_nodes:.2f} seconds"
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test endpoints for Cozy Creator")
    parser.add_argument("command", choices=["train", "generate", "regen", "generate_lora"], help="Command to run")
    
    # Training arguments
    parser.add_argument("--image_directory", help="Directory with images for training")
    parser.add_argument("--lora_name", help="Name for the LoRA model")
    parser.add_argument("--image_paths", nargs='+', help="List of image paths for training")
    parser.add_argument("--flux_version", default="schnell", help="FLUX version to use")
    parser.add_argument("--training_steps", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--trigger_word", help="Trigger word for the LoRA")
    parser.add_argument("--low_vram", action="store_true", help="Enable low VRAM mode")
    
    # Generation arguments
    # parser.add_argument("--prompt", help="Prompt for image generation")
    # parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")

    # Image regeneration arguments
    # Image regeneration arguments
    parser.add_argument("--image_path", help="Path to the input image")
    parser.add_argument("--mask_path", help="Path to the mask image (optional)")
    parser.add_argument("--mask_prompt", help="Text prompt to generate mask (optional)")
    parser.add_argument("--prompt", help="Prompt for image regeneration")
    parser.add_argument("--model_id", help="Model ID for image regeneration")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt for image regeneration")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--strength", type=float, default=0.7, help="Strength of the inpainting effect")
    parser.add_argument("--feather_radius", type=int, default=0, help="Feather radius for the generated mask")

    # New arguments for generate_lora command
    parser.add_argument("--positive_prompt", required=True, help="Positive prompt for image generation")
    # parser.add_argument("--negative_prompt", default="", help="Negative prompt for image generation")
    parser.add_argument("--models", type=json.loads, default='{"stabilityai/stable-diffusion-xl-base-1.0": 1}', help="JSON string of model IDs and number of images to generate")
    parser.add_argument("--random_seed", type=int, help="Random seed for image generation")
    parser.add_argument("--aspect_ratio", default="1/1", help="Aspect ratio for generated images")
    parser.add_argument("--lora_path", help="Path to LoRA file")
    parser.add_argument("--model_scale", type=float, default=1.0, help="Scale for the model")
    parser.add_argument("--text_encoder_scale", type=float, default=1.0, help="Scale for the text encoder")
    parser.add_argument("--text_encoder_2_scale", type=float, default=1.0, help="Scale for the second text encoder")
    parser.add_argument("--adapter_name", help="Name of the adapter")
    parser.add_argument("--controlnet_configs", type=json.loads, help="JSON string of ControlNet configurations")
    parser.add_argument("--ip_adapter_image_path", help="Path to IP-Adapter input image")

    args = parser.parse_args()

    startup_extensions()
    run_parser = argparse.ArgumentParser(description="Cozy Creator")
    init_config(run_parser, parse_known_args_wrapper)

    if args.command == "train":
        asyncio.run(test_flux_train(
            image_directory=args.image_directory,
            lora_name=args.lora_name,
            image_paths=args.image_paths,
            flux_version=args.flux_version,
            training_steps=args.training_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            trigger_word=args.trigger_word,
            low_vram=args.low_vram
        ))
    elif args.command == "regen":
        asyncio.run(test_image_regen(
            image_path=args.image_path,
            mask_path=args.mask_path,
            mask_prompt=args.mask_prompt,
            prompt=args.prompt,
            model_id=args.model_id,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            strength=args.strength,
            feather_radius=args.feather_radius
        ))
    elif args.command == "generate_lora":
        asyncio.run(test_generate_images_with_lora(
            positive_prompt=args.positive_prompt,
            negative_prompt=args.negative_prompt,
            models=args.models,
            random_seed=args.random_seed,
            aspect_ratio=args.aspect_ratio,
            lora_path=args.lora_path,
            model_scale=args.model_scale,
            text_encoder_scale=args.text_encoder_scale,
            text_encoder_2_scale=args.text_encoder_2_scale,
            adapter_name=args.adapter_name,
            input_image=args.image_path,
            controlnet_preprocessor=args.controlnet_preprocessor,
            controlnet_model_id=args.controlnet_model_id,
            canny_threshold1=args.canny_threshold1,
            canny_threshold2=args.canny_threshold2,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            controlnet_guess_mode=args.controlnet_guess_mode
        ))
    elif args.command == "generate":
        asyncio.run(test_generate_images_unified(
            positive_prompt=args.positive_prompt,
            negative_prompt=args.negative_prompt,
            models=args.models,
            random_seed=args.random_seed,
            aspect_ratio=args.aspect_ratio,
            lora_path=args.lora_path,
            model_scale=args.model_scale,
            text_encoder_scale=args.text_encoder_scale,
            text_encoder_2_scale=args.text_encoder_2_scale,
            adapter_name=args.adapter_name,
            controlnet_configs=args.controlnet_configs,
            ip_adapter_image_path=args.ip_adapter_image_path
        ))
    # elif args.command == "generate":
    #     asyncio.run(test_generate_image(
    #         prompt=args.prompt,
    #         num_images=args.num_images
    #     ))
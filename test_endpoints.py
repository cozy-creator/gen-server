import asyncio
from typing import Dict, Any, List
from gen_server.executor.workflows import flux_train_workflow
# from gen_server.base_types.custom_node import get_custom_nodes
from gen_server.config import init_config
from gen_server.globals import update_custom_nodes
from gen_server.utils.extension_loader import load_extensions
import time
from gen_server.base_types.custom_node import custom_node_validator
from gen_server.utils.cli_helpers import parse_known_args_wrapper



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

    async for update in flux_train_workflow(task_data, None):
        print(update)


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
    parser.add_argument("command", choices=["train", "generate"], help="Command to run")
    
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
    parser.add_argument("--prompt", help="Prompt for image generation")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")

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
    # elif args.command == "generate":
    #     asyncio.run(test_generate_image(
    #         prompt=args.prompt,
    #         num_images=args.num_images
    #     ))
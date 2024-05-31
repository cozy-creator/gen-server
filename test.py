import json
import time
from typing import Dict, Optional

import safetensors.torch
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint
from transformers import CLIPTextModel, CLIPTokenizer


def slice_pipeline_safetensors(safetensors_path: str, target_component: str) -> Optional[Dict[str, torch.Tensor]]:
    start_time = time.time()
    """
    Slices a pipeline safetensors file into component state dictionaries.

    Args:
        safetensors_path (str): Path to the pipeline safetensors file.
        target_component (str): Name of the target component 
                                (e.g., "unet", "vae", "text_encoder", "text_encoder_2").

    Returns:
        Optional[Dict[str, torch.Tensor]]: A dictionary containing the state dictionary 
                        for the target component, or None if not found.
    """

    pipeline_state_dict = safetensors.torch.load_file(safetensors_path)
    component_state_dict: Dict[str, torch.Tensor] = {}

    # Key prefixes for different components (with potential variations for SDXL)
    key_prefixes = {
        "unet": ["model.diffusion_model."],
        "vae": ["first_stage_model."],
        "text_encoder": ["cond_stage_model.transformer.", "conditioner.embedders.0.transformer."],
        "text_encoder_2": ["conditioner.embedders.1.model."],
    }

    # Get the relevant key prefixes for the target component
    prefixes = key_prefixes[target_component]

    for key in pipeline_state_dict:
        for prefix in prefixes:
            if key.startswith(prefix):
                # Remove prefix and store in the component state dict
                component_key = key
                component_state_dict[component_key] = pipeline_state_dict[key]
                break  # Move to the next key if a match is found

    end_time = time.time()
    print(f"slice_pipeline_safetensors took {end_time - start_time:.2f} seconds")
    return component_state_dict if component_state_dict else None


def load_unet(safetensors_file: str, config_file: str, device: str = "cpu") -> UNet2DConditionModel:
    start_time = time.time()
    unet_state_dict = slice_pipeline_safetensors(safetensors_file, "unet")
    config = json.load(open(config_file))
    new_unet_state_dict = convert_ldm_unet_checkpoint(unet_state_dict, config=config)

    # unet_config = UNet2DConditionModel.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", subfolder="unet"
    # ).config
    unet = UNet2DConditionModel.from_config(config)
    unet.load_state_dict(new_unet_state_dict, strict=False)
    unet.to(device)
    end_time = time.time()
    print(f"load_unet took {end_time - start_time:.2f} seconds")
    return unet


def load_vae(safetensors_file: str, config_file: str, device: str = "cpu") -> AutoencoderKL:
    start_time = time.time()
    vae_state_dict = slice_pipeline_safetensors(safetensors_file, "vae")
    config = json.load(open(config_file))
    new_vae_state_dict = convert_ldm_vae_checkpoint(vae_state_dict, config=config)
    
    vae = AutoencoderKL.from_config(config)
    # vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    vae.load_state_dict(new_vae_state_dict)
    vae.to(device)
    end_time = time.time()
    print(f"load_vae took {end_time - start_time:.2f} seconds")
    return vae


def load_text_encoder(safetensors_file: str, config_file: str, device: str = "cpu") -> CLIPTextModel:
    start_time = time.time()
    text_encoder_state_dict = slice_pipeline_safetensors(safetensors_file, "text_encoder")
    config = json.load(open(config_file))
    
    CLIPTextModel._from_config(config)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder.load_state_dict(text_encoder_state_dict, strict=False)
    text_encoder.to(device)
    end_time = time.time()
    print(f"load_text_encoder took {end_time - start_time:.2f} seconds")
    return text_encoder


def main():
    overall_start_time = time.time()
    # Path to your pipeline safetensors file
    safetensors_file = "./models/meinamix.safetensors"

    start_time = time.time()
    # Load the individual components
    unet = load_unet(safetensors_file, "unet_config.json", device="cuda")
    vae = load_vae(safetensors_file, "vae_config.json", device="cuda")
    text_encoder = load_text_encoder(safetensors_file, "text_encoder_config.json", device="cuda")
    end_time = time.time()
    print(f"Loading individual components took {end_time - start_time:.2f} seconds")

    start_time = time.time()
    # Instantiate other required components
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    # safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    end_time = time.time()
    print(f"Instantiating other components took {end_time - start_time:.2f} seconds")

    start_time = time.time()
    # Create the StableDiffusionPipeline
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,  # You can load this if you need the safety checker
    )
    end_time = time.time()
    print(f"Creating the StableDiffusionPipeline took {end_time - start_time:.2f} seconds")

    start_time = time.time()
    # Generate images!
    prompt = "Beautiful anime woman, masterpiece, detailed"
    image = pipe(prompt, num_inference_steps=25, negative_prompt="poor quality, worst quality, text, watermark").images[0]
    image.save("generated_image.png")
    end_time = time.time()
    print(f"Generating the image took {end_time - start_time:.2f} seconds")

    overall_end_time = time.time()
    print(f"Overall time taken: {overall_end_time - overall_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
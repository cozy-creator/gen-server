import json
import time

from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline

from transformers import CLIPTokenizer
import logging

from _helpers import (
    ModelRegistry,
    load_unet,
    load_vae,
    load_text_encoder,
    load_text_encoder_2,
)
from gen_server.utils.device import get_torch_device

logger = logging.getLogger(__name__)


def main():
    device = get_torch_device()
    overall_start_time = time.time()

    # Path to your pipeline safetensors file
    safetensors_file = "./models/sd_xl_base_1.0.safetensors"
    # safetensors_file = "./models/v1-5-pruned-emaonly.safetensors"

    # Initialize the model registry
    registry = ModelRegistry()

    # registry.register_from_folder(folders["custom_architecture"])

    # Load components based on detected architecture
    components = registry.load_model(safetensors_file)

    print("Loading Model:", components["arch_id"])

    arch_id = components["arch_id"]

    # Load common components
    components["vae"] = load_vae(
        components["vae"], json.load(open(components["vae_config"])), device=device
    )
    components["text_encoder"] = load_text_encoder(
        components["text_encoder"],
        json.load(open(components["text_config"])),
        device=device,
    )
    components["tokenizer"] = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    components["unet"] = load_unet(
        components["unet"], json.load(open(components["unet_config"])), device=device
    )
    components["scheduler"] = DDIMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )

    # Delete unnecessary keys
    del components["vae_config"]
    del components["text_config"]
    del components["unet_config"]
    del components["arch_id"]
    try:
        del components["text2_config"]
    except:
        pass

    # Create pipeline based on detected architecture
    if arch_id == "sdxl":
        components["text_encoder_2"] = load_text_encoder_2(
            components["text_encoder_2"],
            json.load(open("sdxl_text2_config.json")),
            device=device,
            has_projection=True,
        )
        components["tokenizer_2"] = CLIPTokenizer.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        )

        pipe = StableDiffusionXLPipeline(**components)
    elif arch_id == "sd1.5":
        pipe = StableDiffusionPipeline(**components)
    # else:
    #     raise ValueError(f"Unsupported architecture: {arch_id}")

    # Generate images!
    prompt = "An ultrarealistic dog on the wall, with a surrealistic background"
    image = pipe(prompt, num_inference_steps=25).images[0]
    image.save("generated_image.png")

    overall_end_time = time.time()
    print(f"Overall time taken: {overall_end_time - overall_start_time:.2f} seconds")


if __name__ == "__main__":
    main()

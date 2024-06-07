import os
import sys
import json
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTokenizer
import time
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from .cli_args import args
# from .common.firebase import initialize
# from .settings import settings
from .extension_loader import API_ENDPOINTS, CUSTOM_NODES, WIDGETS, load_extensions
from gen_server.arch_registry import load_models, ArchDefinition

file_path = os.path.join(os.path.dirname(__file__), "../../../../models/meinamix.safetensors")
output_folder = os.path.join(os.path.dirname(__file__), "../../../../output")

# This can be imported globally
ARCHITECTURES: dict[str, ArchDefinition] = {}


def main():
    print(API_ENDPOINTS)
    print(CUSTOM_NODES)
    print(WIDGETS)
    
    # we load the extensions inside a function to avoid circular dependencies
    ARCHITECTURES = load_extensions('comfy_creator.architectures', expected_type=ArchDefinition)

    models = load_models.from_file(file_path, 'cpu', ARCHITECTURES)
    
    print("Number of items loaded:", len(models))
    
    for model_key in models.keys():
        print(f"Model key: {model_key}")
    
    start = time.time()
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    # print(state_dict)

    # diffusion_pytorch_model.fp16.safetensors
    # playground-v2.5-1024px-aesthetic.fp16.safetensors
    # diffusion_pytorch_model.safetensors
    # darkSushi25D25D_v40.safetensors


    pipe = StableDiffusionPipeline(
        vae=models["core-extension-1.sd1_vae"].model,
        unet=models["core-extension-1.sd1_unet"].model,
        text_encoder=models["core-extension-1.sd1_text_encoder"].model,
        safety_checker=None,
        feature_extractor=None,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    pipe.to("cuda")
    if "xformers" in sys.modules:
        pipe.enable_xformers_memory_efficient_attention()
    if "accelerate" in sys.modules:
        pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

    # Generate images!
    prompt = "beautiful anime woman, detailed, masterpiece, dark skin"
    negative_prompt = "poor quality, worst quality, text, watermark, blurry"
    images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=25, num_images_per_prompt=4).images
    
    for idx, img in enumerate(images):
        img.save(os.path.join(output_folder, f"generated_image_{idx}.png"))
    
    print(f"Image generated in {time.time() - start} seconds")
    
    # if args.run_web_server:
    #     from request_handlers.web_server import start_server

    # if args.run_web_server:
    #     from request_handlers.web_server import start_server
    #
    #     start_server(args.host, args.web_server_port)

    # if args.run_grpc:
    #     from request_handlers.grpc_server import start_server

    #     start_server(args.host, args.grpc_port)


if __name__ == "__main__":
    # initialize(json.loads(settings.firebase.service_account))
    main()

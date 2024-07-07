# # from huggingface_hub import hf_hub_download
# import os
#
# import torch
#
# from diffusers import StableDiffusionXLPipeline
#
# from core_extension_1.common.layerdiffuse.models import TransparentVAEDecoder
# from gen_server.utils import load_state_dict_from_file
#
# models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
#
# transparent_vae = TransparentVAEDecoder.from_pretrained(
#     "madebyollin/sdxl-vae-fp16-fix",

# torch_dtype = torch.float16
# ).to(torch.float16)
#
# transparent_vae.set_transparent_decoder(
#     load_state_dict_from_file(
#         os.path.join(models_dir, "vae_transparent_decoder.safetensors")
#     )
# )
#
# # transparent_vae = TransparentVAEDecoder.from_single_file(
# #     os.path.join(models_dir, "vae_transparent_decoder.safetensors")
# # )
#
# pipeline = StableDiffusionXLPipeline.from_single_file(
#     os.path.join(models_dir, "sd_xl_base_1.0.safetensors"),
#     vae=transparent_vae,
#     torch_dtype=torch.float16,
# )
#
# pipeline.to("mps")
# # pipeline.enable_model_cpu_offload()
#
# print("sss")
# pipeline.load_lora_weights(
#     # os.path.join(models_dir, "layer_xl_transparent_attn.safetensors"),
#     "LayerDiffusion/layerdiffusion-v1",
#     torch_dtype=torch.float16,
#     weight_name="layer_xl_transparent_attn.safetensors",
# )
#
# seed = torch.randint(high=1000000, size=(1,)).item()
# prompt = "a cute corgi"
# negative_prompt = ""
# images = pipeline(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     # generator=torch.Generator(device="mps").manual_seed(seed),
#     num_images_per_prompt=1,
#     return_dict=False,
#     torch_dtype=torch.float16,
# )[0]
#
# print(images[0])
#
# images[0].save("result_sdxl.png")


import os
import torch
from diffusers import StableDiffusionXLPipeline
from gen_server.utils import load_state_dict_from_file

from core_extension_1.common.layerdiffuse.models import (
    TransparentVAEDecoder,
)


models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

# Load the VAE and set it to the correct dtype
transparent_vae = TransparentVAEDecoder.from_pretrained(
    "stabilityai/sdxl-vae", torch_dtype=torch.float16
)

transparent_vae.to(torch.float16)

print("here.... 0")
# # # Load and set the transparent decoder weights
transparent_vae.set_transparent_decoder(
    load_state_dict_from_file(
        os.path.join(models_dir, "vae_transparent_decoder.safetensors")
    )  # Ensure weights are loaded as float16
)

transparent_vae.to(device="mps", dtype=torch.float16)

print("here.... 1")
# Load the Stable Diffusion XL pipeline
pipeline = StableDiffusionXLPipeline.from_single_file(
    os.path.join(models_dir, "sd_xl_base_1.0.safetensors"),
    vae=transparent_vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

print("here.... 2")
# Load the LoRA weights
pipeline.load_lora_weights(
    "LayerDiffusion/layerdiffusion-v1",
    weight_name="layer_xl_transparent_attn.safetensors",
)

# pipeline.load_lora_weights(
#     "rootonchair/diffuser_layerdiffuse",
#     weight_name="diffuser_layer_xl_transparent_attn.safetensors",
# )


print("here.... 3")
pipeline.to(device="mps", dtype=torch.float16)
pipeline.enable_attention_slicing()

print("here.... 4")
seed = torch.randint(high=1000000, size=(1,)).item()
prompt = "a cute corgi"
negative_prompt = ""

print("here.... 5")
# Generate images
images = pipeline(
    prompt=prompt,
    num_inference_steps=5,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    return_dict=False,
    torch_dtype=torch.float16,
)[0]

print(images[0])
images[0].save("result_sdxl.png")

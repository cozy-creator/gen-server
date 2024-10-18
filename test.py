from diffusers import StableDiffusionXLPipeline
import torch
import asyncio

# Create 2 async functions that use the pipeline to generate images

async def generate_image_1():
    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipeline.to("cuda")

    image = pipeline("A man in a suit riding a rainbow unicorn", num_inference_steps=20).images[0]

    image.save("image_1.png")

    print("Image 1 generated")

async def generate_image_2():
    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipeline.to("cuda")

    image = pipeline("A man in a suit riding a rainbow unicorn", num_inference_steps=20).images[0]

    image.save("image_2.png")


# Run the functions

asyncio.run(generate_image_1())
asyncio.run(generate_image_2())


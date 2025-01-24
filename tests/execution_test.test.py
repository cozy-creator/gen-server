import torch
from diffusers import SanaPipeline
import time

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    variant="bf16",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)

start_time = time.time()

prompt = 'a black theme wallpaper with a head silhouette of a cat with two glowing eyes. The sky is dark and the cat is looking at the camera.'
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(43),
)[0]

image[0].save("sana_black_theme_wallpaper_cat_head.png")

end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")

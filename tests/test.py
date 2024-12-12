from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9", text_encoder=None, text_encoder_2=None, variant="fp16", torch_dtype=torch.float16)
pipe2 = StableDiffusionXLPipeline.from_pretrained("playgroundai/playground-v2.5-1024px-aesthetic", variant="fp16", torch_dtype=torch.float16)

pipe.text_encoder = pipe2.text_encoder
pipe.text_encoder_2 = pipe2.text_encoder_2

del pipe2

pipe.to("cuda")

image = pipe(prompt="A beautiful woman with long blonde hair is sitting on a bench in a park.", generator=torch.Generator().manual_seed(0)).images[0]
image.save("example_t2iii.png")


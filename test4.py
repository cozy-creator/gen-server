from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9", custom_pipeline="lpw_stable_diffusion", variant="fp16", torch_dtype=torch.float16).to("cuda")

image = pipe(prompt="A beautiful woman with long blonde hair is sitting on a bench in a park.", height=1024, width=1024, guidance_scale=2.5, seed=0, offload_model=True).images[0]
image.save("example_t2i.png")

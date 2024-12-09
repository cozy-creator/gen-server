from diffusers import DiffusionPipeline
import torch
from huggingface_hub.utils import EntryNotFoundError

pipeline_path = "RunDiffusion/Juggernaut-XL-v9"
custom_pipeline = "lpw_stable_diffusion_xl"

try:
    # Attempt to load the pipeline with the custom pipeline
    pipe = DiffusionPipeline.from_pretrained(
        pipeline_path,
        custom_pipeline=custom_pipeline,
        variant="fp16",
        torch_dtype=torch.float16
    ).to("cuda")
    print("Custom pipeline loaded successfully!")
except EntryNotFoundError as e:
    # Handle the specific error for missing pipeline
    print(f"Custom pipeline '{custom_pipeline}' not found: {e}")
    print("Falling back to the default pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        pipeline_path,
        variant="fp16",
        torch_dtype=torch.float16
    ).to("cuda")
except Exception as e:
    raise e

# Your pipeline is ready to use
print("Pipeline is ready!")

image = pipe(prompt="A beautiful woman with long blonde hair is sitting on a bench in a park.", height=1024, width=1024, guidance_scale=2.5, seed=0, offload_model=True).images[0]
image.save("example_t2i.png")

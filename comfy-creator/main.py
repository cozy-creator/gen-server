from spandrel import ModelLoader
from diffusers import StableDiffusionPipeline, DDIMScheduler
import time
from transformers import CLIPTokenizer




# class ModelLoader(ModelLoader):
#     def __init__(self):
#         super().__init__(registry=MAIN_REGISTRY)

#     def load_from_state_dict(self, state_dict: StateDict) -> dict[str, object]:
#         """
#         Load a model from the given state dict.

#         Throws an `UnsupportedModelError` if the model architecture is not supported.
#         """
#         loaded_components = {}  # Initialize an empty dictionary
#         for arch_support in self.registry.architectures("detection"):  # Iterate through all architectures
#             try:
#                 if arch_support.architecture.detect(state_dict):
#                     loaded_components.update(arch_support.architecture.load(state_dict))  # Update the dictionary with the loaded component
#             except:
#                 pass


start = time.time()
model_loader = ModelLoader()
state_dict = model_loader.load_from_file("darkSushi25D25D_v40.safetensors")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

# print(state_dict)

# diffusion_pytorch_model.fp16.safetensors
# playground-v2.5-1024px-aesthetic.fp16.safetensors
# diffusion_pytorch_model.safetensors
# darkSushi25D25D_v40.safetensors


pipe = StableDiffusionPipeline(
    vae=state_dict['vae'],
    unet=state_dict['unet'],
    text_encoder=state_dict['text_encoder'],
    safety_checker=None,
    feature_extractor=None,
    tokenizer=tokenizer,
    scheduler=scheduler,
)
pipe.to("cuda")

# Generate images!
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=25).images[0]
image.save("generated_image2.png")

print(f"Image generated in {time.time() - start} seconds")


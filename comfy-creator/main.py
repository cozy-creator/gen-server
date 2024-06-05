from spandrel import ModelLoader
from diffusers import StableDiffusionPipeline
import time




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
state_dict = model_loader.load_from_file("playground-v2.5-1024px-aesthetic.fp16.safetensors")

print(state_dict)

# diffusion_pytorch_model.fp16.safetensors
# playground-v2.5-1024px-aesthetic.fp16.safetensors
# diffusion_pytorch_model.safetensors
# darkSushi25D25D_v40.safetensors


# pipe = StableDiffusionPipeline(
#     vae=state_dict['vae'],
#     unet=state_dict['unet'],
#     text_encoder=state_dict['text_encoder'],
#     safety_checker=None,
#     feature_extractor=None,
#     tokenizer=state_dict['tokenizer'],
#     scheduler=state_dict['scheduler']
# )
# # pipe.to("cuda")

# # Generate images!
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt, num_inference_steps=25).images[0]
# image.save("generated_image2.png")

print(f"Image generated in {time.time() - start} seconds")


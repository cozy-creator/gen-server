# import requests
# from typing import List, BinaryIO
# from helper.helper_decorators import convert_image_format
# import os



# class SDAPI:

#     def __init__(self):
#         pass

    
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "positive": ("STRING", {"default": "Worlds colliding", "multiline": True}),
#                 "negative": ("STRING", {"default": "best quality, high quality", "multiline": True}),
#                 "aspect_ratio": (["21:9", "16:9", "5:4", "3:2", "1:1", "2:3", "4:5", "9:16", "9:21"], {"default": "1:1"}),
#                 "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294 }),
#                 "output_format": ([ "png", "jpeg", "webp" ], {"default": "png"}),
#                 "model": (["sd3", "sd3-turbo"],),
#             },

#             "optional": {
#                 "image": ("IMAGE",),  
#                 "strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1.0, "step": 0.01}),
#             }
#         }

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "generate"

#     CATEGORY = "sd3"

#     @convert_image_format
#     def generate(self, positive: str, negative: str, aspect_ratio: List[str], seed, output_format, model, image: BinaryIO = None, strength=None):


#         if model == "sd3-turbo":
#             if image is None:
#                 files = {"none": ''}
#                 data = {
#                     "prompt": positive,
#                     "aspect_ratio": aspect_ratio,
#                     "mode": "text-to-image",
#                     "output_format": output_format,
#                     "seed": seed,
#                     "model": model
#                 }

#             else:
#                 files={
#                         "image": ("image.png", image, 'image/png'), 
#                     }

#                 data = {
#                     "prompt": positive,
#                     "mode": "image-to-image",
#                     "model": model,
#                     "seed": seed,
#                     "strength": strength,
#                     "output_format": output_format,
#                     "image": image
#                 }

#         elif model == "sd3":
#             if image is None:
#                 files = {"none": ''}
#                 data={
#                     "prompt": positive,
#                     "negative_prompt": negative,
#                     "aspect_ratio": aspect_ratio,
#                     "mode": "text-to-image",
#                     "model": model,
#                     "seed": seed,
#                     "output_format": output_format,
#                 }
#             else:

#                 files={
#                         "image": ("image.png", image, 'image/png'), 
#                     }

#                 data = {
#                     "prompt": positive,
#                     "negative_prompt": negative,
#                     "mode": "image-to-image",
#                     "model": model,
#                     "seed": seed,
#                     "strength": strength,
#                     "output_format": output_format,
#                 }
        

#         response = requests.post(
#             f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
#             headers={
#                 "authorization": os.getenv("STABILITY_API_KEY"),
#                 "accept": "image/*"
#             },
#             files=files,
#             data=data,
#         )

#         if response.status_code == 200:

#             return (response.content, )
#         else:
#             print(f"Error: {response.status_code}")
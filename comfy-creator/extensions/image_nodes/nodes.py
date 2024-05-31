import torch
from PIL import Image, ImageOps
import numpy as np

class ImageInvert:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "invert"

    CATEGORY = "image"

    DISPLAY_NAME = "Invert Image Colors"  # Specify the display name here

    def invert(self, image):
        # Convert to PIL Image for easier manipulation
        pil_image = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
        
        # Invert the image
        inverted_image = ImageOps.invert(pil_image)
        
        # Convert back to torch tensor
        inverted_tensor = torch.from_numpy(np.array(inverted_image).astype(np.float32) / 255.0).unsqueeze(0)
        return (inverted_tensor,)
    

class ImageInvert2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "invert"

    CATEGORY = "image"

    DISPLAY_NAME = "Invert Image Colors"  # Specify the display name here

    def invert(self, image):
        # Convert to PIL Image for easier manipulation
        pil_image = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
        
        # Invert the image
        inverted_image = ImageOps.invert(pil_image)
        
        # Convert back to torch tensor
        inverted_tensor = torch.from_numpy(np.array(inverted_image).astype(np.float32) / 255.0).unsqueeze(0)
        return (inverted_tensor,)
    

def get_nodes():
    return {
        "ImageInvert": ImageInvert,  # Map node name to class
        "ImageInvert2": ImageInvert2
    }
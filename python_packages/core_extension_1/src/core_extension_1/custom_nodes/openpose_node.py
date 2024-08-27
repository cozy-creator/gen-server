import torch
from gen_server.base_types import CustomNode
from controlnet_aux import OpenposeDetector 
import numpy as np 
import os
from PIL import Image
import json
from typing import Dict, Union
from torchvision import transforms as T


# class OpenPoseNode(CustomNode):
#     """Detects human poses in an image using OpenPose."""

#     def __init__(self):
#         super().__init__()
#         self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    
#     def __call__(self, image: any) -> Image.Image:
#         """
#         Args:
#             image: Input image tensor (C, H, W) or PIL Image.
#         Returns:
#             A dictionary containing the detected OpenPose keypoints as a NumPy array.
#         """
#         try:
#             if isinstance(image, torch.Tensor):
#                 if image.is_cuda:  # Move to CPU if it's on GPU
#                     image = image.cpu()
#                 if image.dim() == 3:  # Assuming the tensor is in (C, H, W) format
#                     image = image.permute(1, 2, 0).numpy()
#                 else:
#                     image = image.numpy()  # For other tensor shapes, convert directly
#             elif isinstance(image, Image.Image):
#                 image = np.array(image)
#             else:
#                 raise TypeError("Input image must be a torch.Tensor or PIL Image.")
            
#             openpose_image = self.openpose(image)
#             return {"openpose_image": openpose_image}

#         except Exception as e:
#             raise ValueError(f"Error detecting poses with OpenPose: {e}")
    
#     @staticmethod
#     def get_spec():
#         """Returns the node specification."""
#         spec_file = os.path.join(os.path.dirname(__file__), 'openpose_node.json') 
#         with open(spec_file, 'r', encoding='utf-8') as f:
#             spec = json.load(f)
#         return spec

class OpenPoseNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    async def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Dict[str, torch.Tensor]: # type: ignore
        # if isinstance(image, Image.Image):
        #     image = T.ToTensor()(image).unsqueeze(0)
        # elif isinstance(image, torch.Tensor) and image.dim() == 3:
        #     image = image.unsqueeze(0)
        
        openpose_image = self.openpose(image)
        # Save the image to a file
        openpose_image.save("openpose_image.png")
        print("Done")
        return {"openpose_image": openpose_image}

    
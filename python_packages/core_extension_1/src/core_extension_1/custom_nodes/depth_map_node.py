import torch
from gen_server.base_types import CustomNode
import os

from transformers import pipeline
from controlnet_aux import MidasDetector
import cv2
from PIL import Image
import numpy as np
import json

from gen_server.utils.device import get_torch_device
from typing import Dict, Union
from torchvision import transforms as T


# Helper function
# def pil_to_cv2(pil_image: Image.Image):
#     """Convert PIL Image to OpenCV format."""
#     return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# class DepthMapNode(CustomNode):
#     """Estimates a depth map from an image using MiDaS or Depth Anything (lazily loaded)."""

#     def __init__(self):
#         super().__init__()
#         self.DEVICE = get_torch_device()
#         self.midas_detector = None
#         self.depth_anything_model = None

#     def __call__(
#         self, image: torch.Tensor, model_type: str = "depth_anything_v2"
#     ) -> dict[str, Image.Image]:
#         """
#         Args:
#             image: Input image tensor (C, H, W) or PIL Image.
#             model_type: The type of depth estimation model to use ("midas" or "depth_anything_v2").
#         Returns:
#             A dictionary containing the depth map as a PIL Image.
#         """
#         try:
#             if isinstance(image, torch.Tensor):
#                 image = Image.fromarray(
#                     (image * 255).permute(1, 2, 0).numpy().astype(np.uint8)
#                 )
#             elif isinstance(image, np.ndarray):
#                 image = Image.fromarray(image)
#             elif not isinstance(image, Image.Image):
#                 raise TypeError(
#                     "Input image must be a torch.Tensor, np.ndarray or PIL Image."
#                 )

#             if model_type.lower() == "midas":
#                 if self.midas_detector is None:
#                     self.midas_detector = MidasDetector.from_pretrained(
#                         "lllyasviel/ControlNet"
#                     )
#                 depth_map = self.midas_detector(image)
#             elif model_type.lower() == "depth_anything_v2":
#                 if self.depth_anything_model is None:
#                     depth_map = self.extract_depth_map_depth_anything(image)
#             else:
#                 raise ValueError(
#                     "Invalid model_type. Choose either 'midas' or 'depth_anything_v2'."
#                 )

#             return {"depth_map": depth_map}
#         except Exception as e:
#             raise ValueError(f"Error estimating depth map: {e}")

#     def extract_depth_map_depth_anything(self, image: Image.Image) -> Image.Image:
#         """Extracts and processes the depth map using Depth Anything."""
#         # load pipe
#         pipe = pipeline(
#             task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf"
#         )

#         # inference
#         depth = pipe(image)["depth"]

#         return depth

#     @staticmethod
#     def get_spec():
#         """Returns the node specification."""
#         spec_file = os.path.join(os.path.dirname(__file__), "depth_map_node.json")
#         with open(spec_file, "r", encoding="utf-8") as f:
#             spec = json.load(f)
#         return spec


class DepthMapNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")

    async def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Dict[str, torch.Tensor]: # type: ignore
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = T.ToPILImage()(image.squeeze(0))
        
        depth_image = self.depth_estimator(image)["depth"]
        # depth_tensor = torch.from_numpy(depth).unsqueeze(0)
        return {"depth_map": depth_image}


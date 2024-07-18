import torch
from gen_server.base_types import CustomNode
from depth_anything.depth_anything_v2.dpt import DepthAnythingV2
from controlnet_aux import MidasDetector
import cv2
from PIL import Image
import numpy as np
import os
import json

# Helper function 
def pil_to_cv2(pil_image: Image.Image):
    """Convert PIL Image to OpenCV format."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

class DepthMapNode(CustomNode):
    """Estimates a depth map from an image using MiDaS or Depth Anything (lazily loaded)."""

    def __init__(self):
        super().__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.midas_detector = None 
        self.depth_anything_model = None 

    def __call__(self, image: torch.Tensor, model_type: str = "depth_anything_v2") -> dict[str, Image.Image]:
        """
        Args:
            image: Input image tensor (C, H, W) or PIL Image.
            model_type: The type of depth estimation model to use ("midas" or "depth_anything_v2"). 
        Returns:
            A dictionary containing the depth map as a PIL Image.
        """
        try:
            if isinstance(image, torch.Tensor):
                image = Image.fromarray((image * 255).permute(1, 2, 0).numpy().astype(np.uint8))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise TypeError("Input image must be a torch.Tensor, np.ndarray or PIL Image.")

            if model_type.lower() == "midas":
                if self.midas_detector is None:
                    self.midas_detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")
                depth_map = self.midas_detector(image)
            elif model_type.lower() == "depth_anything_v2":
                if self.depth_anything_model is None:
                    model_configs = {
                        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                    }
                    encoder = 'vitl'
                    self.depth_anything_model = DepthAnythingV2(**model_configs[encoder])
                    self.depth_anything_model.load_state_dict(torch.load(f'models/depth_anything_v2_vitl.pth', map_location='cpu'))
                    self.depth_anything_model = self.depth_anything_model.to(self.DEVICE).eval()
                depth_map = self.extract_depth_map_depth_anything(image)
            else:
                raise ValueError("Invalid model_type. Choose either 'midas' or 'depth_anything_v2'.")
            
            return {"depth_map": depth_map}
        except Exception as e:
            raise ValueError(f"Error estimating depth map: {e}")


    def extract_depth_map_depth_anything(self, image: Image.Image) -> Image.Image:
        """Extracts and processes the depth map using Depth Anything."""
        raw_img = pil_to_cv2(image)
        depth = self.depth_anything_model.infer_image(raw_img)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = np.uint8(depth_norm)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.imwrite('depth_map.png', depth_norm)
        depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        depth_colormap_pil = Image.fromarray(depth_colormap_rgb)
        return depth_colormap_pil 
    
    @staticmethod
    def get_spec():
        """Returns the node specification."""
        spec_file = os.path.join(os.path.dirname(__file__), 'depth_map_node.json') 
        with open(spec_file, 'r', encoding='utf-8') as f:
            spec = json.load(f)
        return spec
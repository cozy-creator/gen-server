import torch 
from gen_server.base_types import CustomNode
from core_extension_1.common.utils import load_image_from_url
import numpy as np
import os
import json
from diffusers.utils.loading_utils import load_image

class ImageNode(CustomNode):
    """Loads an image from a file path or URL."""

    def __call__(self, file_path: str) -> dict[str, torch.Tensor]:
        """
        Args:
            file_path: Path or URL of the image file.
        Returns:
            A dictionary containing the loaded image tensor.
        """
        try:
            image_pil = load_image_from_url(file_path) if 'http' in file_path else load_image(file_path)
            image_tensor = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1) / 255.0 
            return {"image": image_tensor}
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
    
    @staticmethod
    def get_spec():
        """Returns the node specification."""
        spec_file = os.path.join(os.path.dirname(__file__), 'image_node.json') 
        with open(spec_file, 'r', encoding='utf-8') as f:
            spec = json.load(f)
        return spec
    
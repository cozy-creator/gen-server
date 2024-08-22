import cv2
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Union
from gen_server.base_types import CustomNode
from controlnet_aux import OpenposeDetector
import os
from pathlib import Path
from gen_server.utils.paths import get_assets_dir
import json

class ControlNetPreprocessorNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        self.assets_dir = get_assets_dir()

    def __call__( # type: ignore
        self,
        image: Union[Image.Image, Path, str],
        preprocessor: str,
        threshold1: int = 100,
        threshold2: int = 200
    ) -> Dict[str, Any]:
        image = self.load_image(image)
        if preprocessor == "canny":
            return self.preprocess_canny(image, threshold1, threshold2)
        elif preprocessor == "openpose":
            return self.preprocess_openpose(image)
        else:
            raise ValueError(f"Unsupported preprocessor: {preprocessor}")

    def load_image(self, image: Union[Image.Image, Path, str]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_absolute():
                # If the path is relative, assume it's in the assets directory
                path = self.assets_dir / path
            if path.is_file():
                return Image.open(path).convert("RGB")
            else:
                raise ValueError(f"Image file not found: {path}")
        else:
            raise ValueError(f"Unsupported image input type: {type(image)}")

    def preprocess_canny(self, image: Image.Image, threshold1: int, threshold2: int) -> Dict[str, Any]:
        print("Preprocessing image with Canny edge detection")
        image_array = np.array(image)
        canny_image = cv2.Canny(image_array, threshold1, threshold2)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        i = Image.fromarray(canny_image)
        i.save("canny_image.png")
        return {"control_image": Image.fromarray(canny_image)}

    def preprocess_openpose(self, image: Image.Image) -> Dict[str, Any]:
        openpose_image = self.openpose_detector(image)
        return {"control_image": openpose_image}

    @staticmethod
    def get_spec(): # type: ignore
        """Returns the node specification."""
        spec_file = os.path.join(os.path.dirname(__file__), "controlnet_preprocessor_node.json")
        with open(spec_file, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return spec

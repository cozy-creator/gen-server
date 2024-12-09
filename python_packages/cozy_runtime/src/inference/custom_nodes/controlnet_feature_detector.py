import cv2
import numpy as np
from PIL import Image
from typing import Any
from cozy_runtime.base_types import CustomNode
from controlnet_aux import OpenposeDetector
import os
from cozy_runtime.utils.paths import get_assets_dir
import json


# TO DO: consider replacing 'feature_type' string with an enum
class ControlNetFeatureDetector(CustomNode):
    def __init__(self):
        super().__init__()
        self.openpose_detector = OpenposeDetector.from_pretrained(
            "lllyasviel/ControlNet"
        )
        self.assets_dir = get_assets_dir()

    def __call__(
        self,
        image: Image.Image,
        feature_type: str,
        threshold1: int = 100,
        threshold2: int = 200,
    ) -> dict[str, Any]:
        if feature_type == "canny":
            return self.preprocess_canny(image, threshold1, threshold2)
        elif feature_type == "openpose":
            return self.preprocess_openpose(image)
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")

    def preprocess_canny(
        self, image: Image.Image, threshold1: int, threshold2: int
    ) -> dict[str, Any]:
        print("Preprocessing image with Canny edge detection")
        image_array = np.array(image)
        canny_image = cv2.Canny(image_array, threshold1, threshold2)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)

        return {"control_image": Image.fromarray(canny_image)}

    def preprocess_openpose(self, image: Image.Image) -> dict[str, Any]:
        openpose_image = self.openpose_detector(image)

        return {"control_image": openpose_image}

    @staticmethod
    def get_spec() -> dict[str, Any]:
        """Returns the node specification."""
        spec_file = os.path.join(os.path.dirname(__file__), "feature_extractor.json")
        with open(spec_file, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return spec

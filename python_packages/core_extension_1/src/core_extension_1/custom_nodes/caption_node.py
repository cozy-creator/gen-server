from gen_server.base_types import CustomNode
from typing import Dict, List
import os

class CustomCaptionNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.captions = {}


    async def __call__(self, 
                       image_paths: List[str], 
                       captions: Dict[str, str] = None) -> Dict[str, Dict[str, str]]:
        """
        Manage custom captions for images.

        Args:
            image_paths (List[str]): List of paths to images.
            captions (Dict[str, str], optional): Dictionary of image paths and their captions.

        Returns:
            Dict[str, Dict[str, str]]: Dictionary of image paths and their associated information.
        """
        result = {}
        
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            if captions and image_path in captions:
                self.captions[image_path] = captions[image_path]
            elif image_path not in self.captions:
                self.captions[image_path] = ""

            result[image_path] = {
                "name": image_name,
                "caption": self.captions[image_path]
            }

        return {"image_captions": result}




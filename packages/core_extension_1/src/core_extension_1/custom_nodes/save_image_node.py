import torch
from gen_server.base_types import CustomNode
from gen_server.utils.file_handler import get_file_handler
from typing import List, Union, Dict, Any
from PIL import Image, PngImagePlugin
import io
import numpy as np
import os
import asyncio
import json

class SaveImageNode(CustomNode):
    """Saves images to the filesystem (local or S3) and returns their URLs."""

    def __call__(self,
                 images: Union[torch.Tensor, List[bytes]],
                 save_workflow_metadata: bool = True,
                 save_temp: bool = False,
                 file_format: str = "webp"
        ) -> dict[str, list[dict[str, Any]]]:
        """
        Args:
            images: A batch of image tensors or a list of byte data.
            save_workflow_metadata: Whether to save workflow metadata in the image header (default: True).
            save_temp: Whether to save the images as temporary files (default: False).
            file_format: The desired image file format (default: "webp"). 
        Returns:
            A dictionary containing a list of dictionaries, each with the URL and is_temp flag 
            for a saved image.
        """
        try:
            if isinstance(images, torch.Tensor):
                # Convert tensor batch to PIL images
                pil_images = [Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)) 
                               for img in images]
            elif isinstance(images, list) and all(isinstance(img, bytes) for img in images):
                # Convert byte data to PIL images
                pil_images = [Image.open(io.BytesIO(img)) for img in images]
            else:
                raise TypeError("Images must be either a torch.Tensor or a list of bytes.")

            # Prepare metadata
            metadata = PngImagePlugin.PngInfo() if save_workflow_metadata else None
            if metadata:
                pass

            # Get the appropriate FileHandler
            file_handler = get_file_handler()

            # Save the images and get URLs
            file_urls = []
            for pil_image in pil_images:
                async def save_and_get_url():
                    # Save the image and get the URL from the FileHandler
                    url = await file_handler.upload_png_files(
                        [pil_image], metadata=metadata, is_temp=save_temp
                    )
                    return url

                loop = asyncio.get_running_loop()
                url = loop.run_until_complete(save_and_get_url())
                
                file_urls.append(url[0]) 

            return {"urls": file_urls}  

        except Exception as e:
            raise ValueError(f"Error saving images: {e}")

    @staticmethod
    def get_spec():
        """Returns the node specification."""
        spec_file = os.path.join(os.path.dirname(__file__), 'save_image_node.json')
        with open(spec_file, 'r', encoding='utf-8') as f:
            spec = json.load(f)
        return spec
import torch
from gen_server.base_types import CustomNode
from gen_server.utils.file_handler import get_file_handler
from typing import List, Union, Tuple
from PIL import Image
import io
import numpy as np
import os
import json
from enum import Enum


class SizeHandling(Enum):
    UNIFORM_SIZE = "uniform_size"
    RESIZE = "resize"
    BATCH_BY_SIZE = "batch_by_size"


class LoadImageNode(CustomNode):
    """Loads images from file IDs or byte data, handling size variations."""

    async def __call__(self, 
                 filenames: list[str | bytes],
                 size_handling: SizeHandling = SizeHandling.BATCH_BY_SIZE,
                 target_size: Tuple[int, int] = (1024, 1024)
        ) -> dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            filenames: A list of file IDs (strings) or binary encoded byte data.
            size_handling: How to handle variations in image sizes:
                - UNIFORM_SIZE: All images must have the same size.
                - RESIZE: Resize all images to a common size (specify target_size).
                - BATCH_BY_SIZE: Create separate batches for each unique image size (default).
        Returns:
            A dictionary containing either:
                - "images": A single batch of image tensors (if size_handling is UNIFORM_SIZE or RESIZE).
                - "image_batches": A list of image tensor batches, grouped by size (if size_handling is BATCH_BY_SIZE).
        """
        try:
            images = []
            for filename in filenames:
                if isinstance(filename, str):
                    # Load from file ID
                    file_handler = get_file_handler()
                    image_bytes = await file_handler.get_file(filename)
                    if image_bytes is None:
                        raise FileNotFoundError(f"File not found: {filename}")
                    image = Image.open(io.BytesIO(image_bytes))
                elif isinstance(filename, bytes):
                    # Load from bytes
                    image = Image.open(io.BytesIO(filename))
                else:
                    raise TypeError("Invalid filename type. Must be string (file ID) or bytes.")

                images.append(image)

            if size_handling == SizeHandling.UNIFORM_SIZE:
                self.check_uniform_size(images)
                image_batch = self.process_images(images)
                return {"images": image_batch}
            elif size_handling == SizeHandling.RESIZE:
                # Get target size from node input
                resized_images = [img.resize(target_size) for img in images]
                image_batch = self.process_images(resized_images)
                return {"images": image_batch}
            elif size_handling == SizeHandling.BATCH_BY_SIZE:
                image_batches = self.batch_images_by_size(images)
                return {"image_batches": image_batches}
            else:
                raise ValueError("Invalid size_handling option.")

        except Exception as e:
            raise ValueError(f"Error loading images: {e}")

    def check_uniform_size(self, images: List[Image.Image]):
        """Checks if all images in the list have the same dimensions."""
        if len(images) <= 1:
            return  # No need to check for single or no images

        first_width, first_height = images[0].size
        for image in images[1:]:
            width, height = image.size
            if width != first_width or height != first_height:
                raise ValueError("Images must have uniform size.")

    def process_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Converts PIL Images to tensors and stacks them into a batch."""
        tensor_images = [torch.from_numpy(np.array(img)).permute(2, 0, 1) / 255.0 for img in images] 
        return torch.stack(tensor_images)

    def batch_images_by_size(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """Groups images by size and creates separate batches."""
        size_batches = {}
        for image in images:
            size = image.size
            if size not in size_batches:
                size_batches[size] = []
            size_batches[size].append(image)

        image_batches = [self.process_images(batch) for batch in size_batches.values()]
        return image_batches
    
    @staticmethod
    def get_spec():
        """Returns the node specification."""
        spec_file = os.path.join(os.path.dirname(__file__), 'load_image_node.json')
        with open(spec_file, 'r', encoding='utf-8') as f:
            spec = json.load(f)
        return spec


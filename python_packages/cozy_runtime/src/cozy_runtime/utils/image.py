import os.path
from typing import Union

import numpy as np
import torch
from PIL import Image

from .device import get_torch_device
from .load_models import from_file
from torchvision.transforms import ToPILImage
from io import BytesIO

model_path = os.path.join(os.path.dirname(__file__), "models", "RealESRGAN_x8.pth")


def image_to_tensor(image: Union[str, Image.Image, bytes]):
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"File '{image}' not found.")
        pil_image = Image.open(image)
    elif isinstance(image, bytes):
        pil_image = Image.open(image)
    else:
        raise TypeError("Input must be a str, PIL.Image.Image, or bytes.")

    return pil_image_to_torch_bgr(pil_image)


def save_image(image: Union[Image.Image, bytes], path, format="PNG"):
    if isinstance(image, bytes):
        image = Image.open(image)
    image.save(path, format)


def pil_image_to_torch_bgr(img: Image) -> torch.Tensor:
    img = np.array(img.convert("RGB"))
    img = img[:, :, ::-1]  # flip RGB to BGR
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img) / 255  # Rescale to [0, 1]
    return torch.from_numpy(img).unsqueeze(0).float()


def torch_bgr_to_pil_image(tensor: torch.Tensor) -> Image:
    if tensor.ndim == 4:
        # If we're given a tensor with a batch dimension, squeeze it out
        # (but only if it's a batch of size 1).
        if tensor.shape[0] != 1:
            raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"
    # TODO: is `tensor.float().cpu()...numpy()` the most efficient idiom?
    arr = tensor.float().cpu().clamp_(0, 1).numpy()  # clamp
    arr = 255.0 * np.moveaxis(arr, 0, 2)  # CHW to HWC, rescale
    arr = arr.round().astype(np.uint8)
    arr = arr[:, :, ::-1]  # flip BGR to RGB
    return Image.fromarray(arr, "RGB")


def tensor_to_pil(tensor: torch.Tensor) -> list[Image.Image]:
    """
    Convert a batch of PyTorch tensors to a list of PIL Images.

    Parameters:
    - tensor: torch.Tensor - The tensor to convert. Assumes the tensor has a batch dimension.

    Returns:
    - list[PIL.Image.Image]: The list of tensors as PIL images.
    """
    # Convert to fp16 and move to CPU
    # ToPILImage Transform does not support bfloat16 or some other formats
    tensor = tensor.to(dtype=torch.float16, device="cpu")

    transform = ToPILImage()
    images = [transform(t) for t in tensor]
    return images


def tensor_to_bytes(tensor: torch.Tensor, format: str = "BMP"):
    tensor = tensor.to(dtype=torch.float16, device="cpu")

    images: list[bytes] = []
    transform = ToPILImage()

    for t in tensor:
        image = transform(t)
        bytes_io = BytesIO()
        image.save(bytes_io, format=format)
        images.append(bytes_io.getvalue())

    return images


def aspect_ratio_to_dimensions(aspect_ratio: str, class_name: str) -> tuple[int, int]:
    aspect_ratio_map = {
        "21/9": {"small": (896, 384), "default": (1536, 640)},
        "16/9": {"small": (768, 448), "default": (1344, 768)},
        "4/3": {"small": (704, 512), "default": (1152, 896)},
        "1/1": {"small": (512, 512), "default": (1024, 1024)},
        "3/4": {"small": (512, 704), "default": (896, 1152)},
        "9/16": {"small": (448, 768), "default": (768, 1344)},
        "9/21": {"small": (384, 896), "default": (640, 1536)},
    }

    if aspect_ratio not in aspect_ratio_map:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")

    small_models = {
        "StableDiffusionPipeline",
        "StableDiffusionInpaintPipeline",
    }

    size = "small" if class_name in small_models else "default"

    return aspect_ratio_map[aspect_ratio][size]

import os.path
from typing import Union

import numpy as np
import torch
from PIL import Image

from gen_server.base_types.architecture import architecture_validator
from gen_server.globals import ARCHITECTURES
from gen_server.utils import load_extensions
from gen_server.utils.load_models import from_file

model_path = os.path.join(os.path.dirname(__file__), "models", "RealESRGAN_x8.pth")

ARCHITECTURES.update(
    load_extensions("comfy_creator.architectures", validator=architecture_validator)
)


def image_to_tensor(image: Union[str, Image, bytes]):
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"File '{image}' not found.")
        pil_image = Image.open(image)
    elif isinstance(image, bytes):
        pil_image = Image.open(image)
    # elif isinstance(image, Image):
    #     pil_image = image
    else:
        raise TypeError("Input must be a str, PIL.Image.Image, or bytes.")

    return pil_image_to_torch_bgr(pil_image)


def save_image(image: Union[Image, bytes], path, format="PNG"):
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


def upscale_image(
    component_namespace: str,
    image_path: str,
    model_path: str,
    output_path: str,
):
    components = from_file(model_path, device="mps")
    component = components.get(component_namespace)
    if component is None:
        raise TypeError(f"Component '{component_namespace}' not found in model.")

    input_tensor = image_to_tensor(image_path)
    with torch.no_grad():
        output_tensor = component.model(input_tensor.to("mps"))

        output_img = torch_bgr_to_pil_image(output_tensor)
        save_image(output_img, output_path)


def upscale(model, image: torch.Tensor, device):
    if image.device != torch.device(device):
        image = image.to(device)

    # TODO: Check if the model is already on the correct device
    with torch.no_grad():
        return model(image)


def remove_background(model, image: torch.Tensor, device):
    if image.device != torch.device(device):
        image = image.to(device)

    with torch.no_grad():
        return model(image)
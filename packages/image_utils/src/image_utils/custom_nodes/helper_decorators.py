import inspect
from typing import BinaryIO, Dict, List, Optional, Tuple, Union
import torch
import io
from PIL import Image, ImageOps, ImageSequence
import numpy as np
from aiohttp import payload




# ---------------------------------------- Convert image to tensor decorator -------------------------------------- #
def convert_image_format(func):
    """
    Decorator to convert the input data to the expected format based on the function signature.
    The decorator supports conversion between PyTorch tensors and byte strings.
    """

    # Get the type annotations for the function parameters
    signature = inspect.signature(func)
    parameter_types = {
        name: param.annotation
        for name, param in signature.parameters.items()
        if param.annotation != inspect.Parameter.empty
    }

    def wrapper(self, *args, **kwargs):
        # Get the bound arguments of the function
        bound_args = signature.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        # Convert the input data to the expected format
        for name, value in bound_args.arguments.items():
            expected_type = parameter_types.get(name, None)
            if expected_type == torch.Tensor:
                if isinstance(value, bytes):
                    bound_args.arguments[name] = bytes_to_tensor(value)
                    print(f"Byte - Tensor:")
            elif expected_type == Union[bytes, List[bytes]]:
                print("I'm HERE\n\n\nI'm Here\n\n")
                if isinstance(value, torch.Tensor):
                    bound_args.arguments[name] = tensor_to_bytes(value)
                    print(f"Tensor - Byte:")
                elif isinstance(value, Image):
                    bound_args.arguments[name] = pil_to_bytes(value)
                    print(f"PIL - Byte:")
            elif expected_type == BinaryIO:
                if isinstance(value, torch.Tensor):
                    bound_args.arguments[name] = tensor_to_binaryio(value)
                    print(f"Tensor - BinaryIO:")
            elif expected_type == str:
                if isinstance(value, torch.Tensor):
                    bound_args

        # Call the wrapped function with the converted arguments
        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper



def bytes_to_tensor(byte_string):
    """
    Converts a byte string representing an image to a PyTorch tensor.
    """
    byte_data = io.BytesIO(byte_string)
    img = Image.open(byte_data)

    output_images = []

    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None, :]

        output_images.append(image)

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
    else:
        output_image = output_images[0]

    return output_image

def pil_to_bytes(image: Image, format: str = 'PNG') -> bytes:
    """
    Convert a PIL Image to a byte array.

    Parameters:
    - image: PIL.Image.Image - The image to convert.
    - format: str - The format to save the image in (default is 'PNG').

    Returns:
    - bytes: The image as a byte array.
    """
    with io.BytesIO() as output:
        image.save(output, format=format)
        return output.getvalue()

def tensor_to_bytes(tensor):
    """
    Converts a PyTorch tensor representing an RGB image or batch of images to a list of byte strings.

    Args:
        tensor (torch.Tensor): A PyTorch tensor with shape (batch_size, channels, height, width) or (channels, height, width).

    Returns:
        List[bytes]: A list of byte strings representing the input tensor(s) as image(s).
    """
    # Check if the tensor has a batch dimension
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # Add batch dimension if absent

    if tensor.ndim != 4:
        raise ValueError("Input tensor must have 3 or 4 dimensions.")

    byte_strings = []
    for image_tensor in tensor:
        # Convert tensor to NumPy array and clip pixel values to [0, 255]
        i = 255.0 * image_tensor.cpu().numpy()
        i = np.clip(i, 0, 255).astype(np.uint8)

        # Convert NumPy array to PIL Image and then to byte string
        img = Image.fromarray(i)
        byte_io = io.BytesIO()
        img.save(byte_io, format='PNG')
        byte_strings.append(byte_io.getvalue())

    return byte_strings


def tensor_to_binaryio(tensor):
    """
    Converts a PyTorch tensor representing an RGB image to a BinaryIO.

    Args:
        tensor (torch.Tensor): A PyTorch tensor with shape (channels, height, width) or (batch_size, channels, height, width).

    Returns:
        BinaryIO: A BinaryIO representing the input tensor as an image.
    """
    # Check if the tensor has a batch dimension
    if tensor.ndim == 4:
        # Squeeze the batch dimension if it's 1
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        else:
            raise ValueError("The input tensor should have a batch size of 1 or no batch dimension.")

    # Convert tensor to NumPy array and clip pixel values to [0, 255]
    i = 255.0 * tensor.cpu().numpy()
    i = np.clip(i, 0, 255).astype(np.uint8)

    # Convert NumPy array to PIL Image and then to byte string
    img = Image.fromarray(i)
    byte_io = io.BytesIO()
    img.save(byte_io, format='PNG')

    byte_io.seek(0)

    return byte_io



class Convert:
    def __init__(self):
        pass

    def pt_to_bytes(self, tensor):
        return tensor_to_bytes(tensor)
    



def working_with_bytes(tensor: torch.Tensor):
    convert = Convert()
    # tensor = torch.rand(3, 224, 224)
    byte_string = convert.pt_to_bytes(tensor)
    print(byte_string)
    tensor = bytes_to_tensor(byte_string)
    print(tensor)
    print(tensor.shape)
import os
import numpy as np
from PIL import Image
from skimage import io

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from gen_server.utils.image import remove_background
from gen_server.utils.load_models import from_file


def preprocess_image(image_path) -> (torch.Tensor, list):
    input_size = [1024, 1024]
    im = io.imread(image_path)
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_shp = im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.upsample(
        torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear"
    ).type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)

    return normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]), im_shp


def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array


# def postprocess_image(output, output_path, shape):
#     output = torch.squeeze(F.upsample(output, shape, mode="bilinear"), 0)
#     # output = F.upsample(output, shape, mode="bilinear")
#     ma = torch.max(output)
#     mi = torch.min(output)
#     output = (output - mi) / (ma - mi)
#     permuted = (output * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
#
#     if output.ndim == 3:
#         permuted = torch.unsqueeze(torch.from_numpy(permuted).float(), 0)
#     io.imsave(os.path.join(output_path), permuted)


def rmbg(image, model_path):
    ns = "core_extension_1.isnet"
    components = from_file(model_path, device="mps")
    component = components.get(ns)
    if component is None:
        raise TypeError(f"Component '{ns}' not found in model.")

    (output) = remove_background(component.model, image.bfloat16(), device="mps")
    return output[0][0]


if __name__ == "__main__":
    image_name = "rabbit.png"
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "isnet.pth")
    input_path = os.path.join(os.path.dirname(__file__), "..", "inputs", image_name)

    output_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "outputs",
        f"{image_name.split('.')[0]}_isnet_output.png",
    )

    image, shape = preprocess_image(input_path)
    output = rmbg(image, model_path)
    img = postprocess_image(output.float(), shape)
    Image.fromarray(img).save(output_path)

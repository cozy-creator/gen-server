import os.path

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from skimage import io

from gen_server.base_types.architecture import architecture_validator
from gen_server.globals import ARCHITECTURES
from gen_server.utils import load_extensions
from gen_server.utils.image import (
    remove_background,
)
from gen_server.utils.load_models import from_file

model_path = os.path.join(os.path.dirname(__file__), "models", "RealESRGAN_x8.pth")

ARCHITECTURES.update(
    load_extensions("comfy_creator.architectures", validator=architecture_validator)
)


def test_rmbg(image, model_path):
    ns = "core_extension_1.briarmbg"
    components = from_file(model_path, device="mps")
    component = components.get(ns)
    if component is None:
        raise TypeError(f"Component '{ns}' not found in model.")

    (output) = remove_background(component.model, image.bfloat16(), device="mps")
    return output[0][0]


def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array


def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(
        torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear"
    ).type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image


if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "models", "BriaRMBG-1.4.pth")

    image_name = "robot_arm.jpg"
    input_path = os.path.join(os.path.dirname(__file__), "inputs", image_name)
    # image = image_to_tensor(input_path)

    output_path = os.path.join(
        os.path.dirname(__file__),
        "outputs",
        f"{image_name.split('.')[0]}_rmbg_output.png",
    )

    model_input_size = [1024, 1024]
    orig_im = io.imread(input_path)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to("mps")

    output = test_rmbg(image, model_path)

    # inference
    # result=net(image)

    # post process
    result_image = postprocess_image(output.float(), orig_im_size)

    # save result
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(input_path)
    no_bg_image.paste(orig_image, mask=pil_im)
    no_bg_image.save(output_path)

    # model_input_size = [1024, 1024]
    # orig_im = io.imread(input_path)
    # orig_im_size = orig_im.shape[0:2]
    # image = preprocess_image(orig_im, model_input_size).to("mps")

    print(output)
    #
    # image2 = Image.fromarray(postprocess_image(output.float(), *image.shape[0:2]))
    # no_bg_image = Image.new("RGBA", image2.size, (0, 0, 0, 0))
    # orig_image = Image.open(image_im)
    # no_bg_image.paste(orig_image, mask=image2)
    # save_image(no_bg_image, output_path)

    # component_namespace = "core_extension_1.spandrel_architectures:ESRGANArch"
    # image_name = "robot_arm.jpg"
    #
    # input_path = os.path.join(os.path.dirname(__file__), "inputs", image_name)
    # output_path = os.path.join(
    #     os.path.dirname(__file__), "outputs", f"{image_name.split('.')[0]}_output.png"
    # )
    #
    # input = image_to_tensor(input_path)
    # upscale_image(component_namespace, input_path, model_path, output_path)

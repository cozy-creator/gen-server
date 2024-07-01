import os.path

from gen_server.base_types.architecture import architecture_validator
from gen_server.globals import ARCHITECTURES
from gen_server.utils import load_extensions
from gen_server.utils.image_upscaler import upscale_image

model_path = os.path.join(os.path.dirname(__file__), "models", "RealESRGAN_x8.pth")

ARCHITECTURES.update(
    load_extensions("comfy_creator.architectures", validator=architecture_validator)
)


if __name__ == "__main__":
    component_namespace = "core_extension_1.spandrel_architectures:ESRGANArch"
    image_name = "robot_arm.jpg"

    input_path = os.path.join(os.path.dirname(__file__), "inputs", image_name)
    output_path = os.path.join(
        os.path.dirname(__file__), "outputs", f"{image_name.split('.')[0]}_output.png"
    )

    upscale_image(component_namespace, input_path, model_path, output_path)

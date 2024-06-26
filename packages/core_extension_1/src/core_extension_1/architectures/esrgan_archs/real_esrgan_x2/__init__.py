import os

from gen_server.base_types.architecture import SpandrelArchitectureAdapter
from spandrel.architectures.ESRGAN import ESRGANArch

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class RealESRGAN(SpandrelArchitectureAdapter):
    """
    The RealESRGAN_x2 model used for image upscaling
    """

    input_space = "ESRGAN"
    output_space = "ESRGAN"

    def __init__(self):
        super().__init__(ESRGANArch())

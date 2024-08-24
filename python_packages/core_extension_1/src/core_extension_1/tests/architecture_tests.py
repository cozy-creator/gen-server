import os.path
import unittest

from gen_server.base_types.architecture import SpandrelArchitectureAdapter, Architecture
from spandrel.architectures.ESRGAN import ESRGANArch

from gen_server.utils import load_models
from ..architectures.sd3_archs import (
    SD3UNet,
)


class ArchitectureTests(unittest.TestCase):
    def test_load_architecture(self):
        arch = SD3UNet()
        self.assertTrue(
            isinstance(arch, Architecture),
            "Invalid architecture instance",
        )

        self.assertIsNotNone(
            arch.model,
            "Architecture should have a model after initialization",
        )

    def test_spandrel_architecture_type(self):
        arch = SpandrelArchitectureAdapter(ESRGANArch())

        self.assertTrue(
            isinstance(arch, Architecture),
            "SpandrelArchitectureAdapter should be an instance of Architecture",
        )

    def test_load_spandrel_architecture(self):
        arch = SpandrelArchitectureAdapter(ESRGANArch())

        state_dict = load_models.load_state_dict_from_file(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "RealESRGAN_x2.pth"
            )
        )

        self.assertIsNone(
            arch.model,
            "SpandrelArchitectureAdapter model should be empty before loading",
        )

        arch.load(state_dict, None)

        self.assertIsNotNone(
            arch.model, "SpandrelArchitectureAdapter should have a model after loading"
        )


if __name__ == "__main__":
    unittest.main()

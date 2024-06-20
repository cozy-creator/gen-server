import json
import os
import time
from typing import override

import torch
from RealESRGAN.rrdbnet_arch import RRDBNet

from gen_server import Architecture, StateDict, TorchDevice

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class RealESRGAN_x2(Architecture[RRDBNet]):
    """
    The RealESRGAN_x2 model used for image upscaling
    """

    display_name = "RealESRGAN_x2"
    input_space = "ESRGAN"
    output_space = "ESRGAN"

    def __init__(self):
        with open(config_path, "r") as file:
            config = json.load(file)
            rrdb_net = RRDBNet(**config)

            super().__init__(
                model=rrdb_net,
                config=config,
            )

    @classmethod
    def detect(cls, state_dict: StateDict) -> bool:
        required_keys = {
            "body.0.rdb1.conv1.weight",
            "conv_first.weight",
            "conv_body.weight",
            "conv_last.weight",
        }

        return all(key in state_dict for key in required_keys)

    @override
    def load(self, state_dict: StateDict, device: TorchDevice = None):
        start = time.time()

        self.model.load_state_dict(state_dict)

        if device is not None:
            self.model.to(device=device)
        self.model.to(torch.bfloat16)

        print(f"RealESRGAN_x2 state dict loaded in {time.time() - start} seconds")

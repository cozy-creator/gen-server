import json
import time
import os
from typing import Optional

from spandrel import ImageModelDescriptor
from typing_extensions import override
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
from gen_server import Architecture, StateDict, TorchDevice
import torch
# from paths import folders

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class SD1UNet(Architecture[UNet2DConditionModel]):
    """
    The Unet for the Stable Diffusion 1 pipeline
    """

    display_name = "SD1 UNet"
    input_space = "SD1"
    output_space = "SD1"

    def __init__(self):
        with open(config_path, "r") as file:
            # Create diffusers class
            config = json.load(file)

        super().__init__(
            model=UNet2DConditionModel(**config),
            config=config,
        )
    
    @override
    @classmethod
    def detect(cls, state_dict: StateDict) -> bool:
        return (
            "model.diffusion_model.input_blocks.0.0.bias" in state_dict
            and "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight"
            in state_dict
        )

    @override
    def load(self, state_dict: StateDict, device=None):
        print("Loading SD1.5 UNet")
        start = time.time()

        # Slice state-dict and convert key keys to canonical
        unet_state_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("model.diffusion_model.")
        }
        new_unet_state_dict = convert_ldm_unet_checkpoint(
            unet_state_dict, config=self.config
        )

        self.model.load_state_dict(new_unet_state_dict)

        if device is not None:
            self.model.to(device=device)
        self.model.to(torch.bfloat16)

        print(f"UNet state dict loaded in {time.time() - start} seconds")
        return ImageModelDescriptor(
            self.model,
            unet_state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[],
            supports_half=True,
            supports_bfloat16=True,
            # scale=1,
            # input_channels=in_nc,
            # output_channels=out_nc,
            # size_requirements=SizeRequirements(
            #     minimum=2,
            #     multiple_of=4 if shuffle_factor else 1,
            # ),
        )

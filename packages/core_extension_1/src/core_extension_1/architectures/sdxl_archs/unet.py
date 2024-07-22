import os
import json
import time
import torch
from typing import Any, Optional
from typing_extensions import override
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
from contextlib import nullcontext
from diffusers.utils.import_utils import is_accelerate_available
import re
import logging

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config_unet.json"
)


class SDXLUNet(Architecture[UNet2DConditionModel]):
    """
    The UNet used for the SDXL pipeline
    """

    def __init__(self, **ignored: Any):
        with open(config_path, "r") as file:
            config = json.load(file)

        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            model = UNet2DConditionModel(**config)

        self._model = model
        self._config = config

        self._display_name = "SDXL UNet"
        self._input_space = "SDXL"
        self._output_space = "SDXL"

    @classmethod
    def detect( # type: ignore
        cls,
        state_dict: StateDict,
        **ignored: Any,
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "model.diffusion_model.input_blocks.0.0.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn1.to_k.weight",
        }

        return (
            ComponentMetadata(
                display_name="SDXL UNet",
                input_space="SDXL",
                output_space="SDXL",
            )
            if all(key in state_dict for key in required_keys)
            else None
        )

    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        print("Loading SDXL UNet")
        start = time.time()

        config = self.config
        unet = self.model

        unet_state_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("model.diffusion_model.")
        }
        new_unet_state_dict = convert_ldm_unet_checkpoint(
            unet_state_dict, config=config
        )

        if is_accelerate_available():
            from diffusers.models.model_loading_utils import load_model_dict_into_meta

            print("Using accelerate")
            unexpected_keys = load_model_dict_into_meta(unet, new_unet_state_dict, dtype=torch.float16)
            if unet._keys_to_ignore_on_load_unexpected is not None:
                for pat in unet._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [
                        k for k in unexpected_keys if re.search(pat, k) is None
                    ]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint were not used when initializing {unet.__name__}: \n {[', '.join(unexpected_keys)]}"
                )
        else:
            unet.load_state_dict(new_unet_state_dict)

        if device is not None:
            unet.to(device=device)

        print(f"UNet state dict loaded in {time.time() - start} seconds")


if __name__ == "__main__":
    start_performance_timer = time.time()
    instances = [SDXLUNet() for _ in range(5)]
    total_time = time.time() - start_performance_timer
    average_time = total_time / 5
    print(f"Average instantiation time for SDXLUNet: {average_time} seconds")

import json
import time
import os
import torch
from typing import Optional, Any
from typing_extensions import override
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
from contextlib import nullcontext
import logging
import re
from diffusers.utils.import_utils import is_accelerate_available
if is_accelerate_available():
    from accelerate import init_empty_weights

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_unet.json")



logger = logging.getLogger(__name__)


class SD1UNet(Architecture[UNet2DConditionModel]):
    """
    The Unet for the Stable Diffusion 1 pipeline
    """

    def __init__(self, **ignored: Any):
        with open(config_path, "r") as file:
            # Create diffusers class
            config = json.load(file)
            ctx = init_empty_weights if is_accelerate_available() else nullcontext
            with ctx():
                self._model = UNet2DConditionModel(**config)

            self._config = config
        
        self._display_name = "SD1 UNet"
        self._input_space = "SD1"
        self._output_space = "SD1"

    @classmethod
    def detect( # type: ignore
        cls,
        state_dict: StateDict,
        **ignored: Any,
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "model.diffusion_model.input_blocks.0.0.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight",
        }

        return (
            ComponentMetadata(
                display_name="SD1 UNet",
                input_space="SD1",
                output_space="SD1",
            )
            if all(key in state_dict for key in required_keys)
            else None
        )

    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        start = time.time()

        unet = self._model

        # Slice state-dict and convert key keys to cannonical
        unet_state_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("model.diffusion_model.")
        }
        new_unet_state_dict = convert_ldm_unet_checkpoint(
            unet_state_dict, config=self._config
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

            unet.to(torch.float16)
        

        print(f"UNet state dict loaded in {time.time() - start} seconds")

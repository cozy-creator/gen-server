import json
import time
from typing import Optional, Any
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.loaders.single_file_utils import (
    convert_sd3_transformer_checkpoint_to_diffusers,
)
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.models.model_loading_utils import load_model_dict_into_meta
import torch
import logging
import re
import os
from contextlib import nullcontext

logger = logging.getLogger(__name__)

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config_unet.json"
)


if is_accelerate_available():
    from accelerate import init_empty_weights


class SD3UNet(Architecture[SD3Transformer2DModel]):
    """
    Architecture definition for the SD3 U-Net model.
    """

    def __init__(self, **ignored: Any):
        with open(config_path, "r") as file:
            # Create diffusers class
            config = json.load(file)
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            model = SD3Transformer2DModel(**config)

        self._model = model
        self._config = config

        self._display_name = "SD3 U-Net"
        self._input_space = "SD3"
        self._output_space = "SD3"

    @classmethod
    def detect( # type: ignore
        cls, state_dict: StateDict, **ignored: Any
    ) -> Optional[ComponentMetadata]:
        """
        Detects whether the given state dictionary matches the SD3 U-Net architecture.
        """
        required_keys = {
            "model.diffusion_model.joint_blocks.0.context_block.attn.proj.bias",
            "model.diffusion_model.joint_blocks.0.context_block.attn.proj.weight",
        }

        return (
            ComponentMetadata(
                display_name="SD3 U-Net",
                input_space="SD3",
                output_space="SD3",
            )
            if all(key in state_dict for key in required_keys)
            else None
        )

    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        """
        Loads the SD3 U-Net model from the given state dictionary.
        """
        print("Loading SD3 U-Net")
        start = time.time()

        unet = self._model
        unet_state_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("model.diffusion_model.")
        }
        new_unet_state_dict = convert_sd3_transformer_checkpoint_to_diffusers(
            unet_state_dict, config=self._config
        )

        if is_accelerate_available():
            print("Using accelerate")
            unexpected_keys = load_model_dict_into_meta(
                unet, new_unet_state_dict, dtype=torch.float16
            )
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
            unet.to(torch.float16)


        print(f"UNet loaded in {time.time() - start:.2f} seconds")

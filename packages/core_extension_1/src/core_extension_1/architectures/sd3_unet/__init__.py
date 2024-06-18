import json
import time
from typing import Dict, Optional
from gen_server import Architecture, StateDict, TorchDevice
from diffusers import SD3Transformer2DModel
from diffusers.loaders.single_file_utils import convert_sd3_transformer_checkpoint_to_diffusers
import safetensors
from diffusers.utils import is_accelerate_available
from diffusers.models.modeling_utils import load_model_dict_into_meta
import torch
import logging
import re
import os
from contextlib import nullcontext

logger = logging.getLogger(__name__)


config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")



if is_accelerate_available():
    from accelerate import init_empty_weights

class SD3UNet(Architecture[SD3Transformer2DModel]):
    """
    Architecture definition for the SD3 U-Net model.
    """

    def __init__(self):
        with open(config_path, 'r') as file:
            # Create diffusers class
            config = json.load(file)
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            model = SD3Transformer2DModel(**config)
        super().__init__(
            model=model,
            config=config,
            input_space="SD3",
            output_space="SD3"
        )

    @classmethod
    def detect(cls, state_dict: StateDict) -> bool:
        """
        Detects whether the given state dictionary matches the SD3 U-Net architecture.
        """
        # Implement logic to check for specific keys in the state dict
        return "model.diffusion_model.joint_blocks.0.context_block.attn.proj.bias" in state_dict and \
               "model.diffusion_model.joint_blocks.0.context_block.attn.proj.weight" in state_dict

    def load(self, state_dict: StateDict, device: TorchDevice = None):
        """
        Loads the SD3 U-Net model from the given state dictionary.
        """
        print("Loading SD3 U-Net")
        start = time.time()

        unet = self.model
        unet_state_dict = {key: state_dict[key] for key in state_dict if key.startswith("model.diffusion_model.")}
        new_unet_state_dict = convert_sd3_transformer_checkpoint_to_diffusers(unet_state_dict, config=self.config)

        if is_accelerate_available():
            print("Using accelerate")
            unexpected_keys = load_model_dict_into_meta(unet, new_unet_state_dict, dtype=torch.float16)
            if unet._keys_to_ignore_on_load_unexpected is not None:
                for pat in unet._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint were not used when initializing {unet.__name__}: \n {[', '.join(unexpected_keys)]}"
                )
        else:
            unet.load_state_dict(new_unet_state_dict)
            unet.to(torch.float16)

        if device is not None:
            unet.to(device)

        print(f"UNet loaded in {time.time() - start:.2f} seconds")
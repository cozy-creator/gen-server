from optparse import Option
import os
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import json
from typing import Optional, Any

from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
import time
import torch
from diffusers.utils.import_utils import is_accelerate_available
from .aura_flow_conversions import convert_ldm_vae_checkpoint
from contextlib import nullcontext
import logging
import re

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aura_vae_config.json"
)

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights



class AuraFlowVAEArch(Architecture[AutoencoderKL]):
    """
    The Variational Auto-Encoder used by AuraFlow models
    """


    def __init__(self, **ignored: Any):

        self._display_name = "AuraFlow VAE"
        self._input_space = "AuraFlow"
        self._output_space = "AuraFlow"

        with open(config_path, "r") as file:
            config = json.load(file)

            # print(f"Metadata: {metadata}")
            ctx = init_empty_weights if is_accelerate_available() else nullcontext
            with ctx():
                vae = AutoencoderKL(**config)

            self._model = vae
            self._config = config

    @classmethod
    def detect( # type: ignore
        cls,
        state_dict: StateDict,
        metadata: dict[str, Any],
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "vae.encoder.conv_in.bias",
        }

        return (
            ComponentMetadata(
                display_name="AuraFlow VAE",
                input_space="AuraFlow",
                output_space="AuraFlow",
            )
            if all(key in state_dict for key in required_keys)
            else None
        )

    def load(
        self, state_dict: StateDict, device: Optional[TorchDevice] = None, **kwargs: Any
    ):
        print("Loading AuraFlow VAE")
        start = time.time()

        vae = self._model

        vae_state_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("vae")
        }

        new_vae_state_dict = convert_ldm_vae_checkpoint(
            vae_state_dict, config=self._config
        )

        # print(self._config)

        # print(new_vae_state_dict.keys())

        if is_accelerate_available():
            from diffusers.models.model_loading_utils import load_model_dict_into_meta

            print("Using accelerate")
            unexpected_keys = load_model_dict_into_meta(vae, new_vae_state_dict)
            if vae._keys_to_ignore_on_load_unexpected is not None:
                for pat in vae._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [
                        k for k in unexpected_keys if re.search(pat, k) is None
                    ]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint were not used when initializing {vae.__name__}: \n {[', '.join(unexpected_keys)]}"
                )
        else:
            vae.load_state_dict(new_vae_state_dict)

        # vae.to(torch.float16)

        print(f"VAE state dict loaded in {time.time() - start} seconds")

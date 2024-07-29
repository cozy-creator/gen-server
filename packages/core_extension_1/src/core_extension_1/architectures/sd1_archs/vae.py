from optparse import Option
import os
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import json
from typing import Optional, Any
from diffusers.loaders.single_file_utils import (
    convert_ldm_vae_checkpoint,
    create_vae_diffusers_config_from_ldm,
)
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
import time
import torch
from diffusers.utils.import_utils import is_accelerate_available
from contextlib import nullcontext
import logging
import re

from gen_server.utils.device import get_torch_device

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights


config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_vae.json")



class SD1VAEArch(Architecture[AutoencoderKL]):
    """
    The Variational Auto-Encoder used by Stable Diffusion models
    """

    def __init__(self, **ignored: Any):
        with open(config_path, "r") as file:
            config = json.load(file)

            # print(f"Metadata: {metadata}")
            ctx = init_empty_weights if is_accelerate_available() else nullcontext
            with ctx():
                vae = AutoencoderKL(**config)

            self._model = vae
            self._config = config

        self._display_name = "SD1 VAE"
        self._input_space = "SD1"
        self._output_space = "SD1"

    @classmethod
    def detect(  # type: ignore
        cls,
        state_dict: StateDict,
        **ignored: Any,
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "first_stage_model.encoder.conv_in.bias",
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"
        }

        return (
            ComponentMetadata(
                display_name="SD1 VAE",
                input_space="SD1",
                output_space="SD1",
            )
            if all(key in state_dict for key in required_keys)
            else None
        )

    def load(
        self, state_dict: StateDict, device: Optional[TorchDevice] = None, **kwargs: Any
    ):
        start = time.time()

        vae = self._model

        vae_state_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("first_stage_model.")
        }

        new_vae_state_dict = convert_ldm_vae_checkpoint(
            vae_state_dict, config=self._config
        )
        # print(self._config)

        # print(new_vae_state_dict.keys())

        if is_accelerate_available():
            from diffusers.models.model_loading_utils import load_model_dict_into_meta

            print("Using accelerate")
            unexpected_keys = load_model_dict_into_meta(vae, new_vae_state_dict, dtype=torch.float16)
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


        print(f"VAE state dict loaded in {time.time() - start} seconds")

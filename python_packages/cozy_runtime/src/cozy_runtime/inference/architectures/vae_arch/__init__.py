from optparse import Option
import os
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import json
from typing import Optional, Any
from diffusers.loaders.single_file_utils import (
    convert_ldm_vae_checkpoint,
    create_vae_diffusers_config_from_ldm,
)
from cozy_runtime import Architecture, StateDict, TorchDevice, ComponentMetadata
import time
import torch
from diffusers.utils.import_utils import is_accelerate_available
from contextlib import nullcontext
import logging
import re

from cozy_runtime.utils.device import get_torch_device

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights


class VAEArch(Architecture[AutoencoderKL]):
    """
    The Variational Auto-Encoder used by Stable Diffusion models
    """

    # The PyTorch model definition for the VAEs for SD1, SDXL, and SD3 are the same,
    # except for the changes specified in their config files.
    @staticmethod
    def _determine_type(metadata: dict[str, Any]) -> tuple[ComponentMetadata, str]:
        result: dict = {}
        architecture = metadata.get("modelspec.architecture", "")
        # print(f"metadata: {metadata}\n\n")
        # print(f"architecture: {architecture}\n\n")

        if architecture == "stable-diffusion-xl-v1-base":
            result: ComponentMetadata = {
                "display_name": "SDXL VAE",
                "input_space": "SDXL",
                "output_space": "SDXL",
            }
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "sdxl_config.json"
            )

            print("In SDXL Base")
        elif metadata == {}:
            result: ComponentMetadata = {
                "display_name": "Playgound VAE",
                "input_space": "Playground",
                "output_space": "Playground",
            }
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "play_config.json"
            )
            print("In play")

        elif "sd_merge_models" in metadata:
            sd_merge_models = json.loads(metadata["sd_merge_models"])
            for model_key, model_data in sd_merge_models.items():
                if "name" in model_data and "PONY" in model_data["name"]:
                    pony_name = model_data["name"]
                    print(pony_name)
                    result: ComponentMetadata = {
                        "display_name": "Pony VAE",
                        "input_space": "PONY",
                        "output_space": "PONY",
                    }
                    config_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "sdxl_config.json"
                    )
                    break

            if result == {}:
                result: ComponentMetadata = {
                    "display_name": "SDXL VAE",
                    "input_space": "SDXL",
                    "output_space": "SDXL",
                }
                config_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "sdxl_config.json"
                )
        else:
            result: ComponentMetadata = {
                "display_name": "SDXL VAE",
                "input_space": "SDXL",
                "output_space": "SDXL",
            }
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "sdxl_config.json"
            )

            print("In SDXL Base")

        return result, config_path

    def __init__(self, metadata: dict[str, Any], **ignored: Any):
        result, config_path = self._determine_type(metadata)
        # print(metadata)
        # print(config_path)
        self._display_name = result["display_name"]
        self._input_space = result["input_space"]
        self._output_space = result["output_space"]

        with open(config_path, "r") as file:
            config = json.load(file)

            # print(f"Metadata: {metadata}")
            ctx = init_empty_weights if is_accelerate_available() else nullcontext
            with ctx():
                vae = AutoencoderKL(**config)

            self._model = vae
            self._config = config

    @classmethod
    def detect(  # type: ignore
        cls,
        state_dict: StateDict,
        metadata: dict[str, Any],
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "first_stage_model.encoder.conv_in.bias",
            "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
        }

        if all(key in state_dict for key in required_keys):
            component_metadata, _ = cls._determine_type(metadata)
            return component_metadata

        return None

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
            unexpected_keys = load_model_dict_into_meta(
                vae, new_vae_state_dict, dtype=torch.float16
            )
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

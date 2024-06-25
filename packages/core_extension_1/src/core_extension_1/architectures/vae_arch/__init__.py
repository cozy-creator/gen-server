from optparse import Option
import os
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import json
from typing import Optional, Any
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
import time
import torch
from diffusers.utils.import_utils import is_accelerate_available
from contextlib import nullcontext

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
        architecture = metadata.get("modelspec.architecture", "")

        if architecture == "stable-diffusion-v3-medium":
            result: ComponentMetadata = {
                "display_name": "SD3 VAE",
                "input_space": "SD3",
                "output_space": "SD3",
            }
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "sd3_config.json"
            )
        elif architecture == "stable-diffusion-xl-v1-base":
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
                "display_name": "SD1 VAE",
                "input_space": "SD1",
                "output_space": "SD1",
            }
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "sd1_config.json"
            )

        return result, config_path

    def __init__(self, metadata: dict[str, Any]):
        result, config_path = self._determine_type(metadata)
        self.display_name = result["display_name"]
        self.input_space = result["input_space"]
        self.output_space = result["output_space"]

        with open(config_path, "r") as file:
            config = json.load(file)
            # print(f"Metadata: {metadata}")
            ctx = init_empty_weights if is_accelerate_available() else nullcontext
            with ctx():
                vae = AutoencoderKL(**config)

            self.model = vae
            self.config = config

    @classmethod
    def detect(
        cls,
        state_dict: StateDict,
        metadata: dict[str, Any],
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "first_stage_model.encoder.conv_in.bias",
            # "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"
        }

        if all(key in state_dict for key in required_keys):
            component_metadata, _ = cls._determine_type(metadata)
            return component_metadata

        return None

    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        print("Loading SD VAE")
        start = time.time()

        vae = self.model

        vae_state_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("first_stage_model.")
        }

        new_vae_state_dict = convert_ldm_vae_checkpoint(
            vae_state_dict, config=self.config
        )
        vae.load_state_dict(new_vae_state_dict)

        if device is not None:
            vae.to(device=device)

        vae.to(torch.bfloat16)

        print(f"VAE state dict loaded in {time.time() - start} seconds")

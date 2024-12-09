from optparse import Option
import os
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import json
from typing import Optional, Any
from diffusers.loaders.single_file_utils import (
    # convert_ldm_vae_checkpoint,
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
from diffusers.loaders.single_file_utils import (
    DIFFUSERS_TO_LDM_MAPPING,
    update_vae_resnet_ldm_to_diffusers,
    update_vae_attentions_ldm_to_diffusers,
    conv_attn_to_linear,
)


logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights


config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sd3_config.json"
)


LDM_VAE_KEY = "vae."


def convert_ldm_vae_checkpoint(checkpoint: StateDict, config: dict[str, Any]):
    # extract state dict for VAE
    # remove the LDM_VAE_KEY prefix from the ldm checkpoint keys so that it is easier to map them to diffusers keys
    vae_state_dict = {}
    keys = list(checkpoint.keys())
    vae_key = LDM_VAE_KEY if any(k.startswith(LDM_VAE_KEY) for k in keys) else ""
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {}
    vae_diffusers_ldm_map = DIFFUSERS_TO_LDM_MAPPING["vae"]
    for diffusers_key, ldm_key in vae_diffusers_ldm_map.items():
        if ldm_key not in vae_state_dict:
            continue
        new_checkpoint[diffusers_key] = vae_state_dict[ldm_key]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len(config["down_block_types"])
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key]
        for layer_id in range(num_down_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [
            key
            for key in down_blocks[i]
            if f"down.{i}" in key and f"down.{i}.downsample" not in key
        ]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"},
        )
        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = (
                vae_state_dict.get(f"encoder.down.{i}.downsample.conv.weight")
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = (
                vae_state_dict.get(f"encoder.down.{i}.downsample.conv.bias")
            )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions,
        new_checkpoint,
        vae_state_dict,
        mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"},
    )

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(config["up_block_types"])
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key]
        for layer_id in range(num_up_blocks)
    }

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key
            for key in up_blocks[block_id]
            if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"},
        )
        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = (
                vae_state_dict[f"decoder.up.{block_id}.upsample.conv.weight"]
            )
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = (
                vae_state_dict[f"decoder.up.{block_id}.upsample.conv.bias"]
            )

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions,
        new_checkpoint,
        vae_state_dict,
        mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"},
    )
    conv_attn_to_linear(new_checkpoint)

    return new_checkpoint


class FluxVAEArch(Architecture[AutoencoderKL]):
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

        self._display_name = "SD3 VAE"
        self._input_space = "SD3"
        self._output_space = "SD3"

    @classmethod
    def detect(  # type: ignore
        cls,
        state_dict: StateDict,
        **ignored: Any,
    ) -> Optional[ComponentMetadata]:
        required_keys = {
            "first_stage_model.encoder.conv_in.bias",
            "text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight",
            "text_encoders.clip_l.transformer.text_model.encoder.layers.11.mlp.fc1.weight",
        }

        return (
            ComponentMetadata(
                display_name="SD3 VAE",
                input_space="SD3",
                output_space="SD3",
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
            key: state_dict[key] for key in state_dict if key.startswith("vae.")
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
                vae, new_vae_state_dict, dtype=torch.bfloat16
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

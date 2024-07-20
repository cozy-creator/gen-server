import json
import time
from typing import Dict, Optional, Any
from gen_server import Architecture, StateDict, TorchDevice, ComponentMetadata
from transformers import CLIPTextModelWithProjection, CLIPTextConfig
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.models.model_loading_utils import load_model_dict_into_meta
import re
import os
import torch
import logging
from contextlib import nullcontext

from gen_server.utils.device import get_torch_device

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights

LDM_OPEN_CLIP_TEXT_PROJECTION_DIM = 1024

SD_2_TEXT_ENCODER_KEYS_TO_IGNORE = [
    "cond_stage_model.model.transformer.resblocks.23.attn.in_proj_bias",
    "cond_stage_model.model.transformer.resblocks.23.attn.in_proj_weight",
    "cond_stage_model.model.transformer.resblocks.23.attn.out_proj.bias",
    "cond_stage_model.model.transformer.resblocks.23.attn.out_proj.weight",
    "cond_stage_model.model.transformer.resblocks.23.ln_1.bias",
    "cond_stage_model.model.transformer.resblocks.23.ln_1.weight",
    "cond_stage_model.model.transformer.resblocks.23.ln_2.bias",
    "cond_stage_model.model.transformer.resblocks.23.ln_2.weight",
    "cond_stage_model.model.transformer.resblocks.23.mlp.c_fc.bias",
    "cond_stage_model.model.transformer.resblocks.23.mlp.c_fc.weight",
    "cond_stage_model.model.transformer.resblocks.23.mlp.c_proj.bias",
    "cond_stage_model.model.transformer.resblocks.23.mlp.c_proj.weight",
    "cond_stage_model.model.text_projection",
]

DIFFUSERS_TO_LDM_MAPPING = {
    "unet": {
        "layers": {
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            "conv_in.weight": "input_blocks.0.0.weight",
            "conv_in.bias": "input_blocks.0.0.bias",
            "conv_norm_out.weight": "out.0.weight",
            "conv_norm_out.bias": "out.0.bias",
            "conv_out.weight": "out.2.weight",
            "conv_out.bias": "out.2.bias",
        },
        "class_embed_type": {
            "class_embedding.linear_1.weight": "label_emb.0.0.weight",
            "class_embedding.linear_1.bias": "label_emb.0.0.bias",
            "class_embedding.linear_2.weight": "label_emb.0.2.weight",
            "class_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
        "addition_embed_type": {
            "add_embedding.linear_1.weight": "label_emb.0.0.weight",
            "add_embedding.linear_1.bias": "label_emb.0.0.bias",
            "add_embedding.linear_2.weight": "label_emb.0.2.weight",
            "add_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
    },
    "controlnet": {
        "layers": {
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            "conv_in.weight": "input_blocks.0.0.weight",
            "conv_in.bias": "input_blocks.0.0.bias",
            "controlnet_cond_embedding.conv_in.weight": "input_hint_block.0.weight",
            "controlnet_cond_embedding.conv_in.bias": "input_hint_block.0.bias",
            "controlnet_cond_embedding.conv_out.weight": "input_hint_block.14.weight",
            "controlnet_cond_embedding.conv_out.bias": "input_hint_block.14.bias",
        },
        "class_embed_type": {
            "class_embedding.linear_1.weight": "label_emb.0.0.weight",
            "class_embedding.linear_1.bias": "label_emb.0.0.bias",
            "class_embedding.linear_2.weight": "label_emb.0.2.weight",
            "class_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
        "addition_embed_type": {
            "add_embedding.linear_1.weight": "label_emb.0.0.weight",
            "add_embedding.linear_1.bias": "label_emb.0.0.bias",
            "add_embedding.linear_2.weight": "label_emb.0.2.weight",
            "add_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
    },
    "vae": {
        "encoder.conv_in.weight": "encoder.conv_in.weight",
        "encoder.conv_in.bias": "encoder.conv_in.bias",
        "encoder.conv_out.weight": "encoder.conv_out.weight",
        "encoder.conv_out.bias": "encoder.conv_out.bias",
        "encoder.conv_norm_out.weight": "encoder.norm_out.weight",
        "encoder.conv_norm_out.bias": "encoder.norm_out.bias",
        "decoder.conv_in.weight": "decoder.conv_in.weight",
        "decoder.conv_in.bias": "decoder.conv_in.bias",
        "decoder.conv_out.weight": "decoder.conv_out.weight",
        "decoder.conv_out.bias": "decoder.conv_out.bias",
        "decoder.conv_norm_out.weight": "decoder.norm_out.weight",
        "decoder.conv_norm_out.bias": "decoder.norm_out.bias",
        "quant_conv.weight": "quant_conv.weight",
        "quant_conv.bias": "quant_conv.bias",
        "post_quant_conv.weight": "post_quant_conv.weight",
        "post_quant_conv.bias": "post_quant_conv.bias",
    },
    "openclip": {
        "layers": {
            "text_model.embeddings.position_embedding.weight": "positional_embedding",
            "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "text_model.embeddings.transformer.text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "text_model.final_layer_norm.weight": "ln_final.weight",
            "text_model.final_layer_norm.bias": "ln_final.bias",
            "text_projection.weight": "text_projection",
        },
        "transformer": {
            "text_model.encoder.layers.": "resblocks.",
            "layer_norm1": "ln_1",
            "layer_norm2": "ln_2",
            ".fc1.": ".c_fc.",
            ".fc2.": ".c_proj.",
            ".self_attn": ".attn",
            "transformer.text_model.final_layer_norm.": "ln_final.",
            "transformer.text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "transformer.text_model.embeddings.position_embedding.weight": "positional_embedding",
        },
    },
}


config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class SD3TextEncoder2(Architecture[CLIPTextModelWithProjection]):
    """
    Architecture definition for the SD3 Text Encoder 2 (CLIP-based).
    """

    display_name = "CLIP Text Encoder With Projection"
    input_space = "SD3"
    output_space = "SD3"

    def __init__(self):
        with open(config_path, "r") as file:
            # Create diffusers class
            config = json.load(file)
        text_encoder_config = CLIPTextConfig.from_dict(config)
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            text_encoder = CLIPTextModelWithProjection(text_encoder_config)

        self.model = text_encoder
        self.config = config

    @classmethod
    def detect(
        cls,
        state_dict: StateDict,
        metadata: dict[str, Any],
    ) -> Optional[ComponentMetadata]:
        """
        Detects whether the given state dictionary matches the SD3 Text Encoder 2 architecture.
        """
        state_key = "text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight"

        return (
            ComponentMetadata(
                display_name=cls.display_name,
                input_space=cls.input_space,
                output_space=cls.output_space,
            )
            if state_key in state_dict
            else None
        )

    def load(self, state_dict: StateDict, device: TorchDevice = None):
        """
        Loads the SD3 Text Encoder 2 model from the given state dictionary.
        """
        print("Loading SD3 Text Encoder 2")
        text_model = self.model
        text_encoder_state_dict = {
            key: state_dict[key]
            for key in state_dict
            if key.startswith("text_encoders.clip_g.")
        }
        text_model_dict = {}
        prefix = "text_encoders.clip_g."
        text_proj_key = prefix + "text_projection"
        text_proj_dim = (
            int(text_encoder_state_dict[text_proj_key].shape[0])
            if text_proj_key in text_encoder_state_dict
            else LDM_OPEN_CLIP_TEXT_PROJECTION_DIM
        )
        text_model_dict["text_model.embeddings.position_ids"] = (
            text_model.text_model.embeddings.get_buffer("position_ids")
        )

        keys = list(text_encoder_state_dict.keys())
        keys_to_ignore = SD_2_TEXT_ENCODER_KEYS_TO_IGNORE

        openclip_diffusers_ldm_map = DIFFUSERS_TO_LDM_MAPPING["openclip"]["layers"]
        for diffusers_key, ldm_key in openclip_diffusers_ldm_map.items():
            ldm_key = prefix + ldm_key
            if ldm_key not in text_encoder_state_dict:
                continue
            if ldm_key in keys_to_ignore:
                continue
            if ldm_key.endswith("text_projection"):
                text_model_dict[diffusers_key] = text_encoder_state_dict[
                    ldm_key
                ].T.contiguous()
            else:
                text_model_dict[diffusers_key] = text_encoder_state_dict[ldm_key]

        for key in keys:
            if key in keys_to_ignore:
                continue

            if not key.startswith(prefix + "transformer."):
                # print(key)
                # print(prefix + "transformer.")
                # print(key.startswith(prefix + "transformer."))
                continue

            diffusers_key = key.replace(prefix + "transformer.", "")
            transformer_diffusers_to_ldm_map = DIFFUSERS_TO_LDM_MAPPING["openclip"][
                "transformer"
            ]
            for new_key, old_key in transformer_diffusers_to_ldm_map.items():
                diffusers_key = (
                    diffusers_key.replace(old_key, new_key)
                    .replace(".in_proj_weight", "")
                    .replace(".in_proj_bias", "")
                )

            if key.endswith(".in_proj_weight"):
                weight_value = text_encoder_state_dict[key]

                text_model_dict[diffusers_key + ".q_proj.weight"] = weight_value[
                    :text_proj_dim, :
                ]
                text_model_dict[diffusers_key + ".k_proj.weight"] = weight_value[
                    text_proj_dim : text_proj_dim * 2, :
                ]
                text_model_dict[diffusers_key + ".v_proj.weight"] = weight_value[
                    text_proj_dim * 2 :, :
                ]

            elif key.endswith(".in_proj_bias"):
                weight_value = text_encoder_state_dict[key]
                text_model_dict[diffusers_key + ".q_proj.bias"] = weight_value[
                    :text_proj_dim
                ]
                text_model_dict[diffusers_key + ".k_proj.bias"] = weight_value[
                    text_proj_dim : text_proj_dim * 2
                ]
                text_model_dict[diffusers_key + ".v_proj.bias"] = weight_value[
                    text_proj_dim * 2 :
                ]
            else:
                text_model_dict[diffusers_key] = text_encoder_state_dict[key]

            if (
                diffusers_key
                == "text_model.embeddings.transformer.text_model.embeddings.token_embedding.weight"
            ):
                text_model_dict["text_model.embeddings.token_embedding.weight"] = (
                    text_encoder_state_dict[key]
                )
                del text_model_dict[diffusers_key]

        if is_accelerate_available():
            # torch_dtype = next(text_model.parameters()).dtype
            unexpected_keys = load_model_dict_into_meta(
                text_model, text_model_dict, dtype=torch.float16
            )
            if text_model._keys_to_ignore_on_load_unexpected is not None:
                for pat in text_model._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [
                        k for k in unexpected_keys if re.search(pat, k) is None
                    ]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint were not used when initializing {text_model.__class__.__name__}: \n {[', '.join(unexpected_keys)]}"
                )

        else:
            if not (
                hasattr(text_model, "embeddings")
                and hasattr(text_model.embeddings.position_ids)
            ):
                text_model_dict.pop("text_model.embeddings.position_ids", None)

            text_model.load_state_dict(text_model_dict)
            text_model.to(torch.float16)

        text_model.to(device=get_torch_device())

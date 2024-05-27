import json
import time
from typing import Dict, Optional

import safetensors.torch
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionXLPipeline as SDXLPipeline
)
from contextlib import nullcontext
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig, CLIPTextModelWithProjection
import re

from diffusers.utils import is_accelerate_available
if is_accelerate_available():
    from accelerate import init_empty_weights

import logging
logger = logging.getLogger(__name__)


LDM_CLIP_PREFIX_TO_REMOVE = ["cond_stage_model.transformer.", "conditioner.embedders.0.transformer."]

# Key prefixes for different components (with potential variations for SDXL)
key_prefixes = {
    "sd1": {
        "unet": ["model.diffusion_model."],
        "vae": ["first_stage_model."],
        "text_encoder": ["cond_stage_model.transformer."],
    },
    "sdxl": {
        "unet": ["model.diffusion_model."],
        "vae": ["first_stage_model."],
        "text_encoder": ["conditioner.embedders.0.transformer."],
        "text_encoder_2": ["conditioner.embedders.1.model."],
    },
}

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


def load_safetensors(safetensors_file: str, model_type: str = "sd1.5", component_name: Optional[str] = None) -> Dict[str, torch.Tensor]:

    
    # Get the relevant key prefixes for the target component
    

    component_state_dict = {}
    
    with safetensors.torch.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        if component_name is not None:
            prefixes = key_prefixes[model_type][component_name]
            for key in f.keys():
                for key_prefix in prefixes:
                    if key.startswith(key_prefix):
                        component_state_dict[key] = f.get_tensor(key)
                        break
        else:
            for key in f.keys():
                component_state_dict[key] = f.get_tensor(key)

    
    return component_state_dict


def load_unet(unet_state_dict: dict, config_file: str, device: str = "cpu", model_type: str ="sd1.5") -> UNet2DConditionModel:

    config = json.load(open(config_file))

    new_unet_state_dict = convert_ldm_unet_checkpoint(unet_state_dict, config=config)
    

    ctx = init_empty_weights if is_accelerate_available() else nullcontext

    with ctx():
        unet = UNet2DConditionModel(**config)


    if is_accelerate_available():
        from diffusers.models.modeling_utils import load_model_dict_into_meta
        print("Using accelerate")
        unexpected_keys = load_model_dict_into_meta(unet, new_unet_state_dict, dtype=None)
        if unet._keys_to_ignore_on_load_unexpected is not None:
            for pat in unet._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint were not used when initializing {unet.__name__}: \n {[', '.join(unexpected_keys)]}"
            )
    else:
        unet.load_state_dict(new_unet_state_dict)

    # safetensors.torch.load_model(unet, "./models/diffusion_pytorch_model.safetensors")
    unet.to(torch.bfloat16)

    return unet


def load_vae(vae_state_dict: dict, config_file: str, device: str = "cpu", model_type: str ="sd1.5") -> AutoencoderKL:

    config = json.load(open(config_file))
    new_vae_state_dict = convert_ldm_vae_checkpoint(vae_state_dict, config=config)

    vae = AutoencoderKL.from_config(config)

    if is_accelerate_available():
        from diffusers.models.modeling_utils import load_model_dict_into_meta
        print("Using accelerate")
        unexpected_keys = load_model_dict_into_meta(vae, new_vae_state_dict)
        if vae._keys_to_ignore_on_load_unexpected is not None:
            for pat in vae._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint were not used when initializing {vae.__name__}: \n {[', '.join(unexpected_keys)]}"
            )
    else:
        vae.load_state_dict(new_vae_state_dict)
    vae.to(torch.bfloat16)

    return vae


def load_text_encoder(text_encoder_state_dict: dict, config_file: str, device: str = "cpu", model_type: str ="sd1.5") -> CLIPTextModel:

    config = json.load(open(config_file))
    text_encoder_config = CLIPTextConfig.from_dict(config)

    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        text_encoder = CLIPTextModel(text_encoder_config)

    remove_prefixes = LDM_CLIP_PREFIX_TO_REMOVE
    keys = list(text_encoder_state_dict.keys())
    text_model_dict = {}

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                diffusers_key = key.replace(prefix, "")
                text_model_dict[diffusers_key] = text_encoder_state_dict[key]

    if is_accelerate_available():
        from diffusers.models.modeling_utils import load_model_dict_into_meta
        torch_dtype = next(text_encoder.parameters()).dtype
        unexpected_keys = load_model_dict_into_meta(text_encoder, text_model_dict, dtype=torch_dtype)
        if text_encoder._keys_to_ignore_on_load_unexpected is not None:
            for pat in text_encoder._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint were not used when initializing {text_encoder.__class__.__name__}: \n {[', '.join(unexpected_keys)]}"
            )

    else:

        if not (hasattr(text_encoder, "embeddings") and hasattr(text_encoder.embeddings.position_ids)):
            text_model_dict.pop("text_model.embeddings.position_ids", None)

        text_encoder.load_state_dict(text_model_dict)
    text_encoder.to(torch.bfloat16)

    return text_encoder



def load_text_encoder_2(text_encoder_state_dict: dict, config_file: str, device: str = "cpu", model_type: str ="sdxl", has_projection: bool = False) -> CLIPTextModel:

    config = json.load(open(config_file))
    text_encoder_config = CLIPTextConfig.from_dict(config)


    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        text_model = CLIPTextModelWithProjection(text_encoder_config) if has_projection else CLIPTextModel(text_encoder_config)

    text_model_dict = {}
    prefix = key_prefixes[model_type]["text_encoder_2"][0]
    text_proj_key = prefix + "text_projection"
    text_proj_dim = (
        int(text_encoder_state_dict[text_proj_key].shape[0]) if text_proj_key in text_encoder_state_dict else LDM_OPEN_CLIP_TEXT_PROJECTION_DIM
    )
    text_model_dict["text_model.embeddings.position_ids"] = text_model.text_model.embeddings.get_buffer("position_ids")

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
            text_model_dict[diffusers_key] = text_encoder_state_dict[ldm_key].T.contiguous()
        else:
            text_model_dict[diffusers_key] = text_encoder_state_dict[ldm_key]

    for key in keys:
        if key in keys_to_ignore:
            continue

        if not key.startswith(prefix + "transformer."):
            continue

        diffusers_key = key.replace(prefix + "transformer.", "")
        transformer_diffusers_to_ldm_map = DIFFUSERS_TO_LDM_MAPPING["openclip"]["transformer"]
        for new_key, old_key in transformer_diffusers_to_ldm_map.items():
            diffusers_key = (
                diffusers_key.replace(old_key, new_key).replace(".in_proj_weight", "").replace(".in_proj_bias", "")
            )

        if key.endswith(".in_proj_weight"):
            weight_value = text_encoder_state_dict[key]

            text_model_dict[diffusers_key + ".q_proj.weight"] = weight_value[:text_proj_dim, :]
            text_model_dict[diffusers_key + ".k_proj.weight"] = weight_value[text_proj_dim : text_proj_dim * 2, :]
            text_model_dict[diffusers_key + ".v_proj.weight"] = weight_value[text_proj_dim * 2 :, :]

        elif key.endswith(".in_proj_bias"):
            weight_value = text_encoder_state_dict[key]
            text_model_dict[diffusers_key + ".q_proj.bias"] = weight_value[:text_proj_dim]
            text_model_dict[diffusers_key + ".k_proj.bias"] = weight_value[text_proj_dim : text_proj_dim * 2]
            text_model_dict[diffusers_key + ".v_proj.bias"] = weight_value[text_proj_dim * 2 :]
        else:
            text_model_dict[diffusers_key] = text_encoder_state_dict[key]

    if is_accelerate_available():
        from diffusers.models.modeling_utils import load_model_dict_into_meta
        torch_dtype = next(text_model.parameters()).dtype
        unexpected_keys = load_model_dict_into_meta(text_model, text_model_dict, dtype=torch_dtype)
        if text_model._keys_to_ignore_on_load_unexpected is not None:
            for pat in text_model._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint were not used when initializing {text_model.__class__.__name__}: \n {[', '.join(unexpected_keys)]}"
            )

    else:
        if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
            text_model_dict.pop("text_model.embeddings.position_ids", None)

        text_model.load_state_dict(text_model_dict)

    if torch_dtype is not None:
        text_model = text_model.to(torch_dtype)

    text_model.to(torch.bfloat16)

    return text_model
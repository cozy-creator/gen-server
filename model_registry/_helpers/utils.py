import time
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)

import torch
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig, CLIPTextModelWithProjection
import logging



logger = logging.getLogger(__name__)


from diffusers.utils import is_accelerate_available
if is_accelerate_available():
    from accelerate import init_empty_weights


LDM_CLIP_PREFIX_TO_REMOVE = ["cond_stage_model.transformer.", "conditioner.embedders.0.transformer."]

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

def load_unet(unet_state_dict: dict, config: dict, device: str = "cpu") -> UNet2DConditionModel:
    start_time = time.time()
    new_unet_state_dict = convert_ldm_unet_checkpoint(unet_state_dict, config=config)
    unet = UNet2DConditionModel(**config)
    unet.load_state_dict(new_unet_state_dict)
    unet.to(device, torch.bfloat16)
    end_time = time.time()
    print(f"load_unet took {end_time - start_time:.2f} seconds")
    return unet


def load_vae(vae_state_dict: dict, config: dict, device: str = "cpu") -> AutoencoderKL:
    start_time = time.time()
    new_vae_state_dict = convert_ldm_vae_checkpoint(vae_state_dict, config=config)
    vae = AutoencoderKL.from_config(config)
    vae.load_state_dict(new_vae_state_dict)
    vae.to(device, torch.bfloat16)
    end_time = time.time()
    
    print(f"load_vae took {end_time - start_time:.2f} seconds")
    return vae


def load_text_encoder(text_encoder_state_dict: dict, config: dict, device: str = "cpu") -> CLIPTextModel:
    start_time = time.time()
    text_encoder_config = CLIPTextConfig.from_dict(config)
    text_encoder = CLIPTextModel(text_encoder_config)

    remove_prefixes = LDM_CLIP_PREFIX_TO_REMOVE
    keys = list(text_encoder_state_dict.keys())
    text_model_dict = {}

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                diffusers_key = key.replace(prefix, "")
                text_model_dict[diffusers_key] = text_encoder_state_dict[key]

    if not (hasattr(text_encoder, "embeddings") and hasattr(text_encoder.embeddings.position_ids)):
        text_model_dict.pop("text_model.embeddings.position_ids", None)

    text_encoder.load_state_dict(text_model_dict, strict=False)
    text_encoder.to(device, torch.bfloat16)
    end_time = time.time()
    print(f"load_text_encoder took {end_time - start_time:.2f} seconds")
    return text_encoder



def load_text_encoder_2(text_encoder_state_dict: dict, config: dict, device: str = "cpu", has_projection: bool = False) -> CLIPTextModel:
    start_time = time.time()
    text_encoder_config = CLIPTextConfig.from_dict(config)
    
    text_model = CLIPTextModelWithProjection(text_encoder_config) if has_projection else CLIPTextModel(text_encoder_config)

    text_model_dict = {}
    prefix = "conditioner.embedders.1.model."
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

    if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
        text_model_dict.pop("text_model.embeddings.position_ids", None)

    text_model.load_state_dict(text_model_dict)
    text_model.to(device, torch.bfloat16)

    end_time = time.time()
    print(f"load_text_encoder_2 took {end_time - start_time:.2f} seconds")

    return text_model
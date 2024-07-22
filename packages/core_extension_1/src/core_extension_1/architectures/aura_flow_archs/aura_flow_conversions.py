from typing import Any
import torch
from diffusers.loaders.single_file_utils import (
    update_vae_resnet_ldm_to_diffusers, 
    update_vae_attentions_ldm_to_diffusers,
    conv_attn_to_linear 
)


LDM_VAE_KEY = "vae."

DIFFUSERS_TO_LDM_MAPPING = {
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

}

# similar to SD3 but only for the last norm layer
def swap_scale_shift(weight: torch.Tensor, dim: int):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def convert_auraflow_transformer_checkpoint_to_diffusers(checkpoint: dict, **kwargs: Any):
    converted_state_dict = {}
    keys = list(checkpoint.keys())
    for k in keys:
        if "model." in k:
            checkpoint[k.replace("model.", "")] = checkpoint.pop(k)


    num_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "single_layers" in k))[-1] + 1  # noqa: C401
    num_layers_joint = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "double_layers" in k))[-1] + 1
    # caption_projection_dim = 1536

    # Positional and patch embeddings.
    converted_state_dict["pos_embed.pos_embed"] = checkpoint.pop("positional_encoding")
    converted_state_dict["pos_embed.proj.weight"] = checkpoint.pop("init_x_linear.weight")
    converted_state_dict["pos_embed.proj.bias"] = checkpoint.pop("init_x_linear.bias")

    # Timestep embeddings.
    converted_state_dict["time_step_proj.linear_1.weight"] = checkpoint.pop(
        "t_embedder.mlp.0.weight"
    )
    converted_state_dict["time_step_proj.linear_1.bias"] = checkpoint.pop("t_embedder.mlp.0.bias")
    converted_state_dict["time_step_proj.linear_2.weight"] = checkpoint.pop(
        "t_embedder.mlp.2.weight"
    )
    converted_state_dict["time_step_proj.linear_2.bias"] = checkpoint.pop("t_embedder.mlp.2.bias")

    # Context projections.
    converted_state_dict["context_embedder.weight"] = checkpoint.pop("cond_seq_linear.weight")


    # Single transformer blocks.
    for i in range(num_layers):
        converted_state_dict[f"single_transformer_blocks.{i}.attn.to_k.weight"] = checkpoint.pop(f"single_layers.{i}.attn.w1k.weight")
        converted_state_dict[f"single_transformer_blocks.{i}.attn.to_out.0.weight"] = checkpoint.pop(f"single_layers.{i}.attn.w1o.weight")
        converted_state_dict[f"single_transformer_blocks.{i}.attn.to_q.weight"] = checkpoint.pop(f"single_layers.{i}.attn.w1q.weight")
        converted_state_dict[f"single_transformer_blocks.{i}.attn.to_v.weight"] = checkpoint.pop(f"single_layers.{i}.attn.w1v.weight")
        converted_state_dict[f"single_transformer_blocks.{i}.ff.linear_1.weight"] = checkpoint.pop(f"single_layers.{i}.mlp.c_fc1.weight")
        converted_state_dict[f"single_transformer_blocks.{i}.ff.linear_2.weight"] = checkpoint.pop(f"single_layers.{i}.mlp.c_fc2.weight")
        converted_state_dict[f"single_transformer_blocks.{i}.ff.out_projection.weight"] = checkpoint.pop(f"single_layers.{i}.mlp.c_proj.weight")
        converted_state_dict[f"single_transformer_blocks.{i}.norm1.linear.weight"] = checkpoint.pop(f"single_layers.{i}.modCX.1.weight")

    # Double transformer blocks.
    for i in range(num_layers_joint):
        converted_state_dict[f"joint_transformer_blocks.{i}.attn.add_k_proj.weight"] = checkpoint.pop(f"double_layers.{i}.attn.w1k.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.attn.to_add_out.weight"] = checkpoint.pop(f"double_layers.{i}.attn.w1o.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.attn.add_q_proj.weight"] = checkpoint.pop(f"double_layers.{i}.attn.w1q.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.attn.add_v_proj.weight"] = checkpoint.pop(f"double_layers.{i}.attn.w1v.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.attn.to_k.weight"] = checkpoint.pop(f"double_layers.{i}.attn.w2k.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.attn.to_out.0.weight"] = checkpoint.pop(f"double_layers.{i}.attn.w2o.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.attn.to_q.weight"] = checkpoint.pop(f"double_layers.{i}.attn.w2q.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.attn.to_v.weight"] = checkpoint.pop(f"double_layers.{i}.attn.w2v.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.ff_context.linear_1.weight"] = checkpoint.pop(f"double_layers.{i}.mlpC.c_fc1.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.ff_context.linear_2.weight"] = checkpoint.pop(f"double_layers.{i}.mlpC.c_fc2.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.ff_context.out_projection.weight"] = checkpoint.pop(f"double_layers.{i}.mlpC.c_proj.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.ff.linear_1.weight"] = checkpoint.pop(f"double_layers.{i}.mlpX.c_fc1.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.ff.linear_2.weight"] = checkpoint.pop(f"double_layers.{i}.mlpX.c_fc2.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.ff.out_projection.weight"] = checkpoint.pop(f"double_layers.{i}.mlpX.c_proj.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.norm1_context.linear.weight"] = checkpoint.pop(f"double_layers.{i}.modC.1.weight")
        converted_state_dict[f"joint_transformer_blocks.{i}.norm1.linear.weight"] = checkpoint.pop(f"double_layers.{i}.modX.1.weight")


    # register tokens
    converted_state_dict["register_tokens"] = checkpoint.pop("register_tokens")


    # Final blocks.
    converted_state_dict["proj_out.weight"] = checkpoint.pop("final_linear.weight")
    # converted_state_dict["norm_out.linear.weight"] = checkpoint.pop("modF.1.weight")

    # converted_state_dict["proj_out.bias"] = checkpoint.pop("final_layer.linear.bias")

    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        checkpoint.pop("modF.1.weight"), dim=None
    )


    return converted_state_dict


def convert_ldm_vae_checkpoint(checkpoint: dict, config: dict):
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
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"},
        )
        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.get(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.get(
                f"encoder.down.{i}.downsample.conv.bias"
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
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(config["up_block_types"])
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"},
        )
        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

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
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )
    conv_attn_to_linear(new_checkpoint)

    return new_checkpoint
import os
from typing_extensions import override
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import json
from typing import Optional
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from gen_server import Architecture, StateDict, TorchDevice
import time
import torch

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class SD3VAE(Architecture[AutoencoderKL]):
    """
    The Variational Auto-Encoder used by Stable Diffusion 3
    """
    display_name = "SD3 VAE"
    input_space = "SD3"
    output_space = "SD3"

    def __init__(self):
        with open(config_path, 'r') as file:
            config = json.load(file)
            vae = AutoencoderKL(**config)

            super().__init__(
                model=vae,
                config=config
            )
    
    @override
    @classmethod
    def detect(cls, state_dict: StateDict) -> bool:
        required_keys = {
            "first_stage_model.encoder.conv_in.bias",
            "text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight"
        }
        
        return all(key in state_dict for key in required_keys)
    
    @override
    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None):
        print("Loading SD3 VAE")
        start = time.time()
        
        vae = self.model

        vae_state_dict = {key: state_dict[key] for key in state_dict if key.startswith("first_stage_model.")}

        new_vae_state_dict = convert_ldm_vae_checkpoint(vae_state_dict, config=self.config)
        vae.load_state_dict(new_vae_state_dict)
        
        if device is not None:
            vae.to(device=device)

        vae.to(torch.bfloat16)
        
        print(f"VAE state dict loaded in {time.time() - start} seconds")

    
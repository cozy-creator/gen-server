import os
from typing import Callable
from diffusers import AutoencoderKL
import json
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from gen_server import ArchDefinition, StateDict, ModelWrapper, TorchDevice
import time

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class SD1VAEArch(ArchDefinition[AutoencoderKL]):
    """_summary_

    Args:
        Architecture (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    @classmethod
    def detect(cls, state_dict: StateDict) -> bool:
        required_keys = {
            "first_stage_model.encoder.conv_in.bias"
        }
        
        return all(key in state_dict for key in required_keys)
    
    @classmethod
    def load(cls, state_dict: StateDict, device: TorchDevice = None) -> ModelWrapper[AutoencoderKL]:
        print("Loading SD1.5 VAE")
        start = time.time()
        
        with open(config_path, 'r') as file:
            config = json.load(file)
            vae = AutoencoderKL(**config)

            vae_state_dict = {key: state_dict[key] for key in state_dict if key.startswith("first_stage_model.")}

            new_vae_state_dict = convert_ldm_vae_checkpoint(vae_state_dict, config=config)
            vae.load_state_dict(new_vae_state_dict)
            
            if device is not None:
                vae.to(device=device)
            
            print(f"VAE state dict loaded in {time.time() - start} seconds")
        
        return ModelWrapper(vae)
    
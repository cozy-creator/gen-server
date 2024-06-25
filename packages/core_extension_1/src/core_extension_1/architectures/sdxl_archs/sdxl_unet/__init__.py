import os
import json
import time
import torch
from typing_extensions import override
from gen_server import Architecture, StateDict, TorchDevice
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
from contextlib import nullcontext
from diffusers.utils.import_utils import is_accelerate_available

if is_accelerate_available():
    from accelerate import init_empty_weights


config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

class SDXLUNet(Architecture[UNet2DConditionModel]):
    """
    The UNet used for the SDXL pipeline
    """
    display_name = "SDXL UNet"
    input_space = "SDXL"
    output_space = "SDXL"

    def __init__(self):
        with open(config_path, 'r') as file:
            config = json.load(file)

        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            model = UNet2DConditionModel(**config)

        super().__init__(
            model=model,
            config=config
        )

    @override
    @classmethod
    def detect(cls, state_dict: StateDict) -> bool:
        required_keys = {
            "model.diffusion_model.input_blocks.0.0.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn1.to_k.weight"
        }
        
        return all(key in state_dict for key in required_keys)
    
    @override
    def load(self, state_dict: StateDict, device: TorchDevice = None):
        print("Loading SDXL UNet")
        start = time.time()

        config = self.config
        unet = self.model

        unet_state_dict = {key: state_dict[key] for key in state_dict if key.startswith("model.diffusion_model.")}
        new_unet_state_dict = convert_ldm_unet_checkpoint(unet_state_dict, config=config)
        unet.load_state_dict(new_unet_state_dict)
        
        if device is not None:
            unet.to(device=device)
        unet.to(torch.bfloat16)

        print(f"UNet state dict loaded in {time.time() - start} seconds")
        

if __name__ == "__main__":
    start_performance_timer = time.time()
    instances = [SDXLUNet() for _ in range(5)]
    total_time = time.time() - start_performance_timer
    average_time = total_time / 5
    print(f"Average instantiation time for SDXLUNet: {average_time} seconds")

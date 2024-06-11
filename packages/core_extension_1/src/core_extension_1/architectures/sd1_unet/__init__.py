import json
import time
import os
from typing_extensions import override
from diffusers import UNet2DConditionModel
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
from gen_server import Architecture, StateDict, TorchDevice
# from paths import folders

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class SD1UNet(Architecture[UNet2DConditionModel]):
    """
    The Unet for the Stable Diffusion 1 pipeline
    """
    def __init__():
        with open(config_path, 'r') as file:
            # Create diffusers class
            config = json.load(file)

            super().__init__(
                model=UNet2DConditionModel(**config),
                config=config,
                input_space="SD1",
                output_space="SD1"
            )
    
    @override
    @classmethod
    def detect(cls, state_dict: StateDict) -> bool:
        return (
            "model.diffusion_model.input_blocks.0.0.bias" in state_dict and
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight" in state_dict
        )
    
    @override
    def load(self, state_dict: StateDict, device: TorchDevice = None):
        print("Loading SD1.5 UNet")
        start = time.time()
        
        unet = self.model
        
        # Slice state-dict and convert key keys to cannonical
        unet_state_dict = {key: state_dict[key] for key in state_dict if key.startswith("model.diffusion_model.")}
        new_unet_state_dict = convert_ldm_unet_checkpoint(unet_state_dict, config=self.config)

        unet.load_state_dict(new_unet_state_dict)
        
        if device is not None:
            unet.to(device=device)
        
        print(f"UNet state dict loaded in {time.time() - start} seconds")
    

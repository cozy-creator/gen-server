import json
import time
import os

from diffusers import UNet2DConditionModel
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint

from spandrel.util import KeyCondition
from gen_server import ArchDefinition, StateDict, ModelWrapper
# from paths import folders

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class SD1UNetArch(ArchDefinition[UNet2DConditionModel]):
    """
    The Unet for the Stable Diffusion 1 pipeline
    """
    
    @classmethod
    def detect(cls, state_dict: StateDict) -> bool:
        return (
            "model.diffusion_model.input_blocks.0.0.bias" in state_dict and
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight" in state_dict
        )
    
    # TO DO: make this return the correct model wrapper
    @classmethod
    def load(cls, state_dict: StateDict) -> ModelWrapper[UNet2DConditionModel]:
        print("Loading SD1.5 UNet")
        start = time.time()
        
        with open(config_path, 'r') as file:
            config = json.load(file)
            unet = UNet2DConditionModel(**config)

            unet_state_dict = {key: state_dict[key] for key in state_dict if key.startswith("model.diffusion_model.")}

            new_unet_state_dict = convert_ldm_unet_checkpoint(unet_state_dict, config=config)
            unet.load_state_dict(new_unet_state_dict)
            print(f"UNet state dict loaded in {time.time() - start} seconds")
            
        return ModelWrapper(model=unet)
    

# MAIN_REGISTRY.add(ArchSupport.from_architecture(UNet()))

# model_loader = ModelLoader()
# state_dict = model_loader.load_from_file("v1-5-pruned-emaonly.safetensors")
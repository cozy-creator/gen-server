from diffusers import UNet2DConditionModel
from spandrel_core import Architecture, StateDict
from spandrel_core.util import KeyCondition
import json
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
import time
from paths import folders

class SD15UNetArch(Architecture[UNet2DConditionModel]):
    def __init__(
            self,
    ) -> None:
        super().__init__(
            id="SD15UNet",
            name="UNet",
            detect=KeyCondition.has_all(
                "model.diffusion_model.input_blocks.0.0.bias",
                "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight"
            ),
        )

    def load(self, state_dict: StateDict) -> UNet2DConditionModel:
        print("Loading SD1.5 UNet")
        start = time.time()
        config = json.load(open(f"{folders['unet']}/sd15_unet_config.json"))
        unet = UNet2DConditionModel(**config)

        unet_state_dict = {key: state_dict[key] for key in state_dict if key.startswith("model.diffusion_model.")}

        new_unet_state_dict = convert_ldm_unet_checkpoint(unet_state_dict, config=config)
        unet.load_state_dict(new_unet_state_dict)
        print(f"UNet state dict loaded in {time.time() - start} seconds")


        return {
            "unet": unet,
            "lineage": "SD1.5"
        }
    

# MAIN_REGISTRY.add(ArchSupport.from_architecture(UNet()))

# model_loader = ModelLoader()
# state_dict = model_loader.load_from_file("v1-5-pruned-emaonly.safetensors")

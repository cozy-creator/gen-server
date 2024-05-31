import torch
from typing import Dict, Callable, List



StateDict = Dict[str, torch.Tensor]

class ModelArchitecture:
    """
    Defines the structure and loading logic for a specific model architecture.
    """

    def __init__(
        self,
        id: str, 
        required_keys: List[str],
        key_converter: Callable[[str], str], 
        decomposer: Callable[[StateDict], Dict[str, StateDict]],
    ):
        self.id = id
        self.required_keys = required_keys
        self.key_converter = key_converter
        self.decomposer = decomposer

    def detect(self, state_dict: StateDict) -> bool:
        """
        Detects if the given state dictionary matches this architecture.
        """
        for key in self.required_keys:
            converted_key = self.key_converter(key)
            if converted_key not in state_dict:
                return False
        return True


    def load(self, state_dict: StateDict) -> Dict[str, StateDict]:
        """
        Loads and decomposes the state dictionary into its components.
        """
        return self.decomposer(state_dict)


# Stable Diffusion 1.5 Architecture
class SD15Arch(ModelArchitecture):
    def __init__(self):
        super().__init__(
            id="sd1.5",
            required_keys=["model.diffusion_model.input_blocks.0.0.weight", 
                           "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"], 
            key_converter=lambda key: key,
            decomposer=self._decompose_sd15,
        )

    def _decompose_sd15(self, state_dict: StateDict) -> Dict[str, StateDict]:
        unet_state_dict = {}
        vae_state_dict = {}
        text_encoder_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("model.diffusion_model."):
                unet_state_dict[key] = value
            elif key.startswith("first_stage_model."):
                vae_state_dict[key] = value
            elif key.startswith("cond_stage_model.transformer."):
                text_encoder_state_dict[key] = value 

        return {
            "unet": unet_state_dict,
            "vae": vae_state_dict,
            "text_encoder": text_encoder_state_dict,
            "text_config": "text_config.json",
            "vae_config": "vae_config.json",
            "unet_config": "unet_config.json",
            "safety_checker": None,
            "feature_extractor": None,
        }

# Stable Diffusion XL Architecture
class SDXLArch(ModelArchitecture):
    def __init__(self):
        super().__init__(
            id="sdxl",
            required_keys=["model.diffusion_model.input_blocks.0.0.weight", 
                           "conditioner.embedders.1.model.token_embedding.weight"], 
            key_converter=lambda key: key, 
            decomposer=self._decompose_sdxl,
        )

    def _decompose_sdxl(self, state_dict: StateDict) -> Dict[str, StateDict]:
        unet_state_dict = {}
        vae_state_dict = {}
        text_encoder_state_dict = {}
        text_encoder_2_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("model.diffusion_model."):
                unet_state_dict[key] = value
            elif key.startswith("first_stage_model."):
                vae_state_dict[key] = value
            elif key.startswith("conditioner.embedders.0.transformer."):
                text_encoder_state_dict[key] = value
            elif key.startswith("conditioner.embedders.1.model."):
                text_encoder_2_state_dict[key] = value
                

        return {
            "unet": unet_state_dict,
            "vae": vae_state_dict,
            "text_encoder": text_encoder_state_dict,
            "text_encoder_2": text_encoder_2_state_dict,
            "text_config": "sdxl_text_config.json",
            "text2_config": "sdxl_text2_config.json",
            "vae_config": "sdxl_vae_config.json",
            "unet_config": "sdxl_unet_config.json",
        }
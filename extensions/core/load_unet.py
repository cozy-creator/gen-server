from typing import Dict
from diffusers import UNet2DConditionModel
from core_library.node_interface import InputSpec, OutputSpec
from core_library.model_handler import load_unet

class LoadUnet:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, InputSpec]:
        return {
            "unet_state_dict": {
                "display_name": "UNet State Dict",
                "edge_type": "Dict[str, torch.Tensor]",
                "spec": {},
                "required": True,
            },
            "config_file": {
                "display_name": "Config File",
                "edge_type": "str",
                "spec": {},
                "required": True,
            },
            "device": {
                "display_name": "Device",
                "edge_type": "str",
                "spec": {},
                "required": False,
            },
            "model_type": {
                "display_name": "Model Type",
                "edge_type": "str",
                "spec": {},
                "required": True,
            },
        }

    @property
    def RETURN_TYPES(self) -> OutputSpec:
        return {
            "display_name": "Loaded UNet",
            "edge_type": "UNet2DConditionModel",
        }

    def __call__(self, unet_state_dict: dict, config_file: str, device: str = "cpu", model_type: str = "sd1.5") -> UNet2DConditionModel:
        return load_unet(unet_state_dict, config_file, device, model_type)

    @property
    def CATEGORY(self) -> str:
        return "model_loading"

    @property
    def display_name(self) -> Dict[str, str]:
        return {
            "en": "Load UNet",
            "es": "Cargar UNet",
        }

    @property
    def description(self) -> Dict[str, str]:
        return {
            "en": "Loads a UNet model from a state dictionary and configuration file.",
            "es": "Carga un modelo UNet desde un diccionario de estado y un archivo de configuración.",
        }
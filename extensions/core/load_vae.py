from typing import Dict
from core_library.node_interface import InputSpec, OutputSpec
from diffusers import AutoencoderKL
from core_library.model_handler import load_vae

class LoadVae:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, InputSpec]:
        return {
            "vae_state_dict": {
                "display_name": "VAE State Dict",
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
            "display_name": "Loaded VAE",
            "edge_type": "AutoencoderKL",
        }

    def __call__(self, vae_state_dict: dict, config_file: str, device: str = "cpu", model_type: str = "sd1.5") -> AutoencoderKL:
        return load_vae(vae_state_dict, config_file, device, model_type)

    @property
    def CATEGORY(self) -> str:
        return "model_loading"

    @property
    def display_name(self) -> Dict[str, str]:
        return {
            "en": "Load VAE",
            "es": "Cargar VAE",
        }

    @property
    def description(self) -> Dict[str, str]:
        return {
            "en": "Loads a VAE (Variational Autoencoder) model from a state dictionary and configuration file.",
            "es": "Carga un modelo VAE (Autoencoder Variacional) desde un diccionario de estado y un archivo de configuración.",
        }

from typing import Dict
from core_library.node_interface import InputSpec, OutputSpec
from transformers import CLIPTextModel
from core_library.model_handler import load_text_encoder, load_text_encoder_2

class LoadTextEncoder:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, InputSpec]:
        return {
            "text_encoder_state_dict": {
                "display_name": "Text Encoder State Dict",
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
            "display_name": "Loaded Text Encoder",
            "edge_type": "CLIPTextModel",
        }

    def __call__(self, text_encoder_state_dict: dict, config_file: str, device: str = "cpu", model_type: str = "sd1.5") -> CLIPTextModel:
        return load_text_encoder(text_encoder_state_dict, config_file, device, model_type)

    @property
    def CATEGORY(self) -> str:
        return "model_loading"

    @property
    def display_name(self) -> Dict[str, str]:
        return {
            "en": "Load Text Encoder",
            "es": "Cargar Codificador de Texto",
        }

    @property
    def description(self) -> Dict[str, str]:
        return {
            "en": "Loads a text encoder model from a state dictionary and configuration file.",
            "es": "Carga un modelo de codificador de texto desde un diccionario de estado y un archivo de configuración.",
        }


class LoadTextEncoder2:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, InputSpec]:
        return {
            "text_encoder_state_dict": {
                "display_name": "Text Encoder State Dict",
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
            "has_projection": {
                "display_name": "Has Projection",
                "edge_type": "bool",
                "spec": {},
                "required": False,
            },
        }

    @property
    def RETURN_TYPES(self) -> OutputSpec:
        return {
            "display_name": "Loaded Text Encoder 2",
            "edge_type": "CLIPTextModel",
        }

    def __call__(self, text_encoder_state_dict: dict, config_file: str, device: str = "cpu", model_type: str = "sdxl", has_projection: bool = False) -> CLIPTextModel:
        return load_text_encoder_2(text_encoder_state_dict, config_file, device, model_type, has_projection)

    @property
    def CATEGORY(self) -> str:
        return "model_loading"

    @property
    def display_name(self) -> Dict[str, str]:
        return {
            "en": "Load Text Encoder 2",
            "es": "Cargar Codificador de Texto 2",
        }

    @property
    def description(self) -> Dict[str, str]:
        return {
            "en": "Loads a text encoder model (version 2) from a state dictionary and configuration file. This is useful for loading sdxl models.",
            "es": "Carga un modelo de codificador de texto (versión 2) desde un diccionario de estado y un archivo de configuración. Esto es útil para cargar modelos sdxl.",
        }
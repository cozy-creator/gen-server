from core_library.node_interface import InputSpec, OutputSpec
from typing import Dict, Optional
import torch
from core_library.model_handler import load_safetensors


class LoadSafetensors:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, InputSpec]:
        return {
            "safetensors_file": {
                "display_name": "Safetensors File",
                "edge_type": "str",
                "spec": {},
                "required": True,
            },
            "model_type": {
                "display_name": "Model Type",
                "edge_type": "str",
                "spec": {},
                "required": False,
            },
            "component_name": {
                "display_name": "Component Name",
                "edge_type": "str",
                "spec": {},
                "required": False,
            },
        }

    @property
    def RETURN_TYPES(self) -> OutputSpec:
        return {
            "output": {
                "display_name": "Loaded Tensors",
                "edge_type": "Dict[str, torch.Tensor]",
            }
        }

    def __call__(self, safetensors_file: str, model_type: str = "sd1.5", component_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        return load_safetensors(safetensors_file, model_type, component_name)

    @property
    def CATEGORY(self) -> str:
        return "model_loading"

    @property
    def display_name(self) -> Dict[str, str]:
        return {
            "en": "Load Safetensors",
            "es": "Cargar Safetensors",
        }

    @property
    def description(self) -> Dict[str, str]:
        return {
            "en": "Loads tensors from a safetensors file.",
            "es": "Carga tensores desde un archivo de safetensors.",
        }

from typing import Dict, Optional
from transformers import CLIPTokenizer
from core_library.node_interface import InputSpec, OutputSpec

class LoadTokenizer:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, InputSpec]:
        return {
            "pretrained_model_name_or_path": {
                "display_name": "Pretrained Model Name or Path",
                "edge_type": "str",
                "spec": {},
                "required": False,
            },
        }

    @property
    def RETURN_TYPES(self) -> OutputSpec:
        return {
            "display_name": "Loaded Tokenizer",
            "edge_type": "CLIPTokenizer",
        }

    def __call__(self, pretrained_model_name_or_path: Optional[str] = None) -> CLIPTokenizer:
        if pretrained_model_name_or_path:
            return CLIPTokenizer.from_pretrained(pretrained_model_name_or_path)
        
        return CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    @property
    def CATEGORY(self) -> str:
        return "model_loading"

    @property
    def display_name(self) -> Dict[str, str]:
        return {
            "en": "Load Tokenizer",
        }

    @property
    def description(self) -> Dict[str, str]:
        return {
            "en": "Loads a tokenizer from a pretrained model or a specified path.",
            "es": "Carga un tokenizador desde un modelo preentrenado o una ruta especificada.",
        }

class LoadTokenizer2:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, InputSpec]:
        return {}

    @property
    def RETURN_TYPES(self) -> OutputSpec:
        return {
            "display_name": "Loaded Tokenizer 2",
            "edge_type": "CLIPTokenizer",
        }

    def __call__(self) -> CLIPTokenizer:
        return CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    @property
    def CATEGORY(self) -> str:
        return "model_loading"

    @property
    def display_name(self) -> Dict[str, str]:
        return {
            "en": "Load Tokenizer 2",
        }

    @property
    def description(self) -> Dict[str, str]:
        return {
            "en": "Loads the second tokenizer for the SDXL model type.",
            "es": "Carga el segundo tokenizador para el tipo de modelo SDXL.",
        }
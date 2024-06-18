from typing import Dict, Union, Any
import safetensors.torch 
from .architectures import ModelArchitecture, SD15Arch, SDXLArch
import os
import importlib


class ModelRegistry:
    """
    Registry for managing supported model architectures.
    """

    def __init__(self):
        self._architectures: Dict[str, ModelArchitecture] = {}
        # Register default architectures
        self.register(SD15Arch())
        self.register(SDXLArch())

    def register(self, architecture: ModelArchitecture):
        """
        Registers a new model architecture.
        """
        if architecture.id in self._architectures:
            raise ValueError(f"Architecture with ID '{architecture.id}' already registered.")
        self._architectures[architecture.id] = architecture
        print("Registered architecture with ID: ", self._architectures[architecture.id].id)


    def register_from_folder(self, folder_path: str):
        """Registers all architectures found in a specified folder."""

        for dir_name in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, dir_name)
            if os.path.isdir(dir_path) and "__init__.py" in os.listdir(dir_path):
                try: 
                    module = importlib.import_module(f"custom_architectures.{dir_name}")
                    arch_class = getattr(module, dir_name.capitalize() + "Arch")
                    self.register(arch_class())  # Instantiate and register the architecture 
                except (ImportError, AttributeError) as e:
                    print(f"Could not load architecture from {dir_path}: {e}")

    def detect_architecture(self, state_dict) -> Union[str, None]:
        """
        Detects the architecture from a state dictionary.
        """
        for arch_id, architecture in self._architectures.items():

            if architecture.detect(state_dict):
                return arch_id
        return None 

    def load_model(
        self, 
        filepath: str, 
    ) -> Any:
        """
        Loads a model from a safetensors file.
        """

        state_dict = safetensors.torch.load_file(filepath)
        arch_id = self.detect_architecture(state_dict)

        if not arch_id:
            raise ValueError(f"Unsupported model architecture found in '{filepath}'")

        components = self._architectures[arch_id].load(state_dict)
        components['arch_id'] = arch_id

        
        return components
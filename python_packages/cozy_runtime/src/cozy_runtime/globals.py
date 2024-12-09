from typing import Type, Any

from .base_types import (
    Architecture,

    TorchDevice,
)
from .utils.device import get_torch_device


_available_torch_device: TorchDevice = get_torch_device()

# Model Memory Manager
_MODEL_MEMORY_MANAGER = None

# Model Downloader
_MODEL_DOWNLOADER = None


_ARCHITECTURES: dict[str, type[Architecture[Any]]] = {}
"""
Global class containing all architecture definitions
"""


def get_model_downloader():
    """Get or create the global ModelManager instance"""
    global _MODEL_DOWNLOADER
    if _MODEL_DOWNLOADER is None:
        from .utils.model_downloader import ModelManager

        _MODEL_DOWNLOADER = ModelManager()
    return _MODEL_DOWNLOADER


def get_model_memory_manager():
    global _MODEL_MEMORY_MANAGER
    if _MODEL_MEMORY_MANAGER is None:
        from .utils.model_memory_manager import ModelMemoryManager

        _MODEL_MEMORY_MANAGER = ModelMemoryManager()
    return _MODEL_MEMORY_MANAGER


def update_architectures(architectures: dict[str, Type["Architecture"]]):
    global _ARCHITECTURES
    _ARCHITECTURES.update(architectures)


def get_architectures() -> dict[str, Type["Architecture"]]:
    return _ARCHITECTURES


# def update_custom_nodes(custom_nodes: dict[str, Type["CustomNode"]]):
#     global _CUSTOM_NODES
#     _CUSTOM_NODES.update(custom_nodes)


# def get_custom_nodes() -> dict[str, Type["CustomNode"]]:
#     return _CUSTOM_NODES



def get_available_torch_device():
    global _available_torch_device
    return _available_torch_device


def set_available_torch_device(device: TorchDevice):
    print("Setting device", device)
    global _available_torch_device
    _available_torch_device = device

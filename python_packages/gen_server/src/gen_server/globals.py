import os
from typing import Type, Any, Iterable, Union, Optional
from aiohttp import web
from .base_types.pydantic_models import ModelConfig
from .utils.download_manager import DownloadManager

from .base_types import (
    Architecture,
    CheckpointMetadata,
    CustomNode,
    ApiAuthenticator,
    TorchDevice,
)
from .utils.device import get_torch_device


_ENABLED_MODELS: dict[str, ModelConfig] = {}

_available_torch_device: TorchDevice = get_torch_device()

# Huggingface Manager
_HF_MODEL_MANAGER = None

# Model Memory Manager
_MODEL_MEMORY_MANAGER = None

# Download Manager
_download_manager: Optional[DownloadManager] = None

# API_ENDPOINTS: dict[str, Callable[[], Iterable[web.AbstractRouteDef]]] = {}
RouteDefinition = Union[Iterable[web.RouteDef], web.RouteTableDef]
_API_ENDPOINTS: dict[str, RouteDefinition] = {}
"""
Route-definitions to be added to Aiohttp
"""

_ARCHITECTURES: dict[str, type[Architecture[Any]]] = {}
"""
Global class containing all architecture definitions
"""

_CUSTOM_NODES: dict[str, Type[CustomNode]] = {}
"""
Nodes to compose together during server-side execution
"""

_WIDGETS: dict = {}
"""
TO DO
"""

_CHECKPOINT_FILES: dict[str, CheckpointMetadata] = {}
"""
Dictionary of all discovered checkpoint files
"""

_api_authenticator: Optional[Type[ApiAuthenticator]] = None


def get_hf_model_manager():
    global _HF_MODEL_MANAGER
    if _HF_MODEL_MANAGER is None:
        from .utils.hf_model_manager import HFModelManager

        _HF_MODEL_MANAGER = HFModelManager()
    return _HF_MODEL_MANAGER


def get_model_memory_manager():
    global _MODEL_MEMORY_MANAGER
    if _MODEL_MEMORY_MANAGER is None:
        from .utils.model_memory_manager import ModelMemoryManager

        _MODEL_MEMORY_MANAGER = ModelMemoryManager()
    return _MODEL_MEMORY_MANAGER


def update_api_endpoints(endpoints: dict[str, RouteDefinition]):
    global _API_ENDPOINTS
    _API_ENDPOINTS.update(endpoints)


def get_api_endpoints() -> dict[str, RouteDefinition]:
    return _API_ENDPOINTS


def update_architectures(architectures: dict[str, Type["Architecture"]]):
    global _ARCHITECTURES
    _ARCHITECTURES.update(architectures)


def get_architectures() -> dict[str, Type["Architecture"]]:
    return _ARCHITECTURES


def update_custom_nodes(custom_nodes: dict[str, Type["CustomNode"]]):
    global _CUSTOM_NODES
    _CUSTOM_NODES.update(custom_nodes)


def get_custom_nodes() -> dict[str, Type["CustomNode"]]:
    return _CUSTOM_NODES


def update_widgets(widgets: dict):
    global _WIDGETS
    _WIDGETS.update(widgets)


def get_widgets() -> dict:
    return _WIDGETS


def update_checkpoint_files(checkpoint_files: dict[str, "CheckpointMetadata"]):
    global _CHECKPOINT_FILES
    _CHECKPOINT_FILES.update(checkpoint_files)


def get_checkpoint_files() -> dict[str, "CheckpointMetadata"]:
    return _CHECKPOINT_FILES


def update_api_authenticator(api_authenticator: Optional[Type["ApiAuthenticator"]]):
    global _api_authenticator
    print(f"Setting api_authenticator to {api_authenticator}")
    _api_authenticator = api_authenticator


def get_api_authenticator() -> Optional[Type["ApiAuthenticator"]]:
    global _api_authenticator
    return _api_authenticator


# def get_hf_model_manager() -> HFModelManager:
#     global _HF_MODEL_MANAGER
#     return _HF_MODEL_MANAGER

# def get_model_memory_manager() -> ModelMemoryManager:
#     global _MODEL_MEMORY_MANAGER
#     return _MODEL_MEMORY_MANAGER


def get_download_manager() -> Optional[DownloadManager]:
    global _download_manager
    return _download_manager


def set_download_manager(download_manager: DownloadManager):
    global _download_manager
    _download_manager = download_manager


def get_available_torch_device():
    global _available_torch_device
    return _available_torch_device


def set_available_torch_device(device: TorchDevice):
    print("Setting device", device)
    global _available_torch_device
    _available_torch_device = device

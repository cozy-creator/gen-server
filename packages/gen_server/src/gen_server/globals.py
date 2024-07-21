from typing import Type, Optional, Any, Iterable, Union
from aiohttp import web

from . import CustomNode
from .base_types import Architecture, CheckpointMetadata


class InstallCommandConfig(BaseSettings):
    """
    Configuration for the `install` CLI command. Loaded by the pydantic-settings library
    """

    model_config = model_config

    def __init__(self, **data: Any):
        workspace_path = (
            data["workspace"]
            if data.get("workspace_path") is not None
            else DEFAULT_WORKSPACE_PATH
        )

        print(f"Workspace path here: {workspace_path}")

        ensure_env_file(workspace_path)
        super().__init__(**data)

    env_file: Optional[str] = Field(
        default=None,
        description="Path to .env file",
    )

    workspace_path: str = Field(
        default_factory=lambda: os.path.expanduser(DEFAULT_WORKSPACE_PATH),
        description="Local file-directory where /assets and /temp files will be loaded from and saved to.",
    )


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

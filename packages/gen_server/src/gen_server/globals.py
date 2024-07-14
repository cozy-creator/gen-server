import os
import json
from enum import Enum
from typing import Type, Optional, Any, Iterable, Union
from aiohttp import web
from pydantic import Extra, Field, BaseModel, field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import CustomNode
from .base_types import Architecture, CheckpointMetadata

DEFAULT_WORKSPACE_PATH = "~/.cozy-creator/"
# Note: the default models_path is [{workspace_path}/models]
# Note: the default assets_path is [{workspace_path}/assets]
# DEFAULT_ENV_FILE_PATH = os.path.join(os.getcwd(), ".env")


class FilesystemTypeEnum(str, Enum):
    LOCAL = "local"
    S3 = "s3"

    # make it case-insensitive
    @classmethod
    def _missing_(cls, value: str):
        for member in cls:
            if member.value.lower() == value.lower():
                return member

#    @classmethod
#     def _missing_(cls, value: object):
#         if not isinstance(value, str):
#             raise ValueError(f"Invalid value for FilesystemTypeEnum: {value}")
#         for member in cls:
#             if member.value.lower() == value.lower():
#                 return member

class S3Credentials(BaseModel):
    """
    Credentials to read from / write to an S3-compatible API
    """

    endpoint_url: Optional[str] = Field(
        default=None,
        description="S3 endpoint url, such as https://<accountid>.r2.cloudflarestorage.com",
    )

    access_key: Optional[str] = Field(
        default=None, description="Access key for S3 authentication"
    )

    secret_key: Optional[str] = Field(
        default=None,
        description="Secret key for S3 authentication",
    )

    region_name: Optional[str] = Field(
        default=None, description="Optional region, such as `us-east-1` or `weur`"
    )

    bucket_name: Optional[str] = Field(
        default=None, description="Name of the S3 bucket to read from / write to"
    )

    folder: Optional[str] = Field(
        default=None, description="Folder within the S3 bucket"
    )


class RunCommandConfig(BaseSettings):
    """
    Configuration for the run CLI command. Loaded by the pydantic-settings library
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        cli_parse_args=True,
        env_prefix="cozy_",
        env_ignore_empty=True,
        env_nested_delimiter="__",
        # env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    env_file: Optional[str] = Field(
        default=None,
        description="Path to .env file",
    )

    secrets_dir: Optional[str] = Field(
        default=None,
        description="Path to secrets directory",
    )

    environment: str = Field(
        default="dev",
        description="Server environment, such as dev or prod",
    )

    host: str = Field(
        default="127.0.0.1",
        description="Hostname or IP-address",
    )

    port: int = Field(
        default=8881,
        description="Port to bind to",
    )

    workspace_path: str = Field(
        default_factory=lambda: os.path.expanduser(DEFAULT_WORKSPACE_PATH),
        description="Local file-directory where /assets and /temp files will be loaded from and saved to.",
    )

    models_path: Optional[str] = Field(
        default=None,
        description=(
            "The directory where models will be saved to and loaded from by default."
            "The default value is {workspace_path}/models"
        ),
    )

    aux_models_paths: list[str] = Field(
        default_factory=list,
        description=(
            "A list of additional directories containing model-files (serialized state dictionaries), "
            "such as .safetensors or .pth files."
        ),
    )

    assets_path: Optional[str] = Field(
        default=None,
        description="Directory for storing assets locally, Default value is {workspace_path}/assets",
    )

    filesystem_type: FilesystemTypeEnum = Field(
        default=FilesystemTypeEnum.LOCAL,
        description=(
            "If `local`, files will be saved to and served from the {assets_path} folder."
            "If `s3`, files will be saved to and served from the specified S3 bucket and folder."
        ),
    )

    s3: Optional[S3Credentials] = Field(
        default=None,
        description="Credentials to read from and write to an S3-compatible API.",
    )

    # This allows the aux_models_paths field to be a comma-separated string of paths
    # or a list of paths
    @field_validator("aux_models_paths", mode="before")
    @classmethod
    def parse_and_set_aux_models_paths(cls, v: Any, info: ValidationInfo) -> list[str]:
        # If no value provided, return an empty list
        if v is None or (isinstance(v, list) and len(v) == 0):
            return []

        # If it's a string, try to parse it
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError:
                parsed = v.split(",")  # fallback to comma-separated values

            # Ensure each path is expanded
            return [os.path.expanduser(path.strip()) for path in parsed]

        # If it's already a list, ensure each path is expanded
        if isinstance(v, list):
            return [os.path.expanduser(path) for path in v]

        # If it's none of the above, return as is
        return v

    @field_validator("s3", mode="before")
    @classmethod
    def validate_s3(cls, v: Any) -> Optional[S3Credentials]:
        if v is None:
            return None
        if isinstance(v, str):
            try:
                # Parse the string as JSON
                s3_dict = json.loads(v)
                # Create an S3Credentials instance from the parsed dict
                return S3Credentials(**s3_dict)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string for S3 credentials: {e}")
        elif isinstance(v, dict):
            # If it's already a dict, create an S3Credentials instance
            return S3Credentials(**v)
        elif isinstance(v, S3Credentials):
            # If it's already an S3Credentials instance, return it as-is
            return v
        else:
            raise ValueError(f"Unexpected type for S3 credentials: {type(v)}")


class BuildWebCommandConfig(BaseSettings):
    """
    Configuration for the build-web CLI command. Loaded by the pydantic-settings library
    """

    model_config = {"extra": Extra.ignore}

    env_file: Optional[str] = Field(
        default=None,
        description="Path to .env file",
    )

    secrets_dir: Optional[str] = Field(
        default=None,
        description="Path to secrets directory",
    )


# API_ENDPOINTS: dict[str, Callable[[], Iterable[web.AbstractRouteDef]]] = {}
RouteDefinition = Union[Iterable[web.RouteDef], web.RouteTableDef]
API_ENDPOINTS: dict[str, RouteDefinition] = {}
"""
Route-definitions to be added to Aiohttp
"""

ARCHITECTURES: dict[str, type[Architecture[Any]]] = {}
"""
Global class containing all architecture definitions
"""

CUSTOM_NODES: dict[str, Type[CustomNode]] = {}
"""
Nodes to compose together during server-side execution
"""

WIDGETS: dict = {}
"""
TO DO
"""

CHECKPOINT_FILES: dict[str, CheckpointMetadata] = {}
"""
Dictionary of all discovered checkpoint files
"""

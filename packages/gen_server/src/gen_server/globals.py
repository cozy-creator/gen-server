import os
import json
from enum import Enum
from typing import Type, Optional, Any, Iterable, Union, Literal
from aiohttp import web
from pydantic import Field, BaseModel, field_validator, ValidationInfo, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import CustomNode
from .base_types import Architecture, CheckpointMetadata


DEFAULT_WORKSPACE_PATH = "~/.cozy-creator/"
# Note: the default model_dirs is [{workspace_path}/models]
# DEFAULT_AUX_MODELS_PATHS = ["~/.cozy-creator/models"]
# DEFAULT_ENV_FILE_PATH = os.path.join(os.getcwd(), ".env")


class S3Credentials(BaseModel):
    """
    Credentials to read from / write to an S3-compatible API
    """
    type: Literal['s3'] = 's3'
    
    endpoint_url: Optional[str] = Field(
        default=None,
        description="S3 endpoint url, such as https://<accountid>.r2.cloudflarestorage.com"
    )

    access_key: Optional[str] = Field(
        default=None,
        description="Access key for S3 authentication"
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

class LocalStorage(BaseModel):
    """
    Where to store files and serve them from locally
    """
    type: Literal['local'] = 'local'
    
    assets_dir: Optional[str] = Field(
        default=None,
        description="Directory for storing assets locally, Default value is {workspace_path}/assets"
    )

class FilesystemType(BaseModel):
    storage: Union[S3Credentials, LocalStorage] = Field(discriminator='type')
    
    @field_validator("storage", mode="before")
    @classmethod
    def validate_storage(cls, v: Any) -> Union[S3Credentials, LocalStorage]:
        if isinstance(v, (S3Credentials, LocalStorage)):
            return v
        if isinstance(v, dict):
            if v.get('type') == 's3':
                return S3Credentials(**v)
            elif v.get('type') == 'local':
                return LocalStorage(**v)
        if isinstance(v, str):
            try:
                data = json.loads(v)
                if data.get('type') == 's3':
                    return S3Credentials(**data)
                elif data.get('type') == 'local':
                    return LocalStorage(**data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string for storage configuration: {e}")
        raise ValueError(f"Invalid storage configuration: {v}")
    
    @model_validator(mode='after')
    def validate_storage_type(cls, values):
        if not isinstance(values.storage, (S3Credentials, LocalStorage)):
            raise ValueError('Invalid storage configuration')
        return values

# # To use the storage field:
# if filesystem.type == 's3':
#     s3_creds = filesystem.storage
#     # Use s3_creds.endpoint_url, s3_creds.access_key, etc.
# elif filesystem.type == 'local':
#     assets_dir = filesystem.storage
#     # Use assets_dir as a string path

# # You can also use isinstance for type checking:
# if isinstance(filesystem.storage, S3Credentials):
#     # It's S3
#     s3_creds = filesystem.storage
#     # Use s3 credentials
# elif isinstance(filesystem.storage, str):
#     # It's local
#     assets_dir = filesystem.storage
#     # Use local assets directory


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

    filesystem_type: FilesystemType = Field(
        default_factory=lambda: FilesystemType(storage=LocalStorage()),
        description=(
            "If `local`, files will be saved to and served from the {workspace_path}/assets folder. "
            "If `s3`, files will be saved to and served from the specified S3 bucket."
        ),
    )

    @field_validator("aux_models_paths", mode="before")
    @classmethod
    def parse_and_set_aux_models_paths(
        cls,
        v: Any,
        info: ValidationInfo
    ) -> list[str]:
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
    
    @field_validator("filesystem_type", mode="before")
    @classmethod
    def validate_filesystem_type(cls, v: Any) -> FilesystemType:
        if isinstance(v, str):
            try:
                # Try to parse the string as JSON
                data = json.loads(v)
                if isinstance(data, dict):
                    if data.get('type') == 'local':
                        return FilesystemType(storage=LocalStorage(**data))
                    elif data.get('type') == 's3':
                        return FilesystemType(storage=S3Credentials(**data))
            except json.JSONDecodeError:
                # If it's not valid JSON, check for simple string values
                if v.upper() == 'LOCAL':
                    return FilesystemType(storage=LocalStorage())
                elif v.upper() == 'S3':
                    return FilesystemType(storage=S3Credentials())
        
        # If it's already a FilesystemType instance, return it as is
        if isinstance(v, FilesystemType):
            return v
        
        # If it's a dict, try to create a FilesystemType instance
        if isinstance(v, dict):
            if v.get('type') == 'local':
                return FilesystemType(storage=LocalStorage(**v))
            elif v.get('type') == 's3':
                return FilesystemType(storage=S3Credentials(**v))
        
        # If we can't parse it, let Pydantic handle the validation error
        return v


class BuildWebCommandConfig(BaseSettings):
    """
    Configuration for the build-web CLI command. Loaded by the pydantic-settings library
    """
    model_config = { 
        "extra": "ignore"
    }
    
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


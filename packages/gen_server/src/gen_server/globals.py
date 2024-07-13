import os
import json
from typing import Type, Optional, Any, Iterable, Union
from aiohttp import web
from pydantic import Field, BaseModel, field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import CustomNode
from .base_types import Architecture, CheckpointMetadata


DEFAULT_WORKSPACE_DIR = "~/.cozy-creator/"
# Note: the default model_dirs is [{workspace_dir}/models]
# DEFAULT_MODELS_DIRS = ["~/.cozy-creator/models"]
# DEFAULT_ENV_FILE_PATH = os.path.join(os.getcwd(), ".env")


class S3Credentials(BaseModel):
    """
    Credentials to read from / write to an S3-compatible API
    """
    
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

    workspace_dir: str = Field(
        default_factory=lambda: os.path.expanduser(DEFAULT_WORKSPACE_DIR),
        description="Local file-directory where /assets and /temp files will be loaded from and saved to.",
    )

    # allow models_dirs list to be parsed from both JSON and a comma-separated string
    models_dirs: list[str] = Field(
        default_factory=list,
        description=(
            "A list of directories containing model-files (serialized state dictionaries), "
            "such as .safetensors or .pth files. The default value is [{workspace_dir}/models]"
        ),
    )

    filesystem_type: str = Field(
        default="LOCAL",
        description="Type of filesystem to use. (Not currently used; may remove?)",
    )

    s3: Optional[S3Credentials] = Field(
        default=None,
        description="S3 credentials",
    )
    
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

    @field_validator("models_dirs", mode="before")
    @classmethod
    def parse_and_set_models_dirs(
        cls,
        v: Any,
        info: ValidationInfo
    ) -> list[str]:
        values = info.data
        # If no value provided, use default based on workspace_dir
        if v is None or (isinstance(v, list) and len(v) == 0):
            workspace = values.get('workspace_dir', os.path.expanduser(DEFAULT_WORKSPACE_DIR))
            return [os.path.join(workspace, 'models')]
        
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


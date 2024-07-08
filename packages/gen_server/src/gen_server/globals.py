import os
import json
from argparse import ArgumentParser

import boto3
from typing import Type, Optional, Any, Iterable, Union
from aiohttp import web
from dotenv import load_dotenv

from . import CustomNode
from .base_types import Architecture, CheckpointMetadata
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    CliSubCommand,
    CliSettingsSource,
)
from pydantic import Field, BaseModel, field_validator

# class RunCommand(BaseModel):
#     config: CliPositionalArg[str] = Field(description="Path to JSON config file")

# class MainCommand(BaseSettings):
#     config: CliPositionalArg[str] = Field(default=os.getcwd(), description="Path to JSON config file")

#     model_config = SettingsConfigDict(
#         cli_parse_args = True,
#         cli_prog_name = "cozy-creator"
#     )


DEFAULT_WORKSPACE_DIR = "~/.comfy-creator/"
DEFAULT_MODELS_DIRS = ["~/.comfy-creator/models"]
DEFAULT_ENV_FILE_PATH = os.path.join(os.getcwd(), ".env")


# def json_config_settings_source(settings: BaseSettings) -> dict[str, Any]:
#     """
#     Load settings from a JSON file.
#     """
#     encoding = settings.model_config.get('env_file_encoding', 'utf-8')
#     config_path = Path('config.json')
#     if config_path.exists():
#         return json.loads(config_path.read_text(encoding))
#     else:
#         return {}


class S3Credentials(BaseModel):
    """
    Credentials to read from / write to an S3 bucket
    """

    region_name: Optional[str] = Field(
        default=None, description="Name of the S3 bucket region"
    )

    bucket_name: Optional[str] = Field(
        default=None, description="Name of the S3 bucket"
    )

    endpoint_fqdn: Optional[str] = Field(
        default=None, description="Fully Qualified Domain Name of the S3 endpoint"
    )

    folder: Optional[str] = Field(
        default=None, description="Folder within the S3 bucket"
    )

    access_key: Optional[str] = Field(
        default=None, description="Access key for S3 authentication"
    )

    secret_key: Optional[str] = Field(
        default=None,
        description="Secret key for S3 authentication",
    )


class ExtendedRunConfig:
    environment: Optional[str] = Field(
        default=None,
        description="Server environment",
    )

    hostname: str = Field(
        default="0.0.0.0",
        description="Hostname to use",
    )

    port: int = Field(
        default=8080,
        description="Port to use",
    )

    workspace_dir: str = Field(
        default_factory=lambda: os.path.expanduser(DEFAULT_WORKSPACE_DIR),
        description="Workspace directory",
    )

    # allow models_dirs list to be parsed from both JSON and a comma-separated string
    models_dirs: list[str] = Field(
        default_factory=lambda: [
            os.path.expanduser(dir) for dir in DEFAULT_MODELS_DIRS
        ],
        description="Directories containing models",
    )

    filesystem_type: str = Field(
        default="LOCAL",
        description="Type of filesystem to use",
    )

    s3: Optional[S3Credentials] = Field(
        default=None,
        description="S3 credentials",
    )

    @field_validator("models_dirs", mode="before")
    @classmethod
    def parse_models_dirs(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v.split(",")  # fallback to comma-separated values
        return v

    @field_validator("s3", mode="before")
    @classmethod
    def validate_s3(cls, v):
        if v is None:
            s3_config = S3Credentials()
            if any(value is not None for value in s3_config.model_dump().values()):
                return s3_config
        return v


class RunCommandConfig(BaseModel, ExtendedRunConfig):
    model_config = {"extra": "ignore"}

    env_file: Optional[str] = Field(
        default=None,
        description="Path to .env file",
    )

    secrets_dir: Optional[str] = Field(
        default=None,
        description="Path to secrets directory",
    )


class CozyRunConfig(BaseSettings, ExtendedRunConfig):
    """
    Configuration for the `cozy run` command
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        env_nested_delimiter="__",
        env_file=DEFAULT_ENV_FILE_PATH,
        extra="ignore",
    )

    def __init__(
        self,
        secrets_dir: Optional[str],
        env_file: Optional[str] = DEFAULT_ENV_FILE_PATH,
        **kwargs,
    ):
        args = {k: v for k, v in kwargs.items() if v is not None}
        if env_file is None:
            env_file = DEFAULT_ENV_FILE_PATH

        super().__init__(_env_file=env_file, _secrets_dir=secrets_dir, **args)


class CozyCommands(BaseSettings):
    """
    Configuration for the cozy-creator application
    """

    model_config = SettingsConfigDict(
        cli_parse_args=True, case_sensitive=False, extra="ignore"
    )

    def __init__(self, parser: ArgumentParser):
        cli_settings = CliSettingsSource(
            CozyCommands,
            root_parser=parser,
            cli_parse_args=True,
        )

        super().__init__(_cli_settings_source=cli_settings)

    # List of commands to run
    run: CliSubCommand[RunCommandConfig] = Field(description="Run the Cozy Gen Server")

    # @classmethod
    # def settings_customise_sources(
    #     cls,
    #     settings_cls: Type[BaseSettings],
    #     init_settings: PydanticBaseSettingsSource,
    #     env_settings: PydanticBaseSettingsSource,
    #     dotenv_settings: PydanticBaseSettingsSource,
    #     file_secret_settings: PydanticBaseSettingsSource,
    # ) -> tuple[PydanticBaseSettingsSource, ...]:
    #     return (
    #         init_settings,
    #         CliSettingsSource(settings_cls),
    #         env_settings,
    #         dotenv_settings,
    #         file_secret_settings,
    #     )

    # @classmethod
    # def settings_customise_sources(
    #     cls,
    #     settings_cls,
    #     init_settings,
    #     env_settings,
    #     file_secret_settings,
    # ):
    #     return (
    #         init_settings,
    #         json_config_settings_source,
    #         env_settings,
    #         file_secret_settings,
    #     )


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


def initialize_config(
    env_path: Optional[str] = None, config_path: Optional[str] = None
):
    """
    Load the .env file and config file specified into the global configuration dataclass
    """
    global comfy_config

    load_dotenv()

    # These env variables can be access using os.environ or os.getenv
    if env_path:
        print(f"Loading from {env_path}")
        load_dotenv(dotenv_path=env_path, override=True)

    # Parse the config-file
    config_dict = {}
    if config_path:
        try:
            with open(config_path, "r") as file:
                config_dict = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file {config_path} was not found.")
        except json.JSONDecodeError:
            print("Invalid JSON format in the configuration file.")

    # Find our directories
    comfy_config.hostname = config_dict.get("host", "localhost")
    comfy_config.port = config_dict.get("port", 8080)
    comfy_config.filesystem_type = config_dict.get("filesystem_type", "LOCAL")
    comfy_config.workspace_dir = os.path.expanduser(
        config_dict.get("workspace_dir", DEFAULT_WORKSPACE_DIR)
    )
    comfy_config.models_dirs = [
        os.path.expanduser(path)
        for path in config_dict.get("models_dirs", DEFAULT_MODELS_DIRS)
    ]

    #  Load S3 credentials and instantiate the S3 client
    s3_credentials = config_dict.get("s3_credentials")
    if s3_credentials:
        s3_endpoint_fqdn = s3_credentials.get("endpoint_fqdn")
        s3_access_key = s3_credentials.get("access_key")
        bucket_name = s3_credentials.get("bucket_name")
        folder = s3_credentials.get("folder")
        s3_secret_key = os.getenv("S3_SECRET_KEY")

        required_fields = [
            s3_endpoint_fqdn,
            s3_access_key,
            s3_secret_key,
            bucket_name,
            folder,
        ]
        if None in required_fields:
            print("Error: Missing required S3 configuration fields.")
        else:
            comfy_config.s3["bucket_name"] = bucket_name
            comfy_config.s3["folder"] = folder
            comfy_config.s3["url"] = f"https://{bucket_name}.{s3_endpoint_fqdn}"

            try:
                comfy_config.s3["client"] = boto3.client(
                    "s3",
                    region_name=s3_endpoint_fqdn.split(".")[0],
                    endpoint_url=f"https://{s3_endpoint_fqdn}",
                    aws_access_key_id=s3_access_key,
                    aws_secret_access_key=s3_secret_key,
                )
                print("S3 client configured successfully.")
            except Exception as e:
                print(f"Error configuring S3 client: {e}")

    if comfy_config.filesystem_type == "S3" and not comfy_config.s3.get("client"):
        raise ValueError("Failed to load required S3 credentials.")

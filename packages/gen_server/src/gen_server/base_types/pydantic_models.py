from ast import List
import os
import json
import yaml
from enum import Enum
from typing import Type, Optional, Any, Iterable, Union
from pydantic import BaseModel, Field, field_validator, ValidationInfo, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings import PydanticBaseSettingsSource, YamlConfigSettingsSource

DEFAULT_HOME_DIR = os.path.expanduser("~/.cozy-creator/")

class ModelConfig(BaseModel):
    """
    Model configuration loaded from a config.yaml file usually
    """
    category: str
    variant: str


def get_default_home_dir():
    """
    Returns the default home directory for the Cozy Creator.
    """
    cozy_home_dir = os.environ.get("COZY_HOME_DIR")
    if cozy_home_dir:
        return cozy_home_dir
    
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return os.path.join(xdg_data_home, ".cozy-creator")
    
    return DEFAULT_HOME_DIR


def is_running_in_docker() -> bool:
    """
    Determines if our process is currently in a docker environment or not.
    """
    container_env = os.environ.get("container")
    if container_env == "docker":
        return True
    else:
        return os.path.exists("/.dockerenv")


def find_config_file() -> Optional[str]:
    """
    Finds the config.yaml file in various locations.
    Returns the path to the config file if found, None otherwise.
    """
    possible_locations = [
        os.environ.get("COZY_CONFIG_FILE"), # from cli-command or environment variable
        '/etc/config/config.yaml',  # Kubernetes ConfigMap typical mount path
        os.path.join(get_default_home_dir(), 'config.yaml'),  # Cozy's home directory
    ]
    
    for location in possible_locations:
        if location is not None and os.path.exists(location):
            print(f'Config file loaded from: {location}')
            return location
    
    return None


class FilesystemTypeEnum(str, Enum):
    LOCAL = "local"
    S3 = "s3"

    # make it case-insensitive
    @classmethod
    def _missing_(cls, value: object):
        if not isinstance(value, str):
            raise ValueError(f"Invalid value for FilesystemTypeEnum: {value}")
        for member in cls:
            if member.value.lower() == value.lower():
                return member


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

    public_url: Optional[str] = Field(
        default=None,
        description=(
            "Url where the S3 files can be publicly accessed from, example: https://storage.cozy.dev."
            "If not specified, the endpoint-url will be used instead"
        ),
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
        env_file_encoding="utf-8",
        extra="allow",
        yaml_file_encoding="utf-8",
    )
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        This is used so that we can load our config.yaml file from a location determined at runtime
        """
        yaml_settings = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=find_config_file()
        )
        return (
            init_settings,
            yaml_settings,
            env_settings,
            file_secret_settings,
        )
    
    home_dir: str = Field(
        default=get_default_home_dir(),
        description=("Local file-directory where /assets and /model folders will be loaded from and saved to. "
                     "XDG_DATA_HOME is checked as a fallback if not specified."),
    )

    environment: str = Field(
        default="dev",
        description="Server environment, such as dev or prod",
    )

    host: str = Field(
        default="0.0.0.0" if is_running_in_docker() else "localhost",
        description="Hostname or IP-address",
    )

    port: int = Field(
        default=8881,
        description="Port to bind to",
    )

    models_path: Optional[str] = Field(
        default=None,
        description=(
            "The directory where models will be saved to and loaded from by default."
            "The default value is {home}/models"
        ),
    )

    aux_models_paths: list[str] = Field(
        default=list(),
        description=(
            "A list of additional directories containing model-files (serialized state dictionaries), "
            "such as .safetensors or .pth files."
        ),
    )

    assets_path: Optional[str] = Field(
        default=None,
        description="Directory for storing assets locally, Default value is {home}/assets",
    )

    filesystem_type: FilesystemTypeEnum = Field(
        default=FilesystemTypeEnum.LOCAL,
        description=(
            "If `local`, files will be saved to and served from the {assets_path} folder."
            "If `s3`, files will be saved to and served from the specified S3 bucket and folder."
        ),
    )
    
    enabled_models: Optional[dict[str, ModelConfig]] = Field(
        default=None,
        description=("Dictionary of models to be downloaded from hugging face on startup and made available "
                     "for inference.")
    )

    s3: Optional[S3Credentials] = Field(
        default=None,
        description="Credentials to read from and write to an S3-compatible API.",
    )

    api_authenticator: Optional[str] = Field(
        default=None,
        description="The authenticator to be used in authenticating api requests.",
    )

    @field_validator('enabled_models', mode='before')
    def parse_json_string(cls, v: object):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # should we return None instead?
                raise ValueError(f"Invalid JSON string for enabled_models: {v}")
        return v

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

    model_config = SettingsConfigDict(
        case_sensitive=False,
        cli_parse_args=True,
        env_prefix="cozy_",
        env_ignore_empty=True,
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        extra="allow",
    )
    
    # TO DO: add a specifier for web-dir location


class DownloadCommandConfig(BaseSettings):
    """
    Configuration for the `download` CLI command. Loaded by the pydantic-settings library
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        cli_parse_args=True,
        env_prefix="cozy_",
        env_ignore_empty=True,
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        extra="allow",
    )

    repo_id: str = Field(
        default=None,
        description="The ID of the model repository to download from",
    )

    file_name: Optional[str] = Field(
        default=None,
        description="The name of the file to download, if not specified, the entire repo will be downloaded",
    )


    sub_folder: Optional[str] = Field(
        default=None,
        description="The subfolder within the repo to download from",
    )


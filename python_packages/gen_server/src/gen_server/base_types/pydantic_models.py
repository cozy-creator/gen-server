import os
import json
from enum import Enum
from typing import Type, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings import PydanticBaseSettingsSource, YamlConfigSettingsSource

DEFAULT_HOME_DIR = os.path.expanduser("~/.cozy-creator/")


def get_default_home_dir() -> str:
    """
    Returns the default home directory for the Cozy Creator.
    """
    cozy_home = os.environ.get("COZY_HOME")
    if cozy_home:
        return cozy_home

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


class ComponentConfig(BaseModel):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        populate_by_name=True,
    )

    source: str = Field(alias="Source")
    class_name: Optional[str | tuple[str, str]] = Field(default=None, alias="ClassName")


class PipelineConfig(BaseModel):
    """
    Diffusers pipeline configurations; loaded from a config.yaml file usually
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        populate_by_name=True,
    )

    source: str = Field(alias="Source")
    class_name: Optional[str | tuple[str, str]] = Field(default=None, alias="ClassName")
    components: Optional[dict[str, ComponentConfig]] = Field(
        default=None, alias="Components"
    )

    @field_validator("components")
    def validate_components(cls, v: Any) -> Optional[dict[str, ComponentConfig]]:
        if v is not None:
            return {
                key: ComponentConfig(**value) if isinstance(value, dict) else value
                for key, value in v.items()
            }
        return v


class RunCommandConfig(BaseSettings):
    """
    Configuration for the run CLI command. Loaded by the pydantic-settings library
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        cli_parse_args=True,
        extra="allow",
        alias_generator=lambda s: s.replace("_", "-").lower(),
        populate_by_name=True,
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

        return (init_settings,)

    home_dir: str = Field(
        default=get_default_home_dir(),
        description="Cozy creator's home directory, usually ~/.cozy-creator",
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
        default=8882,
        description="Port to bind Python serverto",
    )

    pipeline_defs: dict[str, PipelineConfig] = Field(
        default=dict(),
        description=(
            "Dictionary of models to be downloaded from hugging face on startup and made available "
            "for inference."
        ),
    )

    warmup_models: list[str] = Field(
        default=[],
        description="List of models to warm up on startup.",
    )

    models_path: str = Field(
        default=os.path.join(get_default_home_dir(), "models"),
        description=(
            "The directory where models will be saved to and loaded from by default."
            "The default value is {home}/models"
        ),
    )

    @field_validator("pipeline_defs", mode="before")
    @classmethod
    def parse_pipeline_defs(cls, v: Any) -> dict[str, PipelineConfig]:
        # If it's already a dict, return as is
        if isinstance(v, dict):
            return v

        # If it's a string, try to parse it as JSON
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return {
                    model_id: PipelineConfig(**config)
                    for model_id, config in parsed.items()
                }
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for pipeline_defs: {e}") from e

        return dict()  # Return empty dict as default

    @field_validator("warmup_models", mode="before")
    @classmethod
    def parse_and_set_warmup_models(cls, v: Any, info: ValidationInfo) -> list[str]:
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
            return parsed

        # If it's none of the above, return as is
        return v

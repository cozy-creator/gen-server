import os
import argparse
from typing import Optional, List, Callable
from pydantic_settings import CliSettingsSource

from .base_types.pydantic_models import RunCommandConfig, FilesystemTypeEnum

DEFAULT_HOME_DIR = os.path.expanduser("~/.cozy-creator/")


cozy_config: Optional[RunCommandConfig] = None
"""
Global configuration for the Cozy Gen-Server
"""


def config_loaded() -> bool:
    """
    Returns a boolean indicating whether the config has been loaded.
    This will return True if called within the gen_server runtime, since the config is loaded at the start of the server.
    """
    return cozy_config is not None


def set_config(config: RunCommandConfig):
    """
    Sets the global configuration object .
    """
    global cozy_config
    cozy_config = config


def get_config() -> RunCommandConfig:
    """
    Returns the global configuration object. This is only available if the config has been loaded, which happens at
    the start of the server, else it will raise an error.
    """
    if cozy_config is None:
        raise ValueError("Config has not been loaded yet")

    return cozy_config


ParseArgsMethod = Callable[
    [argparse.ArgumentParser, Optional[List[str]], Optional[argparse.Namespace]],
    Optional[argparse.Namespace],
]


def init_config(
    run_parser: argparse.ArgumentParser,
    parse_args_method: ParseArgsMethod,
    env_file: Optional[str] = None,
    secrets_dir: Optional[str] = "/run/secrets",
) -> RunCommandConfig:
    """
    Loads the configuration for the server.
    This should be called at the start of the Python server.
    """
    cli_settings = CliSettingsSource(
        RunCommandConfig,
        root_parser=run_parser,
        cli_parse_args=True,
        cli_enforce_required=True,
        # overwrite this method so that we don't get errors from unknown args
        parse_args_method=parse_args_method,
    )

    cozy_config = RunCommandConfig(
        _env_file=env_file,  # type: ignore
        _secrets_dir=secrets_dir,  # type: ignore
        _cli_settings_source=cli_settings(args=True),  # type: ignore
    )

    # This updates the configuration globally for the Python server
    set_config(cozy_config)

    return cozy_config


def is_model_enabled(model_name: str) -> bool:
    """
    Returns a boolean indicating whether a model is enabled in the global configuration.
    """
    config = get_config()
    if config.pipeline_defs is None:
        return False

    return model_name in config.pipeline_defs.keys()


def get_mock_config(
    filesystem_type: FilesystemTypeEnum = FilesystemTypeEnum.LOCAL,
) -> RunCommandConfig:
    """
    Returns a mock (or test) version of the global configuration object.
    This can be used outside of the cozy server environment.
    """

    environment = "test"
    home_dir = DEFAULT_HOME_DIR

    if filesystem_type == FilesystemTypeEnum.LOCAL:
        os.environ["COZY_FILESYSTEM_TYPE"] = filesystem_type
        os.environ["COZY_ASSETS_PATH"] = os.path.join(home_dir, "assets")
    else:
        os.environ["COZY_FILESYSTEM_TYPE"] = FilesystemTypeEnum.S3
        os.environ["COZY_S3__FOLDER"] = "public"
        os.environ["COZY_S3__ACCESS_KEY"] = "test"
        os.environ["COZY_S3__SECRET_KEY"] = "test"
        os.environ["COZY_S3__REGION_NAME"] = "us-east-1"
        os.environ["COZY_S3__BUCKET_NAME"] = "test-bucket"
        os.environ["COZY_S3__ENDPOINT_URL"] = (
            "https://voidtech-storage-dev.nyc3.digitaloceanspaces.com"
        )

    return RunCommandConfig(
        port=8881,
        host="127.0.0.1",
        environment=environment,
    )

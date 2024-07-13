from pydantic_settings import CliSettingsSource
from .globals import RunCommandConfig, LocalStorage
import argparse
from typing import Optional, List, Callable
import os


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


def get_config() -> RunCommandConfig:
    """
    Returns the global configuration object. This is only available if the config has been loaded, which happens at
    the start of the server, else it will raise an error.
    """
    if cozy_config is None:
        raise ValueError("Config has not been loaded yet")
    return cozy_config

ParseArgsMethod = Callable[[argparse.ArgumentParser, Optional[List[str]], Optional[argparse.Namespace]], Optional[argparse.Namespace]]

def init_config(
    run_parser: argparse.ArgumentParser,
    parse_args_method: ParseArgsMethod,
    env_file: str = ".env",
    secrets_dir: str = "/run/secrets",
) -> RunCommandConfig:
    """
    Loads the configuration for the server.
    This should be called at the start of the server.
    """
    cli_settings = CliSettingsSource(
        RunCommandConfig,
        root_parser=run_parser,
        cli_parse_args=True,
        cli_enforce_required=False,
        # overwrite this method so that we don't get errors from unknown args
        parse_args_method=parse_args_method,
    )

    # This updates the configuration globally
    global cozy_config
    cozy_config = RunCommandConfig(
        _env_file=env_file,  # type: ignore
        _secrets_dir=secrets_dir,  # type: ignore
        _cli_settings_source=cli_settings(args=True),  # type: ignore
    )

    return cozy_config


def get_assets_dir() -> str:
    """
    Helper function; used to find the /assets directory.
    """
    config = get_config()
    if config.filesystem_type.storage.type == 'local':
        local_storage = config.filesystem_type.storage
        if isinstance(local_storage, LocalStorage) and local_storage.assets_dir:
            return local_storage.assets_dir
    return os.path.join(config.workspace_path, 'assets')


def get_models_dir() -> str:
    """
    Helper function; used to find the /models directory.
    """
    config = get_config()
    if config.models_path:
        return config.models_path
    return os.path.join(config.workspace_path, 'models')

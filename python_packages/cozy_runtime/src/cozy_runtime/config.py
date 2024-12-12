import os
import argparse
from typing import Optional, List, Callable
from .base_types.config import RuntimeConfig


cozy_config: Optional[RuntimeConfig] = None
"""
Global configuration for the Cozy Gen-Server
"""


def config_loaded() -> bool:
    """
    Returns a boolean indicating whether the config has been loaded.
    This will return True if called within the cozy runtime, since the config is loaded at the start.
    """
    return cozy_config is not None


def set_config(config: RuntimeConfig):
    """
    Sets the global configuration object .
    """
    global cozy_config
    cozy_config = config


def get_config() -> RuntimeConfig:
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


def is_model_enabled(model_name: str) -> bool:
    """
    Returns a boolean indicating whether a model is enabled in the global configuration.
    """
    config = get_config()
    if config.pipeline_defs is None:
        return False

    return model_name in config.pipeline_defs.keys()


def get_mock_config() -> RuntimeConfig:
    """
    Returns a mock (or test) version of the global configuration object.
    This can be used outside of the cozy server environment.
    """

    environment = "test"
    # home_dir = DEFAULT_HOME_DIR

    return RuntimeConfig(
        port=8881,
        host="127.0.0.1",
        environment=environment,
    )

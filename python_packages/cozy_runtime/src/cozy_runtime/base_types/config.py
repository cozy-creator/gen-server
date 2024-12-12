import os
from dataclasses import dataclass, field
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

DEFAULT_HOME_DIR = os.path.expanduser("~/.cozy-creator/")


@dataclass
class PipelineConfig:
    source: str
    class_name: Optional[str | tuple[str, str]]
    components: Optional[dict[str, "ComponentConfig"]]


@dataclass
class ComponentConfig:
    source: str
    class_name: Optional[str | tuple[str, str]]
    kwargs: Optional[dict[str, Any]]


def default_home_dir() -> str:
    return DEFAULT_HOME_DIR


def default_models_path() -> str:
    return os.path.join(DEFAULT_HOME_DIR, "models")


def default_host() -> str:
    return "0.0.0.0" if os.path.exists("/.dockerenv") else "localhost"


@dataclass
class RuntimeConfig:
    home_dir: str = field(default_factory=default_home_dir)
    environment: str = "dev"
    host: str = field(default_factory=default_host)
    port: int = 8882
    pipeline_defs: dict[str, PipelineConfig] = field(default_factory=dict)
    enabled_models: list[str] = field(default_factory=list)
    models_path: str = field(default_factory=default_models_path)

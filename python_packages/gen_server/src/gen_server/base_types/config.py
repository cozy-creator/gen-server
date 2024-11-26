import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ====== Parse cli arguments ======


@dataclass
class PipelineConfig:
    source: str
    class_name: Optional[str | tuple[str, str]]
    components: Optional[dict[str, "ComponentConfig"]]


@dataclass
class ComponentConfig:
    source: str
    class_name: Optional[str | tuple[str, str]]


@dataclass
class RuntimeConfig:
    home_dir: str
    environment: str
    host: str
    port: int
    pipeline_defs: dict[str, PipelineConfig]
    warmup_models: list[str]
    models_path: str


def parse_pipeline_defs(value):
    """Parse pipeline definitions from command line argument"""
    if not value:
        return {}

    try:
        if isinstance(value, str):
            return json.loads(value)
        if isinstance(value, dict):
            return value
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse pipeline definitions: {e}")
        return {}


def parse_warmup_models(value):
    """Parse warmup models from command line argument"""
    if not value:
        return []

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value.split(",")
    return value


def parse_arguments() -> RuntimeConfig:
    """Parse command line arguments and return configuration"""
    parser = argparse.ArgumentParser(description="Cozy Creator")

    config = get_default_config()

    parser.add_argument(
        "--home-dir", default=config.home_dir, help="Cozy creator's home directory"
    )
    parser.add_argument(
        "--environment",
        default=config.environment,
        help="Server environment (dev/prod)",
    )
    parser.add_argument("--host", default=config.host, help="Hostname or IP-address")
    parser.add_argument(
        "--port",
        type=int,
        default=config.port,
        help="Port to bind Python runtime to",
    )
    parser.add_argument(
        "--pipeline-defs",
        type=str,
        default=config.pipeline_defs,
        help="JSON string of pipeline definitions",
    )
    parser.add_argument(
        "--warmup-models",
        type=str,
        default=config.warmup_models,
        help="Comma-separated list or JSON array of models to warm up",
    )
    parser.add_argument(
        "--models-path",
        type=str,
        default=config.models_path,
        help="Path to models directory",
    )

    args = parser.parse_args()

    # Update config with parsed arguments
    config = RuntimeConfig(
        home_dir=args.home_dir,
        environment=args.environment,
        host=args.host,
        port=args.port,
        pipeline_defs=parse_pipeline_defs(args.pipeline_defs),
        warmup_models=parse_warmup_models(args.warmup_models),
        models_path=args.models_path or os.path.join(args.home_dir, "models"),
    )

    return config


def get_default_config() -> RuntimeConfig:
    """Returns default configuration values"""
    return RuntimeConfig(
        home_dir=os.path.expanduser("~/.cozy-creator/"),
        environment="dev",
        host="0.0.0.0" if os.path.exists("/.dockerenv") else "localhost",
        port=8882,
        pipeline_defs={},
        warmup_models=[],
        models_path=os.path.expanduser("~/.cozy-creator/models"),
    )

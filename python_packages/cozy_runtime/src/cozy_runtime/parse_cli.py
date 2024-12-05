import argparse
import json
import os
import logging
from .base_types.config import RuntimeConfig, PipelineConfig
from typing import Optional

logger = logging.getLogger(__name__)

# ====== Parse cli arguments ======


def parse_pipeline_defs(value: Optional[str]) -> dict[str, PipelineConfig]:
    """Parse pipeline definitions from command line argument"""
    if not value:
        return {}

    try:
        loaded = json.loads(value)
        if isinstance(loaded, dict):
            return loaded
        else:
            logger.error("Pipeline definitions are not a dictionary")
            return {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse pipeline definitions: {e}")
        return {}


def parse_enabled_models(value: Optional[str]) -> list[str]:
    """Parse enabled models from command line argument"""
    if not value:
        return []

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.split(",")


def parse_arguments() -> RuntimeConfig:
    """Parse command line arguments and return configuration"""
    parser = argparse.ArgumentParser(description="Cozy Creator")

    default_config = RuntimeConfig()

    parser.add_argument(
        "--home-dir",
        default=default_config.home_dir,
        help="Cozy creator's home directory",
    )
    parser.add_argument(
        "--environment",
        default=default_config.environment,
        help="Server environment (dev/prod)",
    )
    parser.add_argument(
        "--host", default=default_config.host, help="Hostname or IP-address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_config.port,
        help="Port to bind Python runtime to",
    )
    parser.add_argument(
        "--pipeline-defs",
        type=str,
        default=default_config.pipeline_defs,
        help="JSON string of pipeline definitions",
    )
    parser.add_argument(
        "--enabled-models",
        type=str,
        default=default_config.enabled_models,
        help="Comma-separated list or JSON array of models to warm up",
    )
    parser.add_argument(
        "--models-path",
        type=str,
        default=default_config.models_path,
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
        enabled_models=parse_enabled_models(args.enabled_models),
        models_path=args.models_path or os.path.join(args.home_dir, "models"),
    )

    return config

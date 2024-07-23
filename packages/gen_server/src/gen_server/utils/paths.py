import os
from pathlib import Path
from gen_server.config import get_config

DEFAULT_DIST_PATH = Path("/srv/www/cozy/dist")
DEFAULT_WEB_DIR = Path("/srv/www/cozy/web")


def get_web_root() -> Path:
    """
    Utility function to find the web root directory for the static file server
    """
    dev_path = Path(__file__).parent.parent.parent.parent.parent.parent / "web" / "dist"
    prod_path = DEFAULT_DIST_PATH

    if dev_path.exists():
        return dev_path
    elif prod_path.exists():
        return prod_path
    else:
        raise FileNotFoundError("Neither primary nor secondary web root paths exist.")


def get_web_dir() -> Path:
    """
    Utility function to find the web root directory for the static file server
    """
    dev_path = Path(__file__).parent.parent.parent.parent.parent.parent / "web"
    prod_path = DEFAULT_WEB_DIR

    if dev_path.exists():
        return dev_path
    elif prod_path.exists():
        return prod_path
    else:
        raise FileNotFoundError("Neither primary nor secondary web root paths exist.")


def get_assets_dir() -> str:
    """
    Helper function; used to find the /assets directory.
    """
    config = get_config()
    if config.assets_path:
        return os.path.expanduser(config.assets_path)
    return os.path.join(os.path.expanduser(config.workspace_path), "assets")


def get_models_dir() -> str:
    """
    Helper function; used to find the /models directory.
    """
    config = get_config()

    if config.models_path:
        return os.path.expanduser(config.models_path)
    return os.path.join(os.path.expanduser(config.workspace_path), "models")


def get_s3_public_url() -> str:
    """
    Helper function; used to find the public S3 URL.
    """
    config = get_config()
    if config.s3 and config.s3.public_url:
        return config.s3.public_url
    elif config.s3 and config.s3.endpoint_url:
        return config.s3.endpoint_url
    else:
        raise ValueError("No S3 public URL or endpoint URL found in the configuration")


def should_expand(path: str) -> bool:
    """
    Helper function; used to determine if a path should be expanded.
    """
    return path.startswith("~") or path.startswith("/")

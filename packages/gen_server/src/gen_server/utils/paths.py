import os
import tempfile
from pathlib import Path
from ..config import get_config
import sys
from typing import Optional
from ..static import ENV_TEMPLATE

DEFAULT_DIST_PATH = Path("/srv/www/cozy/dist")
DEFAULT_WEB_DIR = Path("/srv/www/cozy/web")
APP_NAME = "cozy-creator"


def get_assets_dir():
    config = get_config()
    if config.assets_path:
        return config.assets_path
    else:
        return os.path.join(config.home, "assets")


def get_models_dir():
    config = get_config()
    if config.models_path:
        return config.models_path
    else:
        return os.path.join(config.home, "models")


def get_next_counter(assets_dir: str, filename_prefix: str) -> int:
    def map_filename(filename: str) -> tuple[int, str]:
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[: prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1 :].split("_")[0])
        except:
            digits = 0
        return (digits, prefix)

    try:
        counter = (
            max(
                filter(
                    lambda a: a[1][:-1] == filename_prefix and a[1][-1] == "_",
                    map(map_filename, os.listdir(assets_dir)),
                )
            )[0]
            + 1
        )
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(assets_dir, exist_ok=True)
        counter = 1
    return counter


def ensure_app_dirs():
    config = get_config()
    
    # Ensure home directory exists
    home_created = False
    if not os.path.exists(config.home):
        os.makedirs(config.home, exist_ok=True)
        home_created = True
    
    # Ensure assets directory exists
    assets_dir = get_assets_dir()
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir, exist_ok=True)
    
    # Ensure models directory exists
    models_dir = get_models_dir()
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    # If home directory was just created, write the example env file
    if home_created:
        _write_env_example_file(config.home)


def _write_env_example_file(workspace_path: str):
    try:
        env_path = os.path.join(workspace_path, ".env.example")
        if not os.path.exists(env_path):
            with open(env_path, "w") as f:
                f.write(ENV_TEMPLATE)
    except Exception as e:
        print(f"Error while creating initializing env file: {str(e)}")


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

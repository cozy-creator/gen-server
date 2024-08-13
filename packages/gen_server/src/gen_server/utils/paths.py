from functools import cache
import os
import sys
import tempfile
from pathlib import Path
from importlib import resources
import shutil
from typing import Optional

from gen_server import examples
from ..config import get_config

APP_NAME = "cozy-creator"
DEFAULT_DIST_PATH = Path("/srv/www/cozy/dist")
DEFAULT_WEB_DIR = Path("/srv/www/cozy/web")
DEFAULT_HOME_DIR = os.path.expanduser("~/.cozy-creator/")
# Note: the default models_path is [{home}/models]
# Note: the default assets_path is [{home}/assets]
# DEFAULT_ENV_FILE_PATH = os.path.join(os.getcwd(), ".env")


def get_assets_dir():
    config = get_config()
    if config.assets_path:
        return os.path.expanduser(config.assets_path)
    return os.path.join(get_home_dir(), "assets")


def get_models_dir():
    config = get_config()

    if config.models_path:
        return os.path.expanduser(config.models_path)
    return os.path.join(get_home_dir(), "models")

def get_home_dir():
    return os.path.expanduser(get_config().home_dir)

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
    home_dir = get_home_dir()
    if not os.path.exists(home_dir):
        os.makedirs((home_dir), exist_ok=True)
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
        _write_example_files(home_dir)


def _write_example_files(workspace_path: str):
    try:
        # Use importlib.resources to get the path of the example files
        with resources.path(examples, '.env.local.example') as env_example_path, \
             resources.path(examples, 'config.example.yaml') as config_example_path:
            # Construct the destination paths
            env_path = os.path.join(workspace_path, ".env.local.example")
            config_path = os.path.join(workspace_path, "config.example.yaml")

            # Copy the .env.local.example file if it doesn't exist
            if not os.path.exists(env_path):
                shutil.copy2(env_example_path, env_path)

            # Copy the models.yaml file if it doesn't exist
            if not os.path.exists(config_path):
                shutil.copy2(config_example_path, config_path)

    except Exception as e:
        print(f"Error while initializing example files: {str(e)}")


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


def is_runpod_available() -> bool:
    """
    Returns a boolean indicating whether the server is running within a RunPod.
    """
    return os.environ.get("RUNPOD_POD_ID") is not None


def get_runpod_url(port: str | int, token: Optional[str] = None) -> str:
    """
    Returns the URL of the RunPod.
    """

    pod_id = os.environ.get("RUNPOD_POD_ID")
    url = f"https://{pod_id}-{port}.proxy.runpod.net"
    if token is not None:
        return f"{url}?token={token}"
    return url


def get_server_url():
    """
    Returns the URL of the server.
    """

    config = get_config()
    if is_runpod_available():
        return get_runpod_url(config.port)
    return f"http://{config.host}:{config.port}"


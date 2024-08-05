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
WIN_APP_NAME = "Cozy Creator"
MEDIA_DIR_NAME = "media"


def get_absolute_path(path: Optional[str], default: str) -> str:
    if path is None:
        return default
    expanded_path = os.path.expanduser(path)
    return expanded_path if os.path.isabs(expanded_path) else default


def get_app_dirs():
    if sys.platform == "win32":
        # Windows; both 32 and 64 bit versions despite the name 'win32'
        # app_data = get_absolute_path(os.environ.get('APPDATA'), os.path.expanduser('~\\AppData\\Roaming'))
        local_app_data = get_absolute_path(os.environ.get('LOCALAPPDATA'), os.path.expanduser('~\\AppData\\Local'))
        cache_dir = os.path.expanduser('~\\.cache')

        return {
            'data': os.path.join(local_app_data, WIN_APP_NAME),
            'config': os.path.join(local_app_data, WIN_APP_NAME),
            'cache': os.path.join(cache_dir, APP_NAME),
            'state': os.path.join(local_app_data, WIN_APP_NAME),
            'runtime': os.path.join(tempfile.gettempdir(), APP_NAME),
        }
    
    elif sys.platform == "darwin":
        # macOS
        home = os.path.expanduser("~")
        return {
            'data': os.path.join(home, 'Library', 'Application Support', APP_NAME),
            'config': os.path.join(home, 'Library', 'Preferences', APP_NAME),
            'cache': os.path.join(home, 'Library', 'Caches', APP_NAME),
            'state': os.path.join(home, 'Library', 'Application Support', APP_NAME),
            'runtime': os.path.join(tempfile.gettempdir(), APP_NAME),
        }
    
    else:
        # Linux and other Unix-like systems (supports XDG)
        home = os.path.expanduser("~")
        xdg_data_home = get_absolute_path(os.environ.get("XDG_DATA_HOME"), os.path.join(home, ".local", "share"))
        xdg_config_home = get_absolute_path(os.environ.get("XDG_CONFIG_HOME"), os.path.join(home, ".config"))
        xdg_cache_home = get_absolute_path(os.environ.get("XDG_CACHE_HOME"), os.path.join(home, ".cache"))
        xdg_state_home = get_absolute_path(os.environ.get("XDG_STATE_HOME"), os.path.join(home, ".local", "state"))
        xdg_runtime_dir = get_absolute_path(os.environ.get("XDG_RUNTIME_DIR"), tempfile.gettempdir())
        
        # These are not used, but are part of the XDG spec

        # Default values for XDG_DATA_DIRS and XDG_CONFIG_DIRS
        # xdg_data_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share/:/usr/share/").split(":")
        # xdg_config_dirs = os.environ.get("XDG_CONFIG_DIRS", "/etc/xdg").split(":")

        # Ensure all paths are absolute
        # xdg_data_dirs = [os.path.abspath(path) for path in xdg_data_dirs]
        # xdg_config_dirs = [os.path.abspath(path) for path in xdg_config_dirs]

        return {
            'data': os.path.join(xdg_data_home, APP_NAME),
            'config': os.path.join(xdg_config_home, APP_NAME),
            'cache': os.path.join(xdg_cache_home, APP_NAME),
            'state': os.path.join(xdg_state_home, APP_NAME),
            'runtime': os.path.join(xdg_runtime_dir, APP_NAME),
            # 'data_dirs': [os.path.join(path, APP_NAME) for path in xdg_data_dirs],
            # 'config_dirs': [os.path.join(path, APP_NAME) for path in xdg_config_dirs],
        }

def get_media_dir():
    return os.path.join(get_app_dirs()['data'], MEDIA_DIR_NAME)


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


# TO DO: everything below needs to be revised

def ensure_workspace_path(path: str):
    subdirs = ["models", ["assets", "temp"]]

    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        write_env_example_file(
            path
        )  # we only write this the first time the dir is created

    for subdir in subdirs:
        if isinstance(subdir, list):
            subdir_path = os.path.join(path, *subdir)
        else:
            subdir_path = os.path.join(path, subdir)

        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)


def write_env_example_file(workspace_path: str):
    try:
        env_path = os.path.expanduser(os.path.join(workspace_path, ".env.example"))
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

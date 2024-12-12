import os
from ..config import get_config



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

import yaml
from .paths import get_model_config_path

_MODEL_CONFIG = None

def load_model_config():
    global _MODEL_CONFIG
    config_path = get_model_config_path()
    with open(config_path, 'r') as file:
        _MODEL_CONFIG = yaml.safe_load(file)

def get_model_config():
    global _MODEL_CONFIG
    if _MODEL_CONFIG is None:
        load_model_config()
    return _MODEL_CONFIG
import os
import struct
from typing import List, Any, Dict, Optional

import torch

from packages.gen_server.src.gen_server import (
    StateDict,
    Checkpoint,
)
from packages.gen_server.src.gen_server.globals import comfy_config, PRETRAINED_MODELS
from packages.gen_server.src.gen_server.utils import load_models


import json

METADATA_HEADER_SIZE = 8


def extract_safetensors_metadata(file_path) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    if os.stat(file_path).st_size < METADATA_HEADER_SIZE:
        print(f"Error: File '{file_path}' is too small.")
        return

    with open(file_path, "rb") as file:
        header_size_bytes = file.read(METADATA_HEADER_SIZE)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size is None or header_size == 0:
            return
        header_bytes = file.read(header_size)
        header = json.loads(header_bytes)
    return header.get("__metadata__")


def load_checkpoints():
    for model_dir in comfy_config.models_dirs:
        for dirpath, _dirnames, filenames in os.walk(model_dir):
            for filename in filenames:
                model_file = os.path.join(dirpath, filename)
                if not os.path.isfile(model_file):
                    print(f"Error: File '{model_file}' not found.")
                    continue

                try:
                    components = load_models.from_file(
                        str(model_file), device=torch.device("cuda")
                    )
                    metadata = extract_safetensors_metadata(model_file)
                    display_name = (
                        metadata.get("name")
                        if metadata.get("name")
                        else os.path.splitext(filename)[0]
                    )

                    checkpoint = Checkpoint(display_name, components, metadata)
                    PRETRAINED_MODELS.update({str(model_file): checkpoint})
                except Exception as e:
                    print(
                        f"Error: Unexpected error while loading model from file '{model_file}': {e}"
                    )
                    continue


def has_all_keys(keys: List[str], state_dict: StateDict) -> bool:
    """
    Detects if the given state dictionary matches this architecture.
    """
    for key in keys:
        if key not in state_dict:
            return False
    return True

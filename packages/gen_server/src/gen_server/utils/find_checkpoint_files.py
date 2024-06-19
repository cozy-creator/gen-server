import os
import struct
from typing import List, Any, Dict, Optional
import torch
from ..base_types import CheckpointMetadata
from ..globals import comfy_config, CHECKPOINT_FILES
from ..utils import load_models
import json
import uuid
import datetime

METADATA_HEADER_SIZE = 8


def find_checkpoint_files(model_dirs: List[str]) -> dict[str, CheckpointMetadata]:
    """
    Scans the specified directories for checkpoint files.
    Once found, it compiles the checkpoint metadata including the absolute path and returns a dictionary
    of checkpoint metadata keyed by a random UUID.
    """
    checkpoint_metadata: dict[str, CheckpointMetadata] = {}
    valid_extensions = ('.safetensors', '.pth', '.ckpt', '.pt')

    for model_dir in model_dirs:
        for dirpath, _dirnames, filenames in os.walk(model_dir):
            for filename in filenames:
                if not filename.endswith(valid_extensions):
                    continue  # Skip file

                model_file = os.path.join(dirpath, filename)
                absolute_path = os.path.abspath(model_file)
                if not os.path.isfile(absolute_path):
                    print(f"Error: File '{absolute_path}' not found.")
                    continue

                try:
                    state_dict = load_models.state_dict_from_file(absolute_path)
                    components = load_models.detect_all(state_dict)
                    metadata = extract_safetensors_metadata(absolute_path)
                    
                    base, ext = os.path.splitext(filename)
                    display_name = base  # Use the filename as the display name
                    date_modified = datetime.datetime.fromtimestamp(os.path.getmtime(absolute_path))

                    checkpoint = CheckpointMetadata(
                        display_name=display_name,
                        author=metadata.get("author", "Unknown"),
                        components=components,
                        date_modified=date_modified,
                        file_type=ext,
                        file_path=absolute_path  # Include the absolute path in the metadata
                    )
                    
                    # We generate a random UUID for each checkpoint file
                    # TO DO: should we try something stable, like blake3 hashes instead?
                    checkpoint_uuid = str(uuid.uuid4())
                    
                    checkpoint_metadata[checkpoint_uuid] = checkpoint
                except Exception as e:
                    print(
                        f"Error: Unexpected error while loading model from file '{absolute_path}': {e}"
                    )
                    continue
    
    return checkpoint_metadata


def extract_safetensors_metadata(file_path) -> Dict[str, Any]:
    if not file_path.endswith('.safetensors'):
        print(f"Error: File '{file_path}' is not a '.safetensors' file.")
        return {}
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        return {}

    with open(file_path, "rb") as file:
        header_size_bytes = file.read(METADATA_HEADER_SIZE)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size is None or header_size == 0:
            return {}
        header_bytes = file.read(header_size)
        header = json.loads(header_bytes)
        
    return header.get("__metadata__", {})
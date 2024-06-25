import os
import struct
from typing import List, Any, Dict, Optional
from ..base_types import CheckpointMetadata, Architecture
from .load_models import load_state_dict_from_file, components_class_from_state_dict
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
                    metadata = extract_safetensors_metadata(absolute_path)

                    state_dict = load_state_dict_from_file(absolute_path)
                    
                    components = components_class_from_state_dict(state_dict, metadata)
                    
                    
                    # output_space_counts = {'SD1': 0, 'SDXL': 0, 'SD3': 0}
                    # for component in components.values():
                    #     if component.output_space == 'SD1':
                    #         output_space_counts['SD1'] += 1
                    #     elif component.output_space == 'SDXL':
                    #         output_space_counts['SDXL'] += 1
                    #     elif component.output_space == 'SD3':
                    #         output_space_counts['SD3'] += 1
                    
                    base_filename, ext = os.path.splitext(filename)
                    display_name = base_filename  # Use the filename as the display name
                    date_modified = datetime.datetime.fromtimestamp(os.path.getmtime(absolute_path))


                    checkpoint = CheckpointMetadata(
                        display_name=display_name,
                        author = metadata.get("modelspec.author") or metadata.get("author") or "Unknown",
                        category=determine_category(components),
                        components=components,
                        date_modified=date_modified,
                        file_type=ext,
                        file_path=absolute_path  # Include the absolute path in the metadata
                    )

                    
                    # We generate a random UUID for each checkpoint file
                    # TO DO: should we try something stable, like blake3 hashes instead?
                    # For now we're going to just use filenames; not a great strategy over allthough
                    # checkpoint_id = str(uuid.uuid4())
                    checkpoint_id = base_filename
                    
                    checkpoint_metadata[checkpoint_id] = checkpoint
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


# TO DO: this is kind of a dumb way to go about this; see if we can come up
# with something more general
def determine_category(components: Dict[str, Architecture]) -> str:
    output_space_counts = {'SD1': 0, 'SDXL': 0, 'SD3': 0}
    
    for component in components.values():
        if component.get("output_space") == 'SD1':
            output_space_counts['SD1'] += 1
        elif component.get("output_space") == 'SDXL':
            output_space_counts['SDXL'] += 1
        elif component.get("output_space") == 'SD3':
            output_space_counts['SD3'] += 1
    
    max_category = max(output_space_counts.items(), key=lambda x: x[1])
    return max_category[0] if max_category[1] > 0 else 'Unknown'

import os
import glob
from typing import List, Dict
from ..base_types import CheckpointMetadata, ComponentMetadata
from .load_models import (
    load_state_dict_from_file,
    find_component_models,
    read_safetensors_metadata,
)
import datetime


def find_checkpoint_files(models_paths: List[str]) -> dict[str, CheckpointMetadata]:
    """
    Scans the specified directories for checkpoint files.
    Once found, it compiles the checkpoint metadata including the absolute path and returns a dictionary
    of checkpoint metadata keyed by a random UUID.
    """
    checkpoint_metadata: dict[str, CheckpointMetadata] = {}
    # TO DO: do we want to expand support to pickle-based types?
    # We need to support .onnx based models for sure.
    valid_extensions = (".safetensors", ".pth", ".ckpt", ".pt")

    for models_path in models_paths:
        # Use glob to find files with the specified extensions
        pattern = os.path.join(models_path, "**", "*")
        for extension in valid_extensions:
            full_pattern = f"{pattern}{extension}"
            for absolute_path in glob.iglob(full_pattern, recursive=True):
                filename = os.path.basename(absolute_path)
                if not os.path.isfile(absolute_path):
                    print(f"Error: File '{absolute_path}' not found.")
                    continue

                try:
                    if filename.endswith(".safetensors"):
                        metadata = read_safetensors_metadata(absolute_path)
                    else:
                        metadata = {}

                    state_dict = load_state_dict_from_file(absolute_path)
                    components_metadata = find_component_models(state_dict, metadata)

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
                    date_modified = datetime.datetime.fromtimestamp(
                        os.path.getmtime(absolute_path)
                    )

                    checkpoint = CheckpointMetadata(
                        display_name=display_name,
                        author=metadata.get("modelspec.author")
                        or metadata.get("author")
                        or "Unknown",
                        category=determine_category(components_metadata),
                        components=components_metadata,
                        date_modified=date_modified,
                        file_type=ext,
                        file_path=absolute_path,  # Include the absolute path in the metadata
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

        # print(checkpoint_metadata)

    return checkpoint_metadata


# TO DO: this is kind of a dumb way to go about this; see if we can come up
# with something more general
def determine_category(components: Dict[str, ComponentMetadata]) -> str:
    output_space_counts = {"SD1": 0, "SDXL": 0, "SD3": 0, "AuraFlow": 0}

    for component in components.values():
        if component["output_space"] == "SD1":
            output_space_counts["SD1"] += 1
        elif component["output_space"] == "SDXL":
            output_space_counts["SDXL"] += 1
        elif component["output_space"] == "SD3":
            output_space_counts["SD3"] += 1
        elif component["output_space"] == "AuraFlow":
            output_space_counts["AuraFlow"] += 1

    max_category = max(output_space_counts.items(), key=lambda x: x[1])
    return max_category[0] if max_category[1] > 0 else "Unknown"

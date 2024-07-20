from __future__ import annotations
import os
from pathlib import Path
import torch
from typing import Type, Optional, Any
from safetensors.torch import load_file as safetensors_load_file
from spandrel import canonicalize_state_dict
from spandrel.__helpers.unpickler import (
    RestrictedUnpickle,
)  # probably shouldn't import from private modules...

from ..base_types import Architecture, StateDict, TorchDevice, ComponentMetadata
from ..globals import get_architectures

import struct
import json

METADATA_HEADER_SIZE = 8

ARCHITECTURES = get_architectures()


# TO DO: make this more efficient; we don't want to have to evaluate EVERY architecture
# for EVERY file. ALSO we need stop multiple architectures from claiming the same
# keys; i.e., if there are 5 architecture definitions for stable-diffusion-1 installed,
# then only the first one should get to claim those keys, otherwise it gets confusing
# on which model it should use
def from_file(
    path: str | Path,
    device: Optional[TorchDevice] = None,
    registry: dict[str, Type[Architecture]] = ARCHITECTURES,
) -> dict[str, Architecture]:
    # print(registry)
    """
    Loads a model from a file path. It detects the architecture, instantiates the
    architecture, and loads the state dict into the PyTorch class.

    Throws a `ValueError` if the file extension is not supported.
    Returns an empty dictionary if no supported model architecture is found.
    """
    state_dict = load_state_dict_from_file(path, device=device)
    metadata = read_safetensors_metadata(path)
    # print(path)

    return from_state_dict(state_dict, metadata, device, registry)


def from_state_dict(
    state_dict: StateDict,
    metadata: dict[str, Any] = {},
    device: Optional[TorchDevice] = None,
    registry: dict[str, Type[Architecture]] = ARCHITECTURES,
) -> dict[str, Architecture]:
    """
    Load a model from the given state dict.

    Returns an empty dictionary if no supported model architecture is found.
    """
    # Fetch class instances
    components = components_from_state_dict(state_dict, metadata, registry)

    # Load the state dict into the class instance, and move to device
    for _arch_id, architecture in components.items():
        try:
            architecture.load(state_dict, device)
        except Exception as e:
            print(e)


    return components


def components_from_state_dict(
    state_dict: StateDict,
    metadata: dict,
    registry: dict[str, Type[Architecture]] = ARCHITECTURES,
) -> dict[str, Architecture]:
    """
    Detect all models present inside of a state dict; does not load the state-dict into
    memory however; it only calls the Architecture's constructor to return a class instance.
    """
    components: dict[str, Architecture] = {}

    # print(metadata)


    for arch_id, architecture in registry.items():  # Iterate through all architectures
        try:
            # print("Now in load model")
            # print(metadata)
            # print(architecture)
            # print("Done above")
            checkpoint_metadata = architecture.detect(
                state_dict=state_dict, metadata=metadata
            )
            # print(checkpoint_metadata)
            # print("Done in load model")
            # detect_signature = inspect.signature(architecture.detect)
            # if 'state_dict' in detect_signature.parameters and 'metadata' in detect_signature.parameters:
            #     checkpoint_metadata = architecture.detect(state_dict=state_dict, metadata=metadata)
            # elif 'state_dict' in detect_signature.parameters:
            #     checkpoint_metadata = architecture.detect(state_dict=state_dict)
            # elif 'metadata' in detect_signature.parameters:
            #     checkpoint_metadata = architecture.detect(metadata=metadata)
            # else:
            #     continue
        except Exception:
            checkpoint_metadata = None

        if checkpoint_metadata is not None:
            model = architecture(metadata=metadata)
            components.update({arch_id: model})

    return components


def load_state_dict_from_file(
    path: str | Path, device: Optional[TorchDevice] = None
) -> StateDict:
    """
    Load the state dict of a model from the given file path.

    State dicts are typically only useful to pass them into the `load`
    function of a specific architecture.

    Throws a `ValueError` if the file extension is not supported.
    """
    extension = os.path.splitext(path)[1].lower()
    if isinstance(device, str):
        device = torch.device(device)  # make pyright type-checker happy

    state_dict: StateDict
    if extension == ".pt":
        try:
            state_dict = _load_torchscript(path, device)
        except RuntimeError:
            # If torchscript loading fails, try loading as a normal state dict
            try:
                pth_state_dict = _load_pth(path, device)
            except Exception:
                pth_state_dict = None

            if pth_state_dict is None:
                # the file was likely a torchscript file, but failed to load
                # re-raise the original error, so the user knows what went wrong
                raise

            state_dict = pth_state_dict

    elif extension == ".pth" or extension == ".ckpt":
        state_dict = _load_pth(path, device)
    elif extension == ".safetensors":
        state_dict = _load_safetensors(path, device)
    else:
        raise ValueError(
            f"Unsupported model file extension {extension}. Please try a supported model type."
        )

    return canonicalize_state_dict(state_dict)


def _load_pth(path: str | Path, device: Optional[torch.device] = None) -> StateDict:
    return torch.load(
        f=path,
        map_location=device,
        pickle_module=RestrictedUnpickle,
    )


def _load_torchscript(
    path: str | Path, device: Optional[torch.device] = None
) -> StateDict:
    return torch.jit.load(path, map_location=device).state_dict()


def _load_safetensors(
    path: str | Path, device: Optional[TorchDevice] = None
) -> StateDict:
    if device is not None:
        if isinstance(device, torch.device):
            device = str(device)
        return safetensors_load_file(path, device=device)
    else:
        return safetensors_load_file(path)


def read_safetensors_metadata(file_path: str | Path) -> dict[str, Any]:
    if not str(file_path).endswith(".safetensors"):
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


def find_component_models(
    state_dict: StateDict,
    metadata: Optional[dict] = None,
    registry: dict[str, Type[Architecture]] = ARCHITECTURES,
) -> dict[str, ComponentMetadata]:
    """
    Detect all models present inside of a state dict, and return a dict. The keys of
    the dict are the architecture's unique identifier that can be instantiated using
    this state-dict, and the value is the metadata of the corresponding architecture
    if it were instantiated using this same state-dict + metadata.
    """
    components: dict[str, ComponentMetadata] = {}

    for arch_id, architecture in registry.items():  # Iterate through all architectures
        try:
            checkpoint_metadata = architecture.detect(
                state_dict=state_dict, metadata=metadata
            )

            if checkpoint_metadata is not None:
                # this will overwrite previous architectures with the same id
                components.update({arch_id: checkpoint_metadata})
        except Exception as e:
            print(f"Encountered error running architecture.detect for {arch_id}: {e}")

    return components

from __future__ import annotations

import os
from pathlib import Path
import torch
from typing import Type, Optional
from safetensors.torch import load_file as safetensors_load_file
from spandrel import canonicalize_state_dict
from spandrel.__helpers.unpickler import (
    RestrictedUnpickle,
)  # probably shouldn't import from private modules...

from ..base_types import Architecture, StateDict, TorchDevice
from ..globals import ARCHITECTURES

from typing import Protocol, runtime_checkable


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
    """
    Loads a model from a file path. It detects the architecture, instantiates the
    architecture, and loads the state dict into the PyTorch class.

    Throws a `ValueError` if the file extension is not supported.
    Returns an empty dictionary if no supported model architecture is found.
    """
    state_dict = state_dict_from_file(path, device=device)

    return from_state_dict(state_dict, device, registry)


def from_state_dict(
    state_dict: StateDict,
    device: Optional[TorchDevice] = None,
    registry: dict[str, Type[Architecture]] = ARCHITECTURES,
) -> dict[str, Architecture]:
    """
    Load a model from the given state dict.

    Returns an empty dictionary if no supported model architecture is found.
    """
    components = detect_all(state_dict, registry)

    for arch_id, architecture in components.items():
        try:
            # load state dict into the architecture and moves it to the specified device
            architecture.load(state_dict, device)
        except Exception as e:
            print(e)

    return components


def detect_all(
    state_dict: StateDict, registry: dict[str, Type[Architecture]] = ARCHITECTURES
) -> dict[str, Architecture]:
    """
    Detect all models present inside of a state dict; does not load them into memory however;
    it merely returns the Architecture-Definitions for these models.
    """
    components: dict[str, Architecture] = {}

    for arch_id, architecture in registry.items():  # Iterate through all architectures
        try:
            if architecture.detect(state_dict):
                # this will overwrite previous architectures with the same id
                components.update({arch_id: architecture()}) # type: ignore
        except Exception as e:
            print(e)

    return components


def state_dict_from_file(path: str | Path, device: Optional[TorchDevice] = None) -> StateDict:
    """
    Load the state dict of a model from the given file path.

    State dicts are typically only useful to pass them into the `load`
    function of a specific architecture.

    Throws a `ValueError` if the file extension is not supported.
    """
    extension = os.path.splitext(path)[1].lower()
    if isinstance(device, str):
        device = torch.device(device) # make pyright type-checker happy

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


def _load_torchscript(path: str | Path, device: Optional[torch.device] = None) -> StateDict:
    return torch.jit.load(path, map_location=device).state_dict()


def _load_safetensors(path: str | Path, device: Optional[TorchDevice] = None) -> StateDict:
    if device is not None:
        if isinstance(device, torch.device):
            device = str(device)
        return safetensors_load_file(path, device=device)
    else:
        return safetensors_load_file(path)

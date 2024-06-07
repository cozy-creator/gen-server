from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence, Optional, Any
import importlib
import pkg_resources

import torch

from .canonicalize import canonicalize_state_dict
from spandrel import ArchId, Architecture, ModelDescriptor, StateDict

import logging


class UnsupportedModelError(Exception):
    """
    An error that will be thrown by `ArchRegistry` and `ModelLoader` if a model architecture is not supported.
    """


# TO DO: get rid of 'before' functionality. 
# Figure out ids to make model architecture-definitions unique
# Resolve conflicting architecture definitions
@dataclass(frozen=True)
class ArchSupport:
    """
    An entry in an `ArchRegistry` that describes how to detect and load a model architecture.
    """

    architecture: Architecture[torch.nn.Module]
    """
    The architecture.
    """
    
    detect: Callable[[StateDict], bool]
    """
    Inspects the given state dict and returns True if this architecture is detected.

    For most architectures, this will be the architecture's `detect` method.
    """

    @staticmethod
    def from_architecture(arch: Architecture[torch.nn.Module]) -> ArchSupport:
        """
        Creates an `ArchSupport` from an `Architecture` by using the architecture's ``detect`` method.
        """
        return ArchSupport(arch, arch.detect)


class ArchRegistry:
    """
    A registry of architectures.

    Architectures are detected/loaded in insertion order unless `before` is specified.
    """

    def __init__(self, namespace: str):
        self._namespace = namespace
        # the registry is copy on write internally
        self._architectures: Sequence[ArchSupport] = []
        self._by_id: Mapping[ArchId, ArchSupport] = {}
        self._load_architectures()

    def _load_architectures(self):
        print("Loading architectures")
        for entry_point in pkg_resources.iter_entry_points(group=f"comfy_creator.architectures"):
            try:
                module = importlib.import_module(entry_point.module_name)
                print(entry_point.attrs[0] )
                arch_class_name = entry_point.attrs[0] 
                arch_class = getattr(module, arch_class_name)
                if issubclass(arch_class, Architecture):
                    print(f"Loading architecture {entry_point.attrs[0]}")
                    arch_support = ArchSupport.from_architecture(arch_class())
                    self.add(arch_support)
            except Exception as e:
                print(f"Error loading architecture {entry_point.name}: {e}")

    def copy(self) -> ArchRegistry:
        """
        Returns a copy of the registry.
        """
        new = ArchRegistry(self._namespace)
        new._architectures = self._architectures
        new._by_id = self._by_id
        return new

    def __contains__(self, id: ArchId | str) -> bool:
        return id in self._by_id

    def __getitem__(self, id: str | ArchId) -> ArchSupport:
        return self._by_id[ArchId(id)]

    def __iter__(self):
        """
        Returns an iterator over all architectures in insertion order.
        """
        return iter(self.architectures("insertion"))

    def __len__(self) -> int:
        return len(self._architectures)

    def get(self, id: str | ArchId) -> ArchSupport | None:
        return self._by_id.get(ArchId(id), None)

    def architectures(
        self
    ) -> list[ArchSupport]:
        """
        Returns a new list with all architectures in the registry.

        The order of architectures in the list is either insertion order or the order in which architectures are detected.
        """
        return list(self._architectures)

    def add(self, *architectures: ArchSupport):
        """
        Adds the given architectures to the registry.

        Throws an error if an architecture with the same ID already exists.
        Throws an error if a circular dependency of `before` references is detected.

        If an error is thrown, the registry is left unchanged.
        """

        new_architectures = list(self._architectures)
        new_by_id = dict(self._by_id)
        for arch in architectures:
            if arch.architecture.id in new_by_id:
                logging.warning(f"Duplicate architecture: {arch.architecture.id}")
            else:
                new_architectures.append(arch)
                new_by_id[arch.architecture.id] = arch

        self._architectures = new_architectures
        self._by_id = new_by_id
    
    # TO DO: support multiple models
    def load(self, state_dict: StateDict) -> ModelDescriptor:
        """
        Detects the architecture of the given state dict and loads it.

        This will canonicalize the state dict if it isn't already.

        Throws an `UnsupportedModelError` if the model architecture is not supported.
        """

        state_dict = canonicalize_state_dict(state_dict)

        for arch in self._architectures:
            if arch.detect(state_dict):
                return arch.architecture.load(state_dict)

        raise UnsupportedModelError

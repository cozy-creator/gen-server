from typing import Dict
from .architecture import Architecture


class Checkpoint:
    """
    This comfy-creator-specific metadata for a pretrained model / checkpoint file
    Does PyTorch already have something along these lines?
    """

    def __init__(
        self,
        display_name: str,
        components: Dict[str, Architecture] = None,
        metadata: Dict = None,
    ):
        if components is None:
            components = {}
        self._components: Dict[str, Architecture] = components
        self._display_name = display_name
        """
        The display name of the checkpoint. This is the name that will be shown in the UI.
        """
        self.metadata = metadata
        """
        Additional metadata for the checkpoint.
        """

    def add_component(self, name: str, component: Architecture):
        """
        Adds a component to the checkpoint.
        """
        if name in self._components:
            raise ValueError(f"Component with name '{name}' already exists.")
        self._components[name] = component

    @property
    def components(self) -> Dict[str, Architecture]:
        return self._components

    @property
    def display_name(self) -> str:
        return self._display_name

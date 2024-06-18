from typing import Dict
from .architecture import Architecture
from .. import Serializable


class Checkpoint(Serializable):
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

        if metadata is None:
            metadata = {}
        self._date = metadata.get("date")
        self._format = metadata.get("format")
        self._author = metadata.get("author")

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

    @property
    def date(self) -> str:
        return self._date

    @property
    def author(self) -> str:
        return self._author

    @property
    def format(self) -> str:
        return self._format

    def serialize(self) -> Dict:
        return {
            "date": self.date,
            "author": self.author,
            "format": self.format,
            "display_name": self.display_name,
        }

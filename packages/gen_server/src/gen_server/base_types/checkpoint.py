from typing import Dict
from .architecture import Architecture

class Checkpoint:
    """This comfy-creator-specific metadata for a pretrained model / checkpoint file
    Does PyTorch already have something along these lines?
    """
    def __init__(self):
        self._components: Dict[str, Architecture] = {}

    @property
    def components(self) -> Dict[str, Architecture]:
        return self._components

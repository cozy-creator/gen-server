from abc import ABC
import torch
from typing import Union, List
import PIL.Image
import numpy as np
from enum import Enum


class Language(Enum):
    """
    ISO 639-1 language codes; used for localizing text.
    English will be displayed for all text lacking a localization.
    """

    ENGLISH = "en"
    CHINESE = "zh"


class Category(Enum):
    """
    Used to group nodes by category in the client.
    """

    LOADER = {Language.ENGLISH: "Loader", Language.CHINESE: "加载器"}
    PIPE = {Language.ENGLISH: "Pipe", Language.CHINESE: "管道"}


StateDict = dict[str, torch.Tensor]
"""
The parameters of a PyTorch model, serialized as a flat dict.
"""

TorchDevice = Union[str, torch.device]
"""
A string like 'cpu', 'cuda', or 'mps' or a torch device object.
"""

ImageOutputType = Union[List[PIL.Image.Image], np.ndarray]
"""
Static typing for image outputs
"""


class Serializable(ABC):
    """
    Base class for serializable objects
    """

    def serialize(self):
        """
        Serialize the object into a dictionary
        :return: serialized object
        """
        result = {}
        for attr in self.__dir__():
            if not attr.startswith("_") and not callable(self.__getattribute__(attr)):
                result[attr] = self.__getattribute__(attr)
        return result

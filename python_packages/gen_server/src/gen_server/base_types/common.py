import torch
from typing import Union, List, Callable, Any
import PIL.Image
import numpy as np
from enum import Enum
from multiprocessing.connection import Connection


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
    UPSCALER = {Language.ENGLISH: "Upscaler", Language.CHINESE: "升频器"}
    MASK = {Language.ENGLISH: "Mask"}
    INPAINTING = {Language.ENGLISH: "Inpainting"}
    IMAGES = {Language.ENGLISH: "Images"}


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

Validator = Callable[[Any], bool]

JobQueueItem = tuple[dict[str, Any], Connection]
"""
Type of items on the job-queue
"""

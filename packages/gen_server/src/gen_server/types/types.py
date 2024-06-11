import torch
from typing import Union, List
import PIL.Image
import numpy as np


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

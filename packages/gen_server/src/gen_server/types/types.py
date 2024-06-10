import torch
from typing import Union


StateDict = dict[str, torch.Tensor]
"""
The parameters of a PyTorch model, serialized as a flat dict.
"""

TorchDevice = Union[str, torch.device]
"""
A string like 'cpu', 'cuda', or 'mps' or a torch device object.
"""
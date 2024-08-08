from typing import Optional
import torch


def get_torch_device(index: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", index)
    if torch.backends.mps.is_available():
        return torch.device("mps", index)
    if torch.xpu.is_available():
        return torch.device("xpu", index)
    return torch.device("cpu")


def get_torch_device_count() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if torch.backends.mps.is_available():
        return torch.mps.device_count()
    if torch.xpu.is_available():
        return torch.xpu.device_count()
    return 1

import logging
import torch

logger = logging.getLogger(__name__)


def get_torch_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", index)
    if torch.backends.mps.is_available():
        return torch.device("mps", index)
    if torch.xpu.is_available():
        return torch.device("xpu", index)

    logger.warning("No device found, using CPU. This will slow down performance.")
    return torch.device("cpu")


def get_torch_device_count() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if torch.backends.mps.is_available():
        return torch.mps.device_count()
    if torch.xpu.is_available():
        return torch.xpu.device_count()
    return 1

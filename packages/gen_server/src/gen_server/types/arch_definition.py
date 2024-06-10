from abc import abstractmethod
from typing import Any, Dict, Protocol, runtime_checkable, Generic, TypeVar
import torch
from .model_wrapper import ModelWrapper
from .types import TorchDevice, StateDict

T = TypeVar('T', bound=torch.nn.Module, covariant=True)


# TO DO: in the future, maybe we can compare sets of keys, rather than use
# a detect method?
@runtime_checkable
class ArchDefinition(Protocol, Generic[T]):
    """
    The interface that all architecture definitions should implement.
    """

    @classmethod
    def detect(cls, state_dict: StateDict) -> bool:
        """
        Detects whether the given state dictionary matches the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.

        Returns:
            bool: True if the state dictionary matches the architecture, False otherwise.
        """
        pass

    @classmethod
    def load(cls, state_dict: StateDict, device: TorchDevice = None) -> ModelWrapper[T]:
        """
        Loads a model from the given state dictionary according to the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """
        pass

from abc import abstractmethod
from typing import Any, Dict, Protocol, runtime_checkable
import torch
from .model_wrapper import ModelWrapper

StateDict = dict[str, torch.Tensor]


# TO DO: in the future, maybe we can compare sets of keys, rather than use
# a detect method?
@runtime_checkable
class ArchDefinition(Protocol):
    """
    The interface that all architecture definitions should implement.
    """

    @abstractmethod
    def detect(self, state_dict: StateDict) -> bool:
        """
        Detects whether the given state dictionary matches the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.

        Returns:
            bool: True if the state dictionary matches the architecture, False otherwise.
        """
        pass

    @abstractmethod
    def load(self, state_dict: StateDict) -> ModelWrapper:
        """
        Loads a model from the given state dictionary according to the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """
        pass

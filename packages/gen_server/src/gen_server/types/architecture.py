from abc import abstractmethod
from typing import Any, Dict, Protocol, runtime_checkable, Generic, TypeVar
from abc import ABC, abstractmethod
import torch
from .types import TorchDevice, StateDict

T = TypeVar('T', bound=torch.nn.Module, covariant=True)


# TO DO: in the future, maybe we can compare sets of keys, rather than use
# a detect method?
class Architecture(ABC, Generic[T]):
    """
    The interface that all architecture definitions should implement.
    The construct __init__ function should accept no arguments.
    
    A wrapper class for PyTorch models that adds additional properties and methods
    for inspection and management of the model.
    """
    
    def __init__(
        self,
        model: T,
        config: Any = None,
        input_space: str = None,
        output_space: str = None
    ) -> None:
        super().__init__()
        
        self._model = model
        self._config = config
        self._input_space = input_space
        self._output_space = output_space
    
    @property
    def model(self) -> T:
        """
        Access the underlying model.
        """
        return self._model
    
    @property
    def config(self) -> Any:
        """
        Access the underlying config
        """
        return self._config
    
    @property
    def input_space(self) -> str:
        """
        Access the input space of the model.
        """
        return self._input_space

    @property
    def output_space(self) -> str:
        """
        Access the output space of the model.
        """
        return self._output_space
    
    @classmethod
    @abstractmethod
    def detect(cls, state_dict: StateDict) -> bool:
        """
        Detects whether the given state dictionary matches the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.

        Returns:
            bool: True if the state dictionary matches the architecture, False otherwise.
        """
        pass
    
    @abstractmethod
    def load(self, state_dict: StateDict, device: TorchDevice = None) -> None:
        """
        Loads a model from the given state dictionary according to the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """
        pass
    
    # def __repr__(self) -> str:
    #     """
    #     String representation of the ModelWrapper including the type of the wrapped model
    #     """
    #     return f"<ModelWrapper for {self._model.__class__.__name__} with {self.stuff}>"

from typing import Any, Generic, TypeVar, Optional
from abc import ABC, abstractmethod
import torch
from .common import TorchDevice, StateDict

T = TypeVar("T", bound=torch.nn.Module, covariant=True)


# TO DO: in the future, maybe we can compare sets of keys, rather than use
# a detect method?
class Architecture(ABC, Generic[T]):
    """
    The interface that all architecture definitions should implement.
    The construct __init__ function should accept no arguments.

    A wrapper class for PyTorch models that adds additional properties and methods
    for inspection and management of the model.
    """
    display_name: str
    input_space: str
    output_space: str

    def __init__(
        self,
        model: T,
        config: Any = None,
    ) -> None:
        super().__init__()

        self._model = model
        self._config = config

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
    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None) -> None:
        """
        Loads a model from the given state dictionary according to the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """
        pass
    
    @classmethod
    def serialize(cls) -> dict[str, Any]:
        """
        Serialize the Architecture instance to a dictionary.
        """
        return {
            'display_name': cls.display_name,
            'input_space': cls.input_space,
            'output_space': cls.output_space,
        }

    # def __repr__(self) -> str:
    #     """
    #     String representation of the ModelWrapper including the type of the wrapped model
    #     """
    #     return f"<ModelWrapper for {self._model.__class__.__name__} with {self.stuff}>"

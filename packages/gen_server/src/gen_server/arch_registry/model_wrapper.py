from typing import Generic, TypeVar
from abc import ABC
import torch

T = TypeVar('T', bound=torch.nn.Module, covariant=True)


class ModelWrapper(ABC, Generic[T]):
    """
    A wrapper class for PyTorch models that adds additional properties and methods
    for inspection and management of the model.
    """
    def __init__(self, model: T):
        self._model = model

    @property
    def model(self) -> T:
        """
        Access the underlying model.
        """
        return self._model

    @property
    def stuff(self) -> int:
        """
        Example of adding a new property
        """
        return 0

    def __repr__(self) -> str:
        """
        String representation of the ModelWrapper including the type of the wrapped model
        """
        return f"<ModelWrapper for {self._model.__class__.__name__} with {self.stuff}>"

from abc import abstractmethod, ABC
from typing import Any, Generic, TypeVar, Optional, Literal

import torch
from spandrel import Architecture as SpandrelArchitecture, ModelDescriptor
from zope.interface import Interface, implementer, Attribute

from .common import StateDict, TorchDevice


T = TypeVar("T", bound=torch.nn.Module, covariant=True)

ArchIntent = Literal["Generation", "Restoration", "SR"]


class IArchitecture(Interface):
    display_name = Attribute(
        "The name of the architecture.",
    )

    input_space = Attribute(
        "The architecture's input space.",
    )

    output_space = Attribute(
        "The architecture's output space.",
    )

    model = Attribute(
        "The underlying PyTorch model.",
    )

    config = Attribute(
        "The configuration of the architecture.",
    )

    def __init__(model=None, config: Any = None) -> None:
        pass

    def load(state_dict: StateDict, device=None) -> None:
        pass

    def detect(state_dict) -> bool:
        pass


# TO DO: in the future, maybe we can compare sets of keys, rather than use
# a detect method?
@implementer(IArchitecture)
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
        model: Optional[T],
        config: Any = None,
    ) -> None:
        super().__init__()

        self._model = model
        self._config = config

    @property
    def model(self) -> Optional[T]:
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
            device: The device the loaded model is sent to.

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


class SpandrelArchitectureAdapter(Architecture):
    def __init__(self, arch: SpandrelArchitecture):
        super().__init__(model=None, config=None)
        if not isinstance(arch, SpandrelArchitecture):
            raise TypeError("'arch' must be an instance of spandrel Architecture")

        self.inner = arch
        self.display_name = self.inner.name

    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None) -> None:
        descriptor = self.inner.load(state_dict)
        if not isinstance(descriptor, ModelDescriptor):
            raise TypeError("descriptor must be an instance of ModelDescriptor")

        self._model = descriptor.model
        if device is not None:
            self._model.to(device)
        elif descriptor.supports_half:
            self._model.to(torch.float16)
        elif descriptor.supports_bfloat16:
            self._model.to(torch.bfloat16)
        else:
            raise Exception("Device not provided and could not be inferred")

    def detect(self, state_dict: StateDict) -> bool:
        return self.inner.detect(state_dict)

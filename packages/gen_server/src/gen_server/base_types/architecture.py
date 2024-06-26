import torch
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Optional, TypedDict, Generic
from spandrel import Architecture as SpandrelArchitecture, ImageModelDescriptor
from .common import StateDict, TorchDevice

T = TypeVar("T", bound=torch.nn.Module, covariant=True)

ComponentMetadata = TypedDict(
    "ComponentMetadata", {"display_name": str, "input_space": str, "output_space": str}
)


# TO DO: in the future, maybe we can compare sets of keys, rather than use
# a detect method? That might be more optimized.


class Architecture(ABC, Generic[T]):
    """
    The abstract-base-class that all comfy-creator Architectures should implement.
    """

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def input_space(self) -> str:
        return self._input_space

    @property
    def output_space(self) -> str:
        return self._output_space

    @property
    def model(self) -> T:
        """Access the underlying PyTorch model."""
        return self._model  # type: ignore

    @property
    def config(self) -> Any:
        return self._config

    def __init__(
        self,
        *,
        state_dict: Optional[StateDict] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Constructor signature should look like this, although this abstract-base
        class does not (and cannot) enforce your constructor signature.
        """
        self._display_name = "default"
        self._input_space = "default"
        self._output_space = "default"
        self._config = {}
        pass

    @classmethod
    @abstractmethod
    def detect(
        cls,
        *,
        state_dict: Optional[StateDict] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[ComponentMetadata]:
        """
        Detects whether the given state dictionary matches the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.
            metadata (dict[str, Any]): optional additional metadata to help identify the model

        Returns:
            bool: True if the state dictionary matches the architecture, False otherwise.
        """
        pass

    @abstractmethod
    def load(
        self,
        state_dict: StateDict,
        device: Optional[TorchDevice] = None,
    ) -> None:
        """
        Loads a model from the given state dictionary according to the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.
            device: The device the loaded model is sent to.
        """
        pass

    # @classmethod
    # def __subclasshook__(cls, subclass):
    #     """ Used by `issubclass` to validate that the implementation of is correct. """
    #     required_methods = {
    #         'display_name': 'property',
    #         'input_space': 'property',
    #         'output_space': 'property',
    #         'model': 'property',
    #         'config': 'property',
    #         'detect': 'classmethod',
    #         'load': 'method'
    #     }

    #     for method, method_type in required_methods.items():
    #         if not any(method in B.__dict__ for B in subclass.__mro__):
    #             print(f"Missing implementation of {method} in {subclass.__name__}")
    #             return False
    #         if method_type == 'classmethod' and not isinstance(getattr(subclass, method), classmethod):
    #             print(f"{method} in {subclass.__name__} must be a class method")
    #             return False
    #         if method_type == 'property' and not isinstance(getattr(subclass, method), property):
    #             print(f"{method} in {subclass.__name__} must be a property")
    #             return False
    #         if method_type == 'method' and callable(getattr(subclass, method)):
    #             # Check if it's a regular method (not classmethod or staticmethod)
    #             if isinstance(getattr(subclass, method), (staticmethod, classmethod)):
    #                 print(f"{method} in {subclass.__name__} must be a regular method")
    #                 return False
    #     return True

    # def serialize(self) -> dict[str, Any]:
    #     """
    #     Serialize the Architecture instance to a dictionary.
    #     """
    #     ...

    # def __repr__(self) -> str:
    #     """
    #     String representation of the ModelWrapper including the type of the wrapped model
    #     """
    #     return f"<ModelWrapper for {self._model.__class__.__name__} with {self.stuff}>"


class SpandrelArchitectureAdapter(Architecture):
    """
    This class converts architectures from the spandrel library to our own
    Architecture interface.
    """

    def __init__(self, arch: SpandrelArchitecture):
        super().__init__()
        if not isinstance(arch, SpandrelArchitecture):
            raise TypeError("'arch' must be an instance of spandrel Architecture")

        self.inner = arch
        self._model = None
        self._display_name = self.inner.name

    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None) -> None:
        descriptor = self.inner.load(state_dict)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise TypeError("descriptor must be an instance of ImageModelDescriptor")

        self._model = descriptor.model
        if device is not None:
            self._model.to(device)
        elif descriptor.supports_half:
            self._model.to(torch.float16)
        elif descriptor.supports_bfloat16:
            self._model.to(torch.bfloat16)
        else:
            raise Exception("Device not provided and could not be inferred")

    def detect(
        self,
        state_dict: StateDict,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        return self.inner.detect(state_dict)

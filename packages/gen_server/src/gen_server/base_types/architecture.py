import torch
from typing import (
    Any,
    Generic,
    TypeVar,
    Optional,
    Protocol,
    runtime_checkable,
    TypedDict
)
from spandrel import Architecture as SpandrelArchitecture, ModelDescriptor
from .common import StateDict, TorchDevice


# T = TypeVar("T", bound=torch.nn.Module, covariant=True)
T = TypeVar("T", bound=torch.nn.Module)

ComponentMetadata = TypedDict('ComponentMetadata', {
    'display_name': str,
    'input_space': str,
    'output_space': str
})

# TO DO: in the future, maybe we can compare sets of keys, rather than use
# a detect method? That might be more optimized.

@runtime_checkable
class Architecture(Protocol[T]):
    """
    The interface that all comfy-creator Architectures should implement.
    """
    display_name: str
    input_space: str
    output_space: str
    model: T
    config: Any
    
    # @property
    # def config(self) -> dict[str, Any]:
    #     """
    #     Access the configuration dictionary for the architecture.
    #     """
    #     return self._config

    # @property
    # def model(self) -> T:
    #     """
    #     Access the underlying PyTorch model.
    #     """
    #     return self._model
    
    def __init__(self,
        state_dict: Optional[StateDict] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        ...
    
    @classmethod
    def detect(cls,
        state_dict: Optional[StateDict] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> Optional[ComponentMetadata]:
        """
        Detects whether the given state dictionary matches the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.

        Returns:
            bool: True if the state dictionary matches the architecture, False otherwise.
        """
        ...
    
    def load(self,
        state_dict: StateDict,
        device: Optional[TorchDevice] = None
    ) -> None:
        """
        Loads a model from the given state dictionary according to the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.
            device: The device the loaded model is sent to.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """
        ...
    
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

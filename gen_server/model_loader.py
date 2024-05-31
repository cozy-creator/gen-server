import torch
from typing import runtime_checkable, Dict, List, Optional, Tuple, Union, Any, BinaryIO, Protocol, TypedDict, Set, Generic, TypeVar
from abc import ABC

# T = TypeVar("T", bound=torch.nn.Module, covariant=True)
# class Architecture(ABC, Generic[T]):

class Architecture(Protocol):
    """
    Interface for all architectures.
    """

    @property
    def required_keys(self) -> Set[str]:
        """
        something
        """
        pass

    @property
    def name(self) -> str:
        """
        The name of the architecture.

        This is often the same as `id`.
        """
        return self._name

    def detect(self, state_dict: StateDict) -> bool:
        """
        Inspects the given state dict and returns ``True`` if it is a state dict of this architecture.

        This guarantees that there are no false negatives, but there might be false positives.
        This is important to remember when ordering architectures in a registry.

        (Note: while false positives are allowed, they are supposed to be rare. So we do accept bug reports for false positives.)
        """
        return self._detect(state_dict)

    @abstractmethod
    def load(
        self, state_dict: StateDict
    ) -> ImageModelDescriptor[T] | MaskedImageModelDescriptor[T]:
        """
        Loads the given state dict into a model. The hyperparameters will automatically be deduced.

        The state dict is assumed to be a state dict of this architecture, meaning that `detect` returned `True` for the state dict.
        If this is not the case, then the behavior of this function is unspecified (the model may be loaded incorrect or an error is thrown).
        """


@runtime_checkable
class ModelDefinition(Protocol):
    """
    Something
    """
    
    def required_keys() -> Set[str]:
        pass
    
    def __call__(self, *args, **kwargs):
        """
        Method that must be implemented to allow the node to be called as a function.
        """
        ...


def identify_model_type(metadata):
    # Dictionary of model types and their required keys as sets
    model_requirements = {
        "controlnet": {"key1", "key2", "key3"},  # Example keys for controlnet
        "conditioner": {"keyA", "keyB", "keyC"},  # Example keys for conditioner
        "cond_stage_model": {"keyX", "keyY", "keyZ"},  # Example keys for cond_stage_model
        "lora": {"keyL", "keyM", "keyN"}  # Example keys for lora
    }

    # Convert the metadata keys to a set for efficient comparison
    metadata_keys_set = set(metadata.keys())

    # Check each model type's required keys against the metadata keys
    for model_type, required_keys in model_requirements.items():
        if required_keys <= metadata_keys_set:
            return model_type

    return "Unknown"

from .architecture import Architecture
from typing import Type, Any


class ModelConstraint:
    """
    A constraint on a model input. This can be introspected, for display in the UI, and also
    called in order to validate inputs.
    
    Constraints are used when building a graph, not when running a graph, hence they should
    considered the equivalent of static type checking, not runtime type checking.
    
    If a constraint (type, input_space, or output_space) is None, that constraint is ignored.
    """
    def __init__(self, *, model_type: Type[Any] = None, input_space: str = None, output_space: str = None):
        self._model_type = model_type
        self._input_space = input_space
        self._output_space = output_space

    @property
    def model_type(self):
        return self._model_type

    @property
    def input_space(self):
        return self._input_space

    @property
    def output_space(self):
        return self._output_space
    
    def __call__(self, model_wrapper: Architecture) -> bool:
        """
        Checks if the specified architecture passes all constraints or not.
        """
        if self._model_type is not None and not isinstance(model_wrapper, self._model_type):
            return False

        if self._input_space is not None and getattr(model_wrapper, 'input_space', None) != self.input_space:
            return False

        if self._output_space is not None and getattr(model_wrapper, 'output_space', None) != self.output_space:
            return False

        return True

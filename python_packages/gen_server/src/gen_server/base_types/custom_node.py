from typing import TypedDict, Any
from abc import ABC, abstractmethod
from .common import Language, Category
import os
import inspect
import json
from typing import Dict, Any, Union


class NodeInterfaceInput(TypedDict):
    name: str
    type: str
    default: str
    required: bool


class NodeInterfaceOutput(TypedDict):
    name: str
    type: str


class NodeInterface(TypedDict):
    inputs: dict[str, Any]
    outputs: dict[str, Any]


# TODO: might be useful someday?
class InputWrapper:
    pass


class OutputWrapper:
    pass

# For now, I'm making the node-interface dependent ONLY upon the inputs of
# the node. If it were to change based on the outputs as well (i.e., other
# nodes can request new output-types) that means finding a stable network
# would become HARD, because instead of passing from input to output we
# instead could flow _backwards_ and end up in infinite loops of modifying
# interfaces infinitely.
class CustomNode(ABC):
    """
    The interface that all custom-nodes should implement.
    """

    display_name: dict[Language, str]
    """ The name of the node, displayed in the client. """

    category: Category
    """
    Category used to group nodes in the client.
    """

    description: dict[Language, str]
    """ Description, displayed in the client. Localized by language. """

    @classmethod
    def get_spec(cls) -> Dict[str, Any]:
        """Returns the node specification."""
        class_file = os.path.abspath(inspect.getfile(cls))
        spec_file = os.path.join(os.path.dirname(class_file), f"{cls.__name__}.json")
        with open(spec_file, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return spec

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Union[dict[str, Any]]:
        """
        Runs the node.
        """
        pass


def custom_node_validator(plugin: Any) -> bool:
    try:
        return issubclass(plugin, CustomNode)

    except TypeError:
        print(f"Invalid plugin type: {plugin}")
        return False

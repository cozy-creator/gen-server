from typing import TypedDict, Any, Protocol, runtime_checkable
from typing_extensions import runtime
from .common import Language, Category


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


# class ICustomNode(Interface):
#     display_name = Attribute(
#         "The name of the node.",
#     )

#     category = Attribute(
#         "The category of the node.",
#     )

#     description = Attribute(
#         "The description of the node.",
#     )

#     def update_interface(inputs: dict[str, Any]) -> NodeInterface:
#         pass

#     def __call__(*args, **kwargs) -> Any:
#         pass


# For now, I'm making the node-interface dependent ONLY upon the inputs of
# the node. If it were to change based on the outputs as well (i.e., other
# nodes can request new output-types) that means finding a stable network
# would become HARD, because instead of passing from input to output we
# instead could flow _backwards_ and end up in infinite loops of modifying
# interfaces infinitely.
@runtime_checkable
class CustomNode(Protocol):
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

    @staticmethod
    def update_interface(inputs: dict[str, Any]) -> NodeInterface:
        """
        Updates the node's interface based on the inputs.
        """
        ...

    def __call__(self, *args, **kwargs) -> Any:
        """
        Runs the node.
        """
        ...

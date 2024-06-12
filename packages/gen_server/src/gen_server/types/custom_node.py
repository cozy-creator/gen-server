from typing import Protocol, TypedDict, Any


class NodeInterface(TypedDict):
    inputs: dict[str, Any]
    outputs: dict[str, Any]


# TODO: might be useful someday?
class InputWrapper:
    pass

class OuputWrapper:
    pass


# For now, I'm making the node-interface dependent ONLY upon the inputs of
# the node. If it were to change based on the outputs as well (i.e., other
# nodes can request new output-types) that means finding a stable network
# would become HARD, because instead of passing from input to output we
# instead could flow _backwards_ and end up in infinite loops of modifying
# interfaces infinitely.
class CustomNode(Protocol):
    """
    The interface that all custom-nodes should implement.
    """
    @staticmethod
    def update_interface(inputs: dict[str, Any] = None) -> NodeInterface:
        """
        Updates the node's interface based on the inputs.
        """
        pass
    
    def __call__(self, *args, **kwargs) -> Any:
        """
        Runs the node.
        """
        pass
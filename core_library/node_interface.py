# Note that 'named tensors' are a thing in pytorch; we should use them when possible

from typing import runtime_checkable, Dict, List, Optional, Tuple, Union, Any, BinaryIO, Protocol, TypedDict


class InputSpec(TypedDict):
    display_name: str
    edge_type: str
    spec: Dict[str, Any]
    required: bool

    
class OutputSpec(TypedDict):
    display_name: str
    edge_type: str


@runtime_checkable
class NodeInterface(Protocol):
    """
    Protocol defining the interface for all node types.
    All nodes should conform to this protocol.
    """
    
    def __call__(self, *args, **kwargs):
        """
        Method that must be implemented to allow the node to be called as a function.
        """
        ...
    
    @classmethod
    def INPUT_TYPES(cls) -> Union[Dict[str, Dict[str, Union[Tuple[str], Tuple[str, Dict]]]], TypedDict[str, InputSpec]]:
        """
        Class method that should return a dictionary specifying the input types for the node.
        """
        ...
        
    @property
    def RETURN_TYPES(self) -> Union[Tuple[str, ...], TypedDict[str, OutputSpec]]:
        """
        Specifies the return type (output) of the node. Legacy ComfyUI uses a tuple of strings, while modern
        Comfy Creator uses a dict, so that return-types can be specified by name rather than position.
        """
        
    @property
    def CATEGORY(self) -> str:
        """
        Category of the node displayed in the graph editor, such as 'advanced/conditioning'.
        Used by the graph-editor to categorize nodes.
        """
        ...
        
    @property
    def display_name(self) -> Optional[TypedDict[str, str]]:
        """
        A dictionary, where the keys are ISO 639-1 language-codes (such as 'en', 'zh', or 'ja') and the value is
        the localized display-name of the node for that language.
        """
        return None
    
    @property
    def description(self) -> Optional[TypedDict[str, str]]:
        """
        A dictionary, where the keys are ISO 639-1 language-codes (such as 'en', 'zh', or 'ja') and the value is
        the localized description of the node for that language.
        """
        return None


# Option 1: manually set the function's inspect signature:

# import inspect
# from typing import Any, Dict

# def create_parameter(name: str, annotation: Any, default=inspect.Parameter.empty, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, metadata: Dict[str, Any] = None):
#     return inspect.Parameter(name, kind, default=default, annotation=annotation, metadata=metadata)

# def add_node(a: int, b: int) -> int:
#     return a + b

# params = [
#     create_parameter("a", int, metadata={"display_name": "First Operand", "required": True, "description": "The first integer"}),
#     create_parameter("b", int, metadata={"display_name": "Second Operand", "required": True, "description": "The second integer"})
# ]

# sig = inspect.Signature(parameters=params, return_annotation=int)
# add_node.__signature__ = sig

# # Now you can inspect `add_node` as before
# print(inspect.signature(add_node))


# Option 2: set the function's signature as a class

# from typing import Any, Dict, Type

# class ParamSpec:
#     def __init__(self, display_name: str, type: Type, required: bool, metadata: Dict[str, Any]):
#         self.display_name = display_name
#         self.type = type
#         self.required = required
#         self.metadata = metadata

# class NodeInterface:
#     def __call__(self, *args: Any, **kwargs: Any) -> Any:
#         raise NotImplementedError("Each node must implement the '__call__' method.")

# class AddNode(NodeInterface):
#     def __call__(self, a: ParamSpec("First Operand", int, True, {"description": "The first integer"}),
#                         b: ParamSpec("Second Operand", int, True, {"description": "The second integer"})) -> int:
#         return a + b


# Option 3: more like how Comfy UI works; the input and output args are specified
# by functions or properties in the class itself

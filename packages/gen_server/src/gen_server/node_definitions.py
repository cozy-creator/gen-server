import json
import importlib
from typing import List, Dict, Any
from pkg_resources import iter_entry_points
import inspect
from gen_server.base_types import ModelConstraint, Category, Language
from .globals import _CUSTOM_NODES


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj): # type: ignore
        if isinstance(obj, property):
            if callable(obj.fget):
                return obj.fget(None)
            else:
                return obj.fget
        return super().default(obj)


# Serialization functions
def serialize_model_constraint(obj: Any) -> Dict[str, Any]:
    return {
        "type": "ModelConstraint",
        "model_type": str(obj.model_type),
    }

def serialize(obj: Any) -> str:
    return str(obj)

def serialize(obj: Any) -> str:
    return str(obj)

def serialize_object(obj: Any) -> Any:
    if isinstance(obj, ModelConstraint):
        return serialize_model_constraint(obj)
    elif isinstance(obj, Category):
        return serialize(obj)
    elif isinstance(obj, Language):
        return serialize(obj)
    elif isinstance(obj, type):
        return str(obj.__name__)
    else:
        return str(obj)


def get_custom_nodes_from_entry_points(group: str) -> List[str]:
    nodes = []
    for entry_point in iter_entry_points(group=group):
        nodes.append(entry_point)
    return nodes


def import_module_from_entry_point(entry_point) -> Any: # type: ignore
    return entry_point.load()


def extract_node_metadata(node_class) -> Dict[str, Any]: # type: ignore
    # Check if the 'update_interface' method accepts 'inputs'
    update_interface = node_class.update_interface
    signature = inspect.signature(update_interface)
    
    # If 'inputs' parameter is present, provide a default value
    if 'inputs' in signature.parameters:
        interface = update_interface(inputs={})
    else:
        interface = update_interface()

    serialized_interface = {
        "inputs": {key: serialize_object(value) for key, value in interface["inputs"].items()},
        "outputs": {key: serialize_object(value) for key, value in interface["outputs"].items()}
    }
    
    return {
        "name": node_class.__name__,
        "display_name": serialize_object(getattr(node_class, 'display_name', None)),
        "category": serialize_object(getattr(node_class, 'category', None)),
        "description": serialize_object(getattr(node_class, 'description', None)),
        "interface": serialized_interface
    }


def produce_node_definitions_file(output_file: str):
    entry_point_group = "cozy_creator.custom_nodes"
    custom_nodes = get_custom_nodes_from_entry_points(entry_point_group)
    all_nodes = []

    for entry_point in custom_nodes:
        try:
            node_class = import_module_from_entry_point(entry_point)
            node_metadata = extract_node_metadata(node_class)
            all_nodes.append(node_metadata)
        except Exception as e:
            print(f"Error loading node {entry_point.name}: {e}")

    print(all_nodes)

    with open(output_file, 'w') as f:
        json.dump(all_nodes, f, indent=4, cls=CustomJSONEncoder)



# def compile_node_definitions() -> List[Dict[str, Any]]:
#     entry_point_group = "cozy_creator.custom_nodes"
#     node_definitions = []

#     for entry_point in iter_entry_points(group=entry_point_group):
#         try:
#             node_class = entry_point.load()
#             node_spec = node_class.get_spec()
#             node_definitions.append(node_spec)
#         except Exception as e:
#             print(f"Error loading node {entry_point.name}: {e}")

#     return node_definitions

def save_node_definitions(output_file: str):
    node_definitions = compile_node_definitions()
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(node_definitions, f, ensure_ascii=False, indent=4)


def compile_node_definitions() -> List[Dict[str, Any]]:
    pass
# Example usage
# produce_node_definitions_file('node_definitions.json')

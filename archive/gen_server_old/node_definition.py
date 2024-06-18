import os
import importlib.util
from typing import List, Dict, Any
import json


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, property):
            if callable(obj.fget):
                return obj.fget(None)
            else:
                return obj.fget
        return super().default(obj)


def scan_directory_for_nodes(directory: str) -> List[str]:
    node_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                node_files.append(os.path.join(root, file))
    return node_files

def import_module_from_path(path: str):
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def extract_node_metadata(module) -> List[Dict[str, Any]]:
    nodes = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type):  # Check if it's a class
            if hasattr(attr, 'INPUT_TYPES') and hasattr(attr, 'RETURN_TYPES') and hasattr(attr, 'CATEGORY'):
                nodes.append({
                    "name": attr.__name__,
                    "input_types": attr.INPUT_TYPES(),
                    "return_types": attr.RETURN_TYPES,
                    "category": attr.CATEGORY,
                    "display_name": getattr(attr, 'display_name', None),
                    "description": getattr(attr, 'description', None)
                })
    return nodes

def produce_node_definitions_file(directory: str, output_file: str):
    node_files = scan_directory_for_nodes(directory)
    all_nodes = []
    for file in node_files:
        module = import_module_from_path(file)
        nodes = extract_node_metadata(module)
        all_nodes.extend(nodes)

    
    with open(output_file, 'w') as f:
        json.dump(all_nodes, f, indent=4, cls=CustomJSONEncoder)
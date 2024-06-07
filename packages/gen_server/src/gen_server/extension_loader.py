import os
import importlib
import inspect
import json
import pkg_resources
import configparser
# from .paths import get_folder_path
import logging

import traceback
from typing import Dict, Union, Callable, Any, Type, TypeVar

import sys
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points, EntryPoint
else:
    from importlib.metadata import entry_points, EntryPoint

T = TypeVar('T')  # Generic type variable


def load_extensions(entry_point_group: str, expected_type: Type[T] = object) -> Dict[str, T]:
    components: dict[str, T] = {}
    discovered_plugins = entry_points(group=entry_point_group)
    
    for entry_point in discovered_plugins:
        # Scope the component's name using the distribution name; ex. 'comfy_creator.sdxl' rather than just 'sdxl'
        package_name = entry_point.dist.metadata['Name']
        scoped_name = f"{package_name}.{entry_point.name}"
        
        try:
            component = entry_point.load()
            
            # Optionally verify the loaded component matches our expected type
            if isinstance(component, expected_type):
                components[scoped_name] = component
            else:
                logging.warning(f"Component {scoped_name} does not match the expected type {expected_type.__name__}.")
        except Exception as error:
            logging.error(f"Failed to load component {scoped_name}: {str(error)}")
    
    return components

# Api-endpoints will extend the aiohttp rest server somehow
# Architectures will be classes that can be used to detect models and instantiate them
# custom nodes will define new nodes to be instantiated by the graph-editor
# widgets will somehow define react files to be somehow be imported by the client
API_ENDPOINTS = load_extensions('comfy_creator.api')
CUSTOM_NODES = load_extensions('comfy_creator.custom_nodes')
WIDGETS = load_extensions('comfy_creator.widgets')


def generate_node_definitions():
    node_definitions = []

    for node_name, node_class in CUSTOM_NODES.items():
        node_definition = {
            "name": node_name,
            "display_name": 'TO DO: NAMES',
            "category": node_class.CATEGORY if hasattr(node_class, 'CATEGORY') else "Custom Nodes",
            "inputs": node_class.INPUT_TYPES() if hasattr(node_class, 'INPUT_TYPES') else None,
            "outputs": node_class.RETURN_TYPES if hasattr(node_class, 'RETURN_TYPES') else None,
        }
        node_definitions.append(node_definition)

    return node_definitions


# def load_custom_nodes():
#     # Dictionary to hold the loaded nodes from different packages
#     custom_nodes = {}

#     # Discover and load entry points from the 'comfyui_custom_nodes' group
#     for entry_point in pkg_resources.iter_entry_points('comfyui_custom_nodes'):
#         # Load the entry point (this executes the function or returns the object defined in setup.py)
#         get_nodes_func = entry_point.load()

#         # Call the function to get the nodes (assuming these functions return a list of node instances or definitions)
#         nodes = get_nodes_func()

#         # Store the nodes under the name provided in the entry point
#         custom_nodes[entry_point.name] = nodes

#     return custom_nodes


# class Architecture:
#     pass

# class ModelRegistry:
#     def __init__(self):
#         self.architectures = {}

#     def load_architectures(self):
#         for entry_point in pkg_resources.iter_entry_points('my_model_framework.architectures'):
#             arch_class = entry_point.load()
#             if issubclass(arch_class, Architecture):
#                 self.register(arch_class)

#     def register(self, arch_class):
#         self.architectures[arch_class.id] = arch_class

#     def find_architecture(self, state_dict):
#         for arch in self.architectures.values():
#             if arch.detect(state_dict):
#                 return arch
#         return None



# NODE_CLASSES = {}
# DISPLAY_NAMES = {}

# def discover_custom_nodes():
#     config_path = os.path.join(get_folder_path('extensions'), 'config.txt')
#     # config = configparser.ConfigParser()
#     # config.read(config_path)

#     with open(config_path, 'r') as file:
#         package_names = [line.strip() for line in file]

#     print("Loading custom nodes...")
#     for package_name in package_names:
#         print(f"Package: {package_name}")
#         try:
#             for entry_point in pkg_resources.iter_entry_points("comfyui_custom_nodes"):
#                 # Look for entry points in the specified package
#                 if entry_point.dist.key == package_name:
#                     print(f"Loading custom nodes from package: {package_name}")
#                     print(f"Entry point: {entry_point.name}")
#                     module_name = entry_point.name
#                     node_mapping = entry_point.load()()
#                     namespaced_mapping = {}
#                     for node_name, node_class in node_mapping.items():
#                         namespaced_name = f"{module_name}.{node_name}"
#                         namespaced_mapping[namespaced_name] = node_class
#                         display_name = getattr(node_class, "DISPLAY_NAME", node_name)
#                         DISPLAY_NAMES[namespaced_name] = display_name
#                     NODE_CLASSES.update(namespaced_mapping)
#         except ImportError:
#             print(f"Failed to import package: {package_name}")


# def load_custom_node(extensions_dir):
#     for root, dirs, files in os.walk(extensions_dir):
#         for filename in files:
#             if filename == '__init__.py' or filename == '__pycache__':
#                 continue
#             elif filename.endswith('.py'):
#                 module_name = os.path.splitext(filename)[0]
#                 module_path = os.path.join(root, filename)
#                 spec = importlib.util.spec_from_file_location(module_name, module_path)
#                 module = importlib.util.module_from_spec(spec)
#                 spec.loader.exec_module(module)

#                 for name, obj in inspect.getmembers(module):
#                     if inspect.isclass(obj):
#                         # Ensure it's a class defined in this module
#                         if obj.__module__ == module.__name__ and hasattr(obj, "FUNCTION"):
#                             # Construct the dictionary
#                             folder_name = os.path.basename(root)
#                             class_name = f"{folder_name}.{obj.__name__}"
#                             NODE_CLASSES[class_name] = obj
#                             DISPLAY_NAMES[class_name] = getattr(obj, "DISPLAY_NAME", obj.__name__)




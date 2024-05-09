import os
import importlib
import inspect
import json
import pkg_resources
import configparser
from paths import get_folder_path
import logging
import sys
import traceback



NODE_CLASSES = {}
DISPLAY_NAMES = {}

def discover_custom_nodes():
    config_path = os.path.join(get_folder_path('extensions'), 'config.txt')
    # config = configparser.ConfigParser()
    # config.read(config_path)

    with open(config_path, 'r') as file:
        package_names = [line.strip() for line in file]

    print("Loading custom nodes...")
    for package_name in package_names:
        print(f"Package: {package_name}")
        try:
            for entry_point in pkg_resources.iter_entry_points("comfyui_custom_nodes"):
                # Look for entry points in the specified package
                if entry_point.dist.key == package_name:
                    print(f"Loading custom nodes from package: {package_name}")
                    print(f"Entry point: {entry_point.name}")
                    module_name = entry_point.name
                    node_mapping = entry_point.load()()
                    namespaced_mapping = {}
                    for node_name, node_class in node_mapping.items():
                        namespaced_name = f"{module_name}.{node_name}"
                        namespaced_mapping[namespaced_name] = node_class
                        display_name = getattr(node_class, "DISPLAY_NAME", node_name)
                        DISPLAY_NAMES[namespaced_name] = display_name
                    NODE_CLASSES.update(namespaced_mapping)
        except ImportError:
            print(f"Failed to import package: {package_name}")



def load_nodes_from_directory(directory):
    for filename in os.listdir(directory):
        if filename == '__init__.py' or filename == '__pycache__':
            continue
        elif filename.endswith('.py'):
            module_path = os.path.join(directory, filename)
            load_custom_node(module_path)

def load_all_nodes(extensions_dir):
    for directory_name in os.listdir(extensions_dir):
        directory_path = os.path.join(extensions_dir, directory_name)
        if os.path.isdir(directory_path):
            print(f"Loading nodes from directory: {directory_path}")
            load_nodes_from_directory(directory_path)
        else:
            continue



def load_core_node(core_nodes_dir):
    for filename in os.listdir(core_nodes_dir):
        if filename == '__init__.py' or filename == '__pycache__':
            continue
        elif filename.endswith('.py'):
            module_name = os.path.splitext(filename)[0]
            module_path = os.path.join(core_nodes_dir, filename)
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj):
                    # Ensure it's a class defined in this module
                    if obj.__module__ == module.__name__:
                        # Construct the dictionary
                        class_name = f"core.{obj.__name__}"
                        NODE_CLASSES[class_name] = obj
                        DISPLAY_NAMES[class_name] = getattr(obj, "DISPLAY_NAME", obj.__name__)

def load_custom_node(extensions_dir):
    for root, dirs, files in os.walk(extensions_dir):
        for filename in files:
            if filename == '__init__.py' or filename == '__pycache__':
                continue
            elif filename.endswith('.py'):
                module_name = os.path.splitext(filename)[0]
                module_path = os.path.join(root, filename)
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj):
                        # Ensure it's a class defined in this module
                        if obj.__module__ == module.__name__ and hasattr(obj, "FUNCTION"):
                            # Construct the dictionary
                            folder_name = os.path.basename(root)
                            class_name = f"{folder_name}.{obj.__name__}"
                            NODE_CLASSES[class_name] = obj
                            DISPLAY_NAMES[class_name] = getattr(obj, "DISPLAY_NAME", obj.__name__)


def generate_node_definitions():
    node_definitions = []

    for node_name, node_class in NODE_CLASSES.items():
        node_definition = {
            "name": node_name,
            "display_name": DISPLAY_NAMES.get(node_name, node_name),
            "category": node_class.CATEGORY if hasattr(node_class, 'CATEGORY') else "Custom Nodes",
            "inputs": node_class.INPUT_TYPES() if hasattr(node_class, 'INPUT_TYPES') else None,
            "outputs": node_class.RETURN_TYPES if hasattr(node_class, 'RETURN_TYPES') else None,
        }
        node_definitions.append(node_definition)

    return node_definitions


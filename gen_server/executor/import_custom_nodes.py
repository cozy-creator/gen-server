import os
import importlib
import inspect
import json
import pkg_resources


NODE_CLASSES = {}
DISPLAY_NAMES = {}

def discover_custom_nodes():
    for entry_point in pkg_resources.iter_entry_points("comfyui_custom_nodes"):
        module_name = entry_point.name
        node_mapping = entry_point.load()()
        namespaced_mapping = {}
        for node_name, node_class in node_mapping.items():
            namespaced_name = f"{module_name}.{node_name}"
            namespaced_mapping[namespaced_name] = node_class
            display_name = getattr(node_class, "DISPLAY_NAME", node_name)
            DISPLAY_NAMES[namespaced_name] = display_name
        NODE_CLASSES.update(namespaced_mapping)


def load_core_nodes(core_nodes_dir):
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

def generate_node_definitions():
    node_definitions = []

    for node_name, node_class in NODE_CLASSES.items():
        node_definition = {
            "name": node_name,
            "display_name": DISPLAY_NAMES.get(node_name, node_name),
            "node_class": str(node_class),
        }
        node_definitions.append(node_definition)

    return node_definitions


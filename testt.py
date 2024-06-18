import json
from pprint import pprint
from typing import Type, Any

from core_extension_1.custom_nodes import (
    LoadCheckpoint,
    CreatePipe,
    RunPipe,
)
from core_extension_1.widgets import WidgetDefinition
from gen_server.globals import CUSTOM_NODES
from gen_server.types_1.types_1 import Serializable

CUSTOM_NODES["LoadCheckpoint"] = LoadCheckpoint
CUSTOM_NODES["CreatePipe"] = CreatePipe
CUSTOM_NODES["RunPipe"] = RunPipe


def class_type(cls: Type[Any]):
    package_name = cls.__module__.split(".")[0]
    return f"{package_name}.{cls.__name__}"


def _get_node_definitions():
    definitions = {}
    for node_type in CUSTOM_NODES:
        node = CUSTOM_NODES[node_type]
        interface = node.update_interface()

        ux_widgets = []
        input_list = []
        output_list = []

        inputs = interface.get("inputs")
        if inputs is not None:
            for name in inputs:
                if isinstance(inputs[name], Serializable):
                    spec = inputs[name].serialize()
                    input_type = spec.get("type")
                else:
                    if isinstance(inputs[name], type):
                        input_type = inputs[name].__name__
                    else:
                        input_type = type(inputs[name]).__name__
                if isinstance(inputs[name], WidgetDefinition):
                    ux_widgets.append({"type": class_type(type(inputs[name])), "display_name": name, "spec": spec})
                else:
                    input_list.append({"edge_type": input_type, "display_name": name})

        outputs = interface.get("outputs")
        if outputs is not None:
            for name in outputs:
                if isinstance(outputs[name], type):
                    output_type = outputs[name].__name__
                else:
                    output_type = type(outputs[name]).__name__
                output_list.append({"edge_type": output_type, "display_name": name})

        definitions[node_type] = {
            "name": node.name,
            "type": class_type(node),
            "category": node.category,
            "description": node.description,
            "definition": {
                "inputs": input_list,
                "outputs": output_list,
                "ux_widgets": ux_widgets,
            },
        }

    return definitions


if __name__ == "__main__":
    defs = json.dumps(_get_node_definitions(), indent=2)
    print(defs)

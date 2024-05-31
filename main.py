from gen_server.executor.import_custom_nodes import discover_custom_nodes, load_core_nodes, generate_node_definitions
import json
from folders import get_folder_paths


if __name__ == "__main__":
    discover_custom_nodes()

    core_nodes_path = get_folder_paths("core_nodes")

    load_core_nodes(core_nodes_path)

    node_definitions = generate_node_definitions()

    with open("node_definitions.json", "w") as f:
        json.dump(node_definitions, f, indent=4)


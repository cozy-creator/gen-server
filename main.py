from gen_server.executor.import_custom_nodes import discover_custom_nodes, generate_node_definitions, load_custom_node
import json
from paths import get_folder_path
# from gen_server.request_handlers.web_server import start_server
from gen_server.executor.node_definition import produce_node_definitions_file



if __name__ == "__main__":
    produce_node_definitions_file("extensions/core", "node_definitions.json")
    # discover_custom_nodes()

    # core_nodes_path = get_folder_path("extensions")

    # load_custom_node(core_nodes_path)

    # node_definitions = generate_node_definitions()

    # with open("node_definitions.json", "w") as f:
    #     json.dump(node_definitions, f, indent=4)

    # start_server()


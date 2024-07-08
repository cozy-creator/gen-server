import json
import time


from gen_server.base_types.custom_node import custom_node_validator
from .base_types.architecture import architecture_validator
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from .cli_args import args
# from .common.firebase import initialize
# from .settings import settings
from .api import start_server, api_routes_validator
from .utils import load_extensions, find_checkpoint_files
from .globals import (
    API_ENDPOINTS,
    ARCHITECTURES,
    CUSTOM_NODES,
    WIDGETS,
    CHECKPOINT_FILES,
    CozyRunConfig,
    CozyCommands,
)
import argparse
import asyncio


def main():
    parser = argparse.ArgumentParser(description="Cozy Creator")
    commands = CozyCommands(parser)
    args = parser.parse_args()

    # Pydantic uses the :subcommand attribute to store the subcommand name.
    # And if we set a prefix e.g "cozy", it will store the subcommand name in the attribute "cozy.:subcommand"
    subcommand = getattr(args, ":subcommand")

    # Sub-argument switch
    if subcommand == "run":
        config = CozyRunConfig(**commands.run.dict())
        print(json.dumps(config.dict(), indent=2, default=str))
        run_app(config)
    elif subcommand is None:
        parser.print_help()
    else:
        print(f"\nError: Unknown command '{subcommand}'")
        parser.print_help()


def run_app(config: CozyRunConfig):
    print(f"Running with config: {config}")
    # We load the extensions inside a function to avoid circular dependencies

    # Api-endpoints will extend the aiohttp rest server somehow
    # Architectures will be classes that can be used to detect models and instantiate them
    # custom nodes will define new nodes to be instantiated by the graph-editor
    # widgets will somehow define react files to be somehow be imported by the client

    start_time = time.time()

    # All routes must be a function that returns -> Iterable[web.AbstractRouteDef]
    global API_ENDPOINTS
    start_time_api_endpoints = time.time()
    API_ENDPOINTS.update(
        load_extensions("comfy_creator.api", validator=api_routes_validator)
    )
    # expected_type=Callable[[], Iterable[web.AbstractRouteDef]]
    print(
        f"API_ENDPOINTS loading time: {time.time() - start_time_api_endpoints:.2f} seconds"
    )

    # compile architecture registry
    global ARCHITECTURES
    start_time_architectures = time.time()
    ARCHITECTURES.update(
        load_extensions("comfy_creator.architectures", validator=architecture_validator)
    )
    print(
        f"ARCHITECTURES loading time: {time.time() - start_time_architectures:.2f} seconds"
    )

    global CUSTOM_NODES
    start_time_custom_nodes = time.time()
    CUSTOM_NODES.update(
        load_extensions("comfy_creator.custom_nodes", validator=custom_node_validator)
    )
    print(
        f"CUSTOM_NODES loading time: {time.time() - start_time_custom_nodes:.2f} seconds"
    )

    global WIDGETS
    start_time_widgets = time.time()
    WIDGETS.update(load_extensions("comfy_creator.widgets"))
    print(f"WIDGETS loading time: {time.time() - start_time_widgets:.2f} seconds")

    # compile model registry
    global CHECKPOINT_FILES
    start_time_checkpoint_files = time.time()
    CHECKPOINT_FILES.update(find_checkpoint_files(model_dirs=config.models_dirs))
    print(
        f"CHECKPOINT_FILES loading time: {time.time() - start_time_checkpoint_files:.2f} seconds"
    )

    # debug
    print("Number of checkpoint files:", len(CHECKPOINT_FILES))

    # print(CHECKPOINT_FILES)
    def print_dict(d):
        for key, value in d.items():
            print(f"{key}: {str(value)}")

    print_dict(CHECKPOINT_FILES)

    end_time = time.time()
    print(
        f"Time taken to load extensions and compile registries: {end_time - start_time:.2f} seconds"
    )

    asyncio.run(start_server())

    return


if __name__ == "__main__":
    # initialize(json.loads(settings.firebase.service_account))

    # asyncio.run(main())
    main()

    # Run our REST server
    # web.run_app(app, port=8080)

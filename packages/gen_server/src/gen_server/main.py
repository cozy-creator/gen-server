import json
import time
import argparse
import asyncio
import sys
import os
from pydantic_settings import CliSettingsSource

from .config import init_config
from .base_types.custom_node import custom_node_validator
from .base_types.architecture import architecture_validator
from .api import start_server, api_routes_validator
from .utils import load_extensions, find_checkpoint_files
from .utils.paths import get_models_dir
from .globals import (
    API_ENDPOINTS,
    ARCHITECTURES,
    CUSTOM_NODES,
    WIDGETS,
    CHECKPOINT_FILES,
    RunCommandConfig,
    BuildWebCommandConfig,
)
from .node_definitions import produce_node_definitions_file
from .utils.cli_helpers import find_subcommand, find_arg_value, parse_known_args_wrapper
import warnings

warnings.filterwarnings("ignore", module="pydantic_settings")


def main():
    # produce_node_definitions_file('node_definitions.json')
    root_parser = argparse.ArgumentParser(description="Cozy Creator")

    # When we call parser.parse_args() the arg-parser will stop populating the --help menu
    # So we need to find the arguments _before_ we call parser.parse_args() inside of
    # CliSettingsSource() below.

    env_file = find_arg_value("--env_file") or find_arg_value("--env-file") or None
    # If no .env file is specified, try to find one in the current working directory
    if env_file is None:
        if os.path.exists(".env"):
            env_file = ".env"
        elif os.path.exists(".env.local"):
            env_file = ".env.local"

    secrets_dir = (
        find_arg_value("--secrets_dir")
        or find_arg_value("--secrets-dir")
        or "/run/secrets"
    )
    subcommand = find_subcommand()

    # Add subcommands
    subparsers = root_parser.add_subparsers(dest="command", help="Available commands")
    run_parser = subparsers.add_parser("run", help="Run the Cozy Creator server")
    build_web_parser = subparsers.add_parser("build-web", help="Build the web bundle")

    if subcommand == "run":
        cozy_config = init_config(
            run_parser,
            parse_known_args_wrapper,
            env_file=env_file,
            secrets_dir=secrets_dir,
        )

        produce_node_definitions_file("node_definitions.json")

        print(json.dumps(cozy_config.model_dump(), indent=2, default=str))
        run_app(cozy_config)

    elif subcommand in ["build-web", "build_web"]:
        cli_settings = CliSettingsSource(
            BuildWebCommandConfig, root_parser=build_web_parser
        )

        build_config = BuildWebCommandConfig(
            _env_file=env_file,  # type: ignore
            _secrets_dir=secrets_dir,  # type: ignore
            _cli_settings_source=cli_settings(args=True),  # type: ignore
        )

        print(json.dumps(build_config.model_dump(), indent=2, default=str))

    elif subcommand is None:
        print("No subcommand specified. Please specify a subcommand.")
        root_parser.print_help()
        sys.exit(1)

    else:
        print(f"Unknown subcommand: {subcommand}")
        root_parser.print_help()
        sys.exit(1)


def run_app(cozy_config: RunCommandConfig):
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
        load_extensions("cozy_creator.api", validator=api_routes_validator)
    )
    # expected_type=Callable[[], Iterable[web.AbstractRouteDef]]
    print(
        f"API_ENDPOINTS loading time: {time.time() - start_time_api_endpoints:.2f} seconds"
    )

    # compile architecture registry
    global ARCHITECTURES
    start_time_architectures = time.time()
    ARCHITECTURES.update(
        load_extensions("cozy_creator.architectures", validator=architecture_validator)
    )
    print(
        f"ARCHITECTURES loading time: {time.time() - start_time_architectures:.2f} seconds"
    )

    global CUSTOM_NODES
    start_time_custom_nodes = time.time()
    CUSTOM_NODES.update(
        load_extensions("cozy_creator.custom_nodes", validator=custom_node_validator)
    )
    print(
        f"CUSTOM_NODES loading time: {time.time() - start_time_custom_nodes:.2f} seconds"
    )

    global WIDGETS
    start_time_widgets = time.time()
    WIDGETS.update(load_extensions("cozy_creator.widgets"))
    print(f"WIDGETS loading time: {time.time() - start_time_widgets:.2f} seconds")

    # compile model registry
    global CHECKPOINT_FILES
    start_time_checkpoint_files = time.time()
    models_paths = [get_models_dir()] + cozy_config.aux_models_paths
    CHECKPOINT_FILES.update(find_checkpoint_files(models_paths=models_paths))
    print(
        f"CHECKPOINT_FILES loading time: {time.time() - start_time_checkpoint_files:.2f} seconds"
    )

    # debug
    print("Number of checkpoint files:", len(CHECKPOINT_FILES))

    end_time = time.time()
    print(
        f"Time taken to load extensions and compile registries: {end_time - start_time:.2f} seconds"
    )

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(start_server(cozy_config.host, cozy_config.port))
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up resources...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully...")
        sys.exit(0)
    # initialize(json.loads(settings.firebase.service_account))

    # asyncio.run(main())

    # Run our REST server
    # web.run_app(app, port=8080)

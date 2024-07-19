import json
import time
import argparse
import sys
import os
import concurrent.futures
import signal
from types import FrameType
from typing import Optional

from pydantic_settings import CliSettingsSource
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing.managers import SyncManager
from typing import cast

from .config import init_config
from .base_types.common import JobQueueItem
from .base_types.custom_node import custom_node_validator
from .base_types.architecture import architecture_validator
from .utils import load_extensions, find_checkpoint_files
from .api import start_api_server, api_routes_validator, create_aiohttp_app
from .utils import load_extensions, find_checkpoint_files, load_custom_node_specs
from .utils.paths import get_models_dir
from .utils.file_handler import get_file_handler
from .globals import (
    get_api_endpoints,
    get_custom_nodes,
    update_api_endpoints,
    update_architectures,
    update_custom_nodes,
    update_widgets,
    update_checkpoint_files,
    get_checkpoint_files,
    RunCommandConfig,
    BuildWebCommandConfig,
)
from .node_definitions import produce_node_definitions_file
from .utils.cli_helpers import find_subcommand, find_arg_value, parse_known_args_wrapper
from .executor.io_worker import run_io_worker
from .executor.gpu_worker import run_gpu_worker
import warnings

warnings.filterwarnings("ignore", module="pydantic_settings")


def main():
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

        produce_node_definitions_file(
            f"{cozy_config.workspace_path}/node_definitions.json"
        )

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

    start_time_api_endpoints = time.time()
    update_api_endpoints(
        load_extensions("cozy_creator.api", validator=api_routes_validator)
    )

    # expected_type=Callable[[], Iterable[web.AbstractRouteDef]]
    print(
        f"API_ENDPOINTS loading time: {time.time() - start_time_api_endpoints:.2f} seconds"
    )

    # compile architecture registry
    start_time_architectures = time.time()
    update_architectures(
        load_extensions("cozy_creator.architectures", validator=architecture_validator)
    )

    print(
        f"ARCHITECTURES loading time: {time.time() - start_time_architectures:.2f} seconds"
    )

    start_time_custom_nodes = time.time()
    update_custom_nodes(
        load_extensions("cozy_creator.custom_nodes", validator=custom_node_validator)
    )

    print(
        f"CUSTOM_NODES loading time: {time.time() - start_time_custom_nodes:.2f} seconds"
    )

    start_time_widgets = time.time()
    update_widgets(load_extensions("cozy_creator.widgets"))
    print(f"WIDGETS loading time: {time.time() - start_time_widgets:.2f} seconds")

    # compile model registry
    start_time_checkpoint_files = time.time()
    models_paths = [get_models_dir()] + cozy_config.aux_models_paths
    update_checkpoint_files(find_checkpoint_files(models_paths=models_paths))
    print(
        f"CHECKPOINT_FILES loading time: {time.time() - start_time_checkpoint_files:.2f} seconds"
    )


    # Load custom node specs and send them to the client
    start_time_custom_nodes_specs = time.time()
    custom_node_specs = load_custom_node_specs()
    with open(f"{cozy_config.workspace_path}/custom_node_specs.json", "w") as f:
        json.dump(custom_node_specs, f, indent=2)


    end_time = time.time()
    print(
        f"Time taken to load extensions and compile registries: {end_time - start_time_custom_nodes_specs:.2f} seconds"
    )

    # debug
    print("Number of checkpoint files:", len(get_checkpoint_files()))

    end_time = time.time()
    print(
        f"Time taken to load extensions and compile registries: {end_time - start_time:.2f} seconds"
    )

    # ====== The server is initialized; good spot to run your tests here ======

    # from .executor import generate_images
    # async def test_generate_images():
    #     async for file_metadata in generate_images(
    #         { "dark_sushi_25d_v40": 1 },
    #         "a beautiful anime girl",
    #         "poor quality, worst quality, watermark, blurry",
    #         42069,
    #         "16/9"
    #     ):
    #         print(file_metadata)

    # asyncio.run(test_generate_images())

    try:
        manager = multiprocessing.Manager()
        
        # Gen-tasks will be placed on this by the api-server, and consumed by the
        # gpu-worker.
        job_queue: SyncManager.Queue[JobQueueItem] = manager.Queue()
        
        # A tensor queue so that the gpu-workers process can push their finished
        # files into the io-worker process.
        tensor_queue = manager.Queue()
        
        checkpoint_files = get_checkpoint_files()
        api_endpoints = get_api_endpoints()
        custom_nodes = get_custom_nodes()
        file_handler = get_file_handler()

        # Create a process pool for the workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(start_api_server, cozy_config, job_queue, checkpoint_files, api_endpoints),
                executor.submit(run_gpu_worker, job_queue, tensor_queue, cozy_config, custom_nodes, checkpoint_files),
                executor.submit(run_io_worker, tensor_queue, file_handler)
            ]

            def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
                print("Received shutdown signal. Terminating processes...")
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False)
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Wait for futures to complete (which they won't, unless cancelled)
            concurrent.futures.wait(futures)

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

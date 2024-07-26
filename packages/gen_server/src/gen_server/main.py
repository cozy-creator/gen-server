import json
import logging
import time
import argparse
import sys
import os
import concurrent.futures
from concurrent.futures import Future, ProcessPoolExecutor
import asyncio

from typing import Any, Callable
import signal
from types import FrameType
from typing import Optional

from pydantic_settings import CliSettingsSource
import multiprocessing

from gen_server.utils.file_handler import LocalFileHandler
from gen_server.utils.web import install_and_build_web_dir
from .paths import ensure_workspace_path
from .config import init_config
from .base_types.custom_node import custom_node_validator
from .base_types.architecture import architecture_validator
from .utils import load_extensions, find_checkpoint_files
from .api import start_api_server, api_routes_validator
from .utils import load_custom_node_specs, get_file_handler
from .utils.paths import get_models_dir, get_web_dir
from .globals import (
    get_api_endpoints,
    get_custom_nodes,
    update_api_endpoints,
    update_architectures,
    update_custom_nodes,
    update_widgets,
    update_checkpoint_files,
    get_checkpoint_files,
    get_architectures,
)
from .base_types.pydantic_models import (
    RunCommandConfig,
    BuildWebCommandConfig,
    InstallCommandConfig,
)
from .utils.cli_helpers import find_subcommand, find_arg_value, parse_known_args_wrapper

# from .executor.io_worker import start_io_worker
# from .executor.gpu_worker import start_gpu_worker
from .executor.gpu_worker_non_io import start_gpu_worker_non_io


import warnings

warnings.filterwarnings("ignore", module="pydantic_settings")

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,  # Set the minimum level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    root_parser = argparse.ArgumentParser(description="Cozy Creator")

    # When we call parser.parse_args() the arg-parser will stop populating the --help menu
    # So we need to find the arguments _before_ we call parser.parse_args() inside of
    # CliSettingsSource() below.

    env_file = find_arg_value("--env_file") or find_arg_value("--env-file") or None
    # If no .env file is specified, try to find one in the workspace path
    if env_file is None:
        workspace_path = (
            find_arg_value("--workspace_path")
            or find_arg_value("--workspace-path")
            or os.path.expanduser("~/.cozy-creator")
        )
        if os.path.exists(os.path.join(workspace_path, ".env")):
            env_file = os.path.join(workspace_path, ".env")
        elif os.path.exists(os.path.join(workspace_path, ".env.local")):
            env_file = os.path.join(workspace_path, ".env.local")

    secrets_dir = (
        find_arg_value("--secrets_dir")
        or find_arg_value("--secrets-dir")
        or "/run/secrets"
    )

    subcommand = find_subcommand()

    # Add subcommands
    subparsers = root_parser.add_subparsers(dest="command", help="Available commands")
    run_parser = subparsers.add_parser("run", help="Run the Cozy Creator server")
    install_parser = subparsers.add_parser(
        "install", help="Install and prepare the Cozy environment"
    )
    build_web_parser = subparsers.add_parser("build-web", help="Build the web bundle")

    if subcommand == "run":
        cozy_config = init_config(
            run_parser,
            parse_known_args_wrapper,
            env_file=env_file,
            secrets_dir=secrets_dir,
        )

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
    elif subcommand == "install":
        _config = InstallCommandConfig(
            _env_file=env_file,  # type: ignore
            _cli_settings_source=CliSettingsSource(  # type: ignore
                InstallCommandConfig,
                root_parser=install_parser,
            ),
        )

        # Ensure the web directory exists
        web_dir = get_web_dir()
        if not os.path.exists(web_dir):
            print(f"Web directory not found at {web_dir}")
            sys.exit(1)

        # Install and build the web directory
        install_and_build_web_dir(web_dir)

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

    # Ensure our workspace directory and its subdirectories exist. Add a .env.example if needed
    ensure_workspace_path(cozy_config.workspace_path)

    print("Cozy config:", cozy_config)

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
        # Stores JobQueueItem
        job_queue = manager.Queue()

        # A tensor queue so that the gpu-workers process can push their finished
        # files into the io-worker process.
        # tensor_queue = manager.Queue()

        # shutdown_event = manager.Event()

        # Get global variables that we need to pass to sub-processes
        checkpoint_files = get_checkpoint_files()
        api_endpoints = get_api_endpoints()
        custom_nodes = get_custom_nodes()
        architectures = get_architectures()
        node_specs = load_custom_node_specs(custom_nodes)

        # Create a process pool for the workers
        # Note that we must use 'spawn' rather than 'fork' because CUDA and Windows do not
        # support forking.
        with ProcessPoolExecutor(
            max_workers=3,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [
                named_future(
                    executor,
                    "api_worker",
                    start_api_server,
                    job_queue,
                    cozy_config,
                    checkpoint_files,
                    node_specs,
                    api_endpoints,
                ),
                named_future(
                    executor,
                    "gpu_worker",
                    start_gpu_worker_non_io,
                    job_queue,
                    cozy_config,
                    custom_nodes,
                    checkpoint_files,
                    architectures,
                ),
                # named_future(
                #     executor,
                #     "io_worker",
                #     asyncio.run(start_io_worker(tensor_queue, cozy_config)),
                #     tensor_queue,
                #     cozy_config,
                # ),
            ]

            def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
                print("Received shutdown signal. Terminating processes...")

                # We only delete temp files if we are running locally
                file_handler = get_file_handler()
                if isinstance(file_handler, LocalFileHandler):
                    asyncio.run(file_handler.delete_files(folder_name="temp"))

                # for future in futures:
                #     future.cancel()
                # shutdown_event.set()
                manager.shutdown()
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Print exceptions from futures
            for future in futures:
                future.add_done_callback(
                    lambda f: print(f"Error in {f.__dict__['name']}: {f.exception()}")
                    if f.exception()
                    else None
                )

            # print('All worker processes started successfully', flush=True)

            # Wait for futures to complete (which they won't, unless cancelled)
            _done, not_done = concurrent.futures.wait(
                futures, return_when=concurrent.futures.ALL_COMPLETED
            )
            for future in not_done:
                future.cancel()

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up resources...")


def named_future(
    executor: ProcessPoolExecutor, name: str, func: Callable, *args: Any, **kwargs: Any
) -> Future:
    """
    Assign each future a semantic name for error-logging
    """
    future = executor.submit(func, *args, **kwargs)
    future.__dict__["name"] = name
    return future


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

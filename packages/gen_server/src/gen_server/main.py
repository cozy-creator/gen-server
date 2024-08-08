import os


# Disable triton on Windows since it is not supported
# We do this at the top of the file to ensure it is set before any imports that may trigger triton
if os.name == 'nt':
    print("\n----- Windows detected, disabling Triton -----\n")
    os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = "1"


import logging
import time
import argparse
import sys
import concurrent.futures
from concurrent.futures import Future, ProcessPoolExecutor
import asyncio

from typing import Any, Callable
import signal
from types import FrameType
from typing import Optional

from pydantic_settings import CliSettingsSource
import multiprocessing

from gen_server.base_types.authenticator import api_authenticator_validator
from gen_server.utils.file_handler import LocalFileHandler
from gen_server.utils.web import install_and_build_web_dir
from torch import ge
from .utils.paths import ensure_app_dirs
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
    update_api_authenticator,
    get_api_authenticator,
    get_hf_model_manager,
)
from .base_types.pydantic_models import (
    RunCommandConfig,
    BuildWebCommandConfig,
    DownloadCommandConfig,
)
from .utils.download_manager import DownloadManager
from .utils.cli_helpers import find_subcommand, find_arg_value, parse_known_args_wrapper
from .executor.gpu_worker_non_io import start_gpu_worker
from .utils.paths import DEFAULT_HOME_DIR
import warnings

warnings.filterwarnings("ignore", module="pydantic_settings")

# This is a warning from one of our architectures that uses a deprecated function in one of its dependencies.
warnings.filterwarnings("ignore", message="size_average and reduce args will be deprecated, please use reduction='mean' instead.") 


# Configure the root logger
logging.basicConfig(
    level=logging.INFO,  # Set the minimum level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress the ModuleNotFoundError for triton
# logging.getLogger("xformers").setLevel(logging.ERROR)
# import warnings
# warnings.filterwarnings("ignore", module="pydantic_settings")

def main():


    root_parser = argparse.ArgumentParser(description="Cozy Creator")

    # When we call parser.parse_args() the arg-parser will stop populating the --help menu
    # So we need to find the arguments _before_ we call parser.parse_args() inside of
    # CliSettingsSource() below.

    env_file = find_arg_value("--env_file") or find_arg_value("--env-file") or None
    # If no .env file is specified, try to find one in the workspace path
    if env_file is None:
        home = (
            find_arg_value("--home")
            or find_arg_value("--home-dir")
            or find_arg_value("--home_dir")
            or DEFAULT_HOME_DIR
        )
        if os.path.exists(os.path.join(home, ".env")):
            env_file = os.path.join(home, ".env")
        elif os.path.exists(os.path.join(home, ".env.local")):
            env_file = os.path.join(home, ".env.local")

    secrets_dir = (
        find_arg_value("--secrets_dir")
        or find_arg_value("--secrets-dir")
        or "/run/secrets"
    )

    env_file = find_arg_value("--env_file") or find_arg_value("--env-file") or None
    # If no .env file is specified, try to find one in the workspace path
    if env_file is None:
        home_dir = (
            find_arg_value("--home-dir")
            or find_arg_value("--home_dir")
            or DEFAULT_HOME_DIR
        )
        if os.path.exists(os.path.join(home_dir, ".env")):
            env_file = os.path.join(home_dir, ".env")
        elif os.path.exists(os.path.join(home_dir, ".env.local")):
            env_file = os.path.join(home_dir, ".env.local")

    # Load the environment variables into memory
    if env_file:
        from dotenv import load_dotenv

        load_dotenv(env_file)
        print(f"Loaded environment variables from {env_file}")

    # We don't really do much with this yet...
    secrets_dir = (
        find_arg_value("--secrets_dir")
        or find_arg_value("--secrets-dir")
        or "/run/secrets"
    )

    # Set the config file from the CLI
    config_file = find_arg_value("--config-file") or find_arg_value("--config_file")
    # config_file = resolve_config_path(config_file)
    if config_file is not None:
        if os.path.exists(config_file):
            os.environ["COZY_CONFIG_FILE"] = config_file
        else:
            print(f"Config file not found at {config_file}. Skipping.")

    subcommand = find_subcommand()

    # Add subcommands
    subparsers = root_parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run the Cozy Creator server")
    run_parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        metavar="",
        help="Path to an environment file loaded on startup",
    )
    run_parser.add_argument(
        "--secrets-dir",
        type=str,
        default=None,
        metavar="",
        help="Path to the secrets directory",
    )
    run_parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        metavar="",
        help="Path to a YAML configuration file",
    )

    build_web_parser = subparsers.add_parser("build-web", help="Build the web bundle")
    download_parser = subparsers.add_parser(
        "download", help="Download models to cozy's local cache"
    )

    def get_cli_settings(
        cls: Any, root_parser: argparse.ArgumentParser
    ) -> CliSettingsSource:
        return CliSettingsSource(
            cls,
            root_parser=root_parser,
            cli_parse_args=True,
            cli_enforce_required=False,
            parse_args_method=parse_known_args_wrapper,
        )

    if subcommand == "run":
        cozy_config = init_config(
            run_parser,
            parse_known_args_wrapper,
            secrets_dir=secrets_dir,
        )

        run_app(cozy_config)

    elif subcommand in ["build-web", "build_web"]:
        cli_settings = get_cli_settings(BuildWebCommandConfig, build_web_parser)
        _build_config = BuildWebCommandConfig(
            _secrets_dir=secrets_dir,  # type: ignore
            _cli_settings_source=cli_settings(args=True),  # type: ignore
        )

        # Ensure the web directory exists
        web_dir = get_web_dir()
        if not os.path.exists(web_dir):
            print(f"Web directory not found at {web_dir}")
            sys.exit(1)

        # Install and build the web directory
        install_and_build_web_dir(web_dir)

    elif subcommand == "download":
        cli_settings = get_cli_settings(DownloadCommandConfig, download_parser)
        config = DownloadCommandConfig(
            _cli_settings_source=cli_settings(args=True),  # type: ignore
        )

        hf_manager = get_hf_model_manager()
        asyncio.run(
            hf_manager.download(config.repo_id, config.file_name, config.sub_folder)
        )

    elif subcommand is None:
        print("No subcommand specified. Please specify a subcommand.")
        root_parser.print_help()
        sys.exit(0)

    else:
        print(f"Unknown subcommand: {subcommand}")
        root_parser.print_help()
        sys.exit(0)


def run_app(cozy_config: RunCommandConfig):
    # Ensure our app directories exist.
    ensure_app_dirs()

    # print(f'COZY CONFIG: {cozy_config}')

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

    start_time_api_authenticator = time.time()

    api_authenticators = load_extensions(
        "cozy_creator.api_authenticator", api_authenticator_validator
    )

    if cozy_config.api_authenticator is not None:
        update_api_authenticator(api_authenticators.get(cozy_config.api_authenticator))
    print(
        f"AUTHENTICATOR loading time: {time.time() - start_time_api_authenticator:.2f} seconds"
    )

    # compile model registry
    start_time_checkpoint_files = time.time()
    models_dirs = [get_models_dir()] + cozy_config.aux_models_paths
    update_checkpoint_files(find_checkpoint_files(models_dirs))
    print(
        f"CHECKPOINT_FILES loading time: {time.time() - start_time_checkpoint_files:.2f} seconds"
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
        # Stores JobQueueItem
        job_queue = manager.Queue()

        cancel_registry = manager.dict()

        # A tensor queue so that the gpu-workers process can push their finished
        # files into the io-worker process.
        # tensor_queue = manager.Queue()

        # shutdown_event = manager.Event()

        # Get global variables that we need to pass to sub-processes
        checkpoint_files = get_checkpoint_files()
        api_endpoints = get_api_endpoints()
        custom_nodes = get_custom_nodes()
        architectures = get_architectures()
        api_authenticator = get_api_authenticator()
        node_specs = load_custom_node_specs(custom_nodes)
        download_manager = DownloadManager(hf_manager=get_hf_model_manager())

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
                    cancel_registry,
                    cozy_config,
                    checkpoint_files,
                    node_specs,
                    api_endpoints,
                    api_authenticator,
                    download_manager,
                ),
                named_future(
                    executor,
                    "gpu_worker",
                    start_gpu_worker,
                    job_queue,
                    cancel_registry,
                    cozy_config,
                    custom_nodes,
                    checkpoint_files,
                    architectures,
                ),
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

import os
import time
from pathlib import Path

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
    initialize_config,
    CozyConfig,
    CozyCommands,
)
import argparse
import asyncio
from pydantic_settings import CliSettingsSource, CliSubCommand

file_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../models/sd3_medium_incl_clips_t5xxlfp8.safetensors",
    )
)

def main():
    config = CozyCommands()
    
    if config.run:
        # Use the env_file and secrets_dir from the command line args
        cozy_config = CozyConfig(
            _env_file=config.run.env_file,
            _secrets_dir=config.run.secrets_dir
        )
        run_app(cozy_config)
    else:
        print("Use 'cozy run' to start the server.")
    
    return
    
    parser = argparse.ArgumentParser(description="Cozy Creator")
    
    # add 'run' command
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')
    run_parser = subparsers.add_parser('run', help='Run the Cozy Gen Server')
    run_parser.add_argument('--env-file', help="Path to .env file")
    run_parser.add_argument('--secrets-dir', help="Path to secrets directory")
    
    args = parser.parse_args()
    
    # Sub-argument switch
    if args.command == 'run':
        # use our cli-settings in the constructor of Cozy Creator's config
        cli_settings = CliSettingsSource(CozyConfig, root_parser=run_parser)
        config_kwargs = {'_cli_settings_source': cli_settings(args=True)}
    
        if args.env_file:
            config_kwargs['_env_file'] = args.env_file
        if args.secrets_dir:
            config_kwargs['_secrets_dir'] = args.secrets_dir
        
        config = CozyConfig(**config_kwargs)
        import json
        print(json.dumps(config.dict(), indent=2, default=str))
        run_app(config.run)
    elif args.command is None:
        parser.print_help()
    else:
        print(f"\nError: Unknown command '{args.command}'")
        parser.print_help()
    
    return
        
    # Parse command-line args
    parser = argparse.ArgumentParser(description="Cozy Creator")
    parser.add_argument('--config', help="Path to JSON config file")
    parser.add_argument('--hostname', help="Hostname to use")
    parser.add_argument('--port', type=int, help="Port to use")
    parser.add_argument('--workspace-dir', help="Workspace directory")
    parser.add_argument('--models-dirs', nargs='+', help="Model directories")
    parser.add_argument('--filesystem-type', choices=['LOCAL', 'S3'], help="Filesystem type")
    
    args = parser.parse_args()
    
    # Loads the application's config from environment variables and a .env file in the current
    # working directory
    config = CozyConfig(config_path=args.config)
    
    # Override configurations with command-line arguments provided
    if args.hostname:
        config.hostname = args.hostname
    if args.port:
        config.port = args.port
    if args.workspace_dir:
        config.workspace_dir = args.workspace_dir
    if args.models_dirs:
        config.models_dirs = args.models_dirs
    if args.filesystem_type:
        config.filesystem_type = args.filesystem_type


def run_app(cozy_config: CliSubCommand[CozyConfig]):
    print(f"Running with config: {cozy_config}")
    return
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
    CHECKPOINT_FILES.update(find_checkpoint_files(model_dirs=cozy_config.models_dirs))
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

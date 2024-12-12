import asyncio
from concurrent.futures import Future, ProcessPoolExecutor
import json
import logging
import os
import struct
import sys
import time
import warnings
from typing import Any, Callable


from .tcp_server import TCPServer, RequestContext
from .worker.gpu_worker import generate_images_non_io
from .utils.image import tensor_to_bytes
from .globals import get_model_memory_manager
from .model_command_handler import ModelCommandHandler

from .base_types.config import RuntimeConfig
from .parse_cli import parse_arguments
from .config import set_config

# Ignore warnings from pydantic_settings (/run/secrets does not exist)
warnings.filterwarnings(
    "ignore",
    message=r'directory ".*" does not exist',
    category=UserWarning,
    module="pydantic_settings.sources",
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def request_handler(context: RequestContext):
    """Handles incoming messages from the TCP server"""

    data = context.data()
    logger.info(f"TCP Server received data: {data}")

    try:
        json_data = json.loads(data.decode())

        # Process model management commands
        if "command" in json_data:
            logger.info(f"Processing model command: {json_data['command']}")
            command_handler = ModelCommandHandler()
            try:
                # Run the async command handler in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    command_handler.handle_command(json_data)
                )

                # Send response
                response_bytes = json.dumps(response).encode()
                size = struct.pack("!I", len(response_bytes))
                context.send(size + response_bytes)
            except Exception as e:
                logger.error(f"Error handling model command: {e}")
                error_response = json.dumps(
                    {"status": "error", "error": str(e)}
                ).encode()
                size = struct.pack("!I", len(error_response))
                context.send(size + error_response)
        else:
            # Handle image generation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def generate():
                async for [model_id, images] in generate_images_non_io(json_data):
                    outputs = tensor_to_bytes(images)
                    model_id_bytes = model_id.encode("utf-8")
                    model_id_header = (
                        struct.pack("!I", len(model_id_bytes)) + model_id_bytes
                    )
                    for output in outputs:
                        total_size = struct.pack(
                            "!I", len(model_id_header) + len(output)
                        )
                        context.send(total_size + model_id_header + output)

            loop.run_until_complete(generate())

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON data: {e}")
        error_response = json.dumps(
            {"status": "error", "error": "Invalid JSON"}
        ).encode()
        size = struct.pack("!I", len(error_response))
        context.send(size + error_response)


def run_tcp_server(config: RuntimeConfig):
    server = TCPServer(port=config.port, host=config.host)

    server.set_handler(request_handler)
    server.start(lambda addr, port: print(f"Python runtime available on {addr}:{port}"))


# def startup_extensions(_config: RuntimeConfig):
#     start_time_custom_nodes = time.time()

#     custom_nodes = load_extensions(
#         "cozy_creator.custom_nodes", validator=custom_node_validator
#     )
#     if not custom_nodes:
#         logger.warning("No custom nodes were loaded! Generation cannot function.")

#     update_custom_nodes(custom_nodes)

#     update_architectures(load_extensions("cozy_creator.architectures"))

#     print(
#         f"CUSTOM_NODES loading time: {time.time() - start_time_custom_nodes:.2f} seconds"
#     )


async def load_and_warmup_models(config: RuntimeConfig):
    model_memory_manager = get_model_memory_manager()
    enabled_models = config.enabled_models

    if not enabled_models:
        logger.info("No models configured for startup loading")
        return

    logger.info(f"Initializing startup models: {enabled_models}")

    try:
        await model_memory_manager.initialize_startup_models(enabled_models)
    except Exception as e:
        logger.error(f"Error during startup model initialization: {e}")
        raise


async def main_async():
    config = parse_arguments()

    set_config(config)  # Do we need these dumb global variables?

    # startup_extensions(config)

    # Load and warm up models
    await load_and_warmup_models(config)

    # Run the TCP server
    run_tcp_server(config)


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        os._exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


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
    main()

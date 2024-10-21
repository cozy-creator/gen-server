import argparse
import asyncio
from concurrent.futures import Future, ProcessPoolExecutor
import json
import logging

import struct
import sys
import time
from typing import Any, Callable

from gen_server.base_types.custom_node import custom_node_validator
from gen_server.base_types.pydantic_models import RunCommandConfig
from gen_server.config import init_config
from gen_server.globals import update_custom_nodes, update_architectures
from gen_server.tcp_server import TCPServer, RequestContext
from gen_server.worker.gpu_worker import generate_images_non_io
from gen_server.utils.cli_helpers import parse_known_args_wrapper
from gen_server.utils.extension_loader import load_extensions
from gen_server.utils.image import tensor_to_bytes
import os
from gen_server.globals import get_model_memory_manager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def request_handler(context: RequestContext):
    data = context.data()

    json_data = None
    try:
        json_data = json.loads(data.decode())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON data: {e}")
        return

    async def generate_images():
        async for [model_id, images] in generate_images_non_io(json_data):
            outputs = tensor_to_bytes(images)
            model_id_bytes = model_id.encode("utf-8")
            model_id_header = struct.pack("!I", len(model_id_bytes)) + model_id_bytes
            for output in outputs:
                total_size = struct.pack("!I", len(model_id_header) + len(output))
                context.send(total_size + model_id_header + output)

    asyncio.run(generate_images())


def run_tcp_server(config: RunCommandConfig):
    server = TCPServer(port=config.port, host=config.host)

    server.set_handler(request_handler)
    server.start(lambda addr, port: print(f"Server started on {addr}:{port}"))


def startup_extensions():
    start_time_custom_nodes = time.time()

    update_custom_nodes(
        load_extensions("cozy_creator.custom_nodes", validator=custom_node_validator)
    )

    update_architectures(load_extensions("cozy_creator.architectures"))

    print(
        f"CUSTOM_NODES loading time: {time.time() - start_time_custom_nodes:.2f} seconds"
    )


async def load_and_warm_up_models():
    model_memory_manager = get_model_memory_manager()
    model_ids = model_memory_manager.get_all_model_ids()
    warmup_models = model_memory_manager.get_warmup_models()

    logger.info(f"Starting to load and warm up {len(model_ids)} models")

    print(f"Warmup models: {warmup_models}")

    for model_id in warmup_models:
        if model_id in model_ids:
            try:
                logger.info(f"Loading and warming up model: {model_id}")
                await model_memory_manager.load(model_id, None)
                await model_memory_manager.warm_up_pipeline(model_id)
            except Exception as e:
                logger.error(f"Error loading or warming up model {model_id}: {e}")

    logger.info("Finished loading and warming up models")


async def main_async():
    run_parser = argparse.ArgumentParser(description="Cozy Creator")
    config = init_config(run_parser, parse_known_args_wrapper)

    startup_extensions()

    # Load and warm up models
    await load_and_warm_up_models()

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

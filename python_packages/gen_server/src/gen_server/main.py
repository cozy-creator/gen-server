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
from gen_server.globals import update_custom_nodes
from gen_server.tcp_server import TCPServer, RequestContext
from gen_server.executor.workflows import generate_images_non_io
from gen_server.utils.cli_helpers import parse_known_args_wrapper
from gen_server.utils.extension_loader import load_extensions
from gen_server.utils.image import tensor_to_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def request_handler(context: RequestContext):
    data = context.data()
    json_data = json.loads(data.decode())

    async def generate_images():
        async for images in generate_images_non_io(json_data, None):
            results = tensor_to_bytes(images)
            for result in results:
                result_header = struct.pack("!I", len(result))
                context.send(result_header + result)

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

    print(
        f"CUSTOM_NODES loading time: {time.time() - start_time_custom_nodes:.2f} seconds"
    )


def main():
    run_parser = argparse.ArgumentParser(description="Cozy Creator")
    config = init_config(run_parser, parse_known_args_wrapper)

    startup_extensions()
    run_tcp_server(config)


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

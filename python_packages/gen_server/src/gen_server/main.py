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
from gen_server.utils.model_downloader import ModelSource, ModelManager
from gen_server.config import get_config
from gen_server.utils.utils import serialize_config
from gen_server.model_command_handler import ModelCommandHandler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def verify_and_download_models():
    """Verify and download all models on startup"""
    config = serialize_config(get_config())
    async with ModelManager() as manager:
        for model_id, model_info in config["enabled_models"].items():
            source = ModelSource(model_info["source"])

            # Check if downloaded
            is_downloaded, variant = await manager.is_downloaded(model_id)
            print(
                f"Model {model_id} is downloaded: {is_downloaded}, variant: {variant}"
            )
            if not is_downloaded:
                logger.info(f"Downloading {model_id} from {source.type} source")
                try:
                    await manager.download_model(model_id, source)
                except Exception as e:
                    logger.error(f"Failed to download {model_id}: {e}")
                    continue

            # Handle components
            if "components" in model_info and model_info["components"] is not None:
                for comp_name, comp_info in model_info["components"].items():
                    if isinstance(comp_info, dict) and "source" in comp_info:
                        comp_source = ModelSource(comp_info["source"])
                        comp_id = f"{model_id}/{comp_name}"

                        is_downloaded, _ = await manager.is_downloaded(comp_id)
                        if not is_downloaded:
                            try:
                                await manager.download_model(comp_id, comp_source)
                            except Exception as e:
                                logger.error(
                                    f"Failed to download component {comp_id}: {e}"
                                )


def request_handler(context: RequestContext):
    data = context.data()
    logger.info(f"TCP Server received data: {data}")

    try:
        json_data = json.loads(data.decode())
        logger.info(f"Decoded JSON: {json_data}")

        # Check if this is a model management command
        if "command" in json_data:
            logger.info(f"Processing model command: {json_data['command']}")
            command_handler = ModelCommandHandler()
            try:
                # Run the async command handler in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(command_handler.handle_command(json_data))
                
                # Send response
                response_bytes = json.dumps(response).encode()
                size = struct.pack("!I", len(response_bytes))
                context.send(size + response_bytes)
            except Exception as e:
                logger.error(f"Error handling model command: {e}")
                error_response = json.dumps({
                    "status": "error",
                    "error": str(e)
                }).encode()
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
                    model_id_header = struct.pack("!I", len(model_id_bytes)) + model_id_bytes
                    for output in outputs:
                        total_size = struct.pack("!I", len(model_id_header) + len(output))
                        context.send(total_size + model_id_header + output)

            loop.run_until_complete(generate())

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON data: {e}")
        error_response = json.dumps({
            "status": "error",
            "error": "Invalid JSON"
        }).encode()
        size = struct.pack("!I", len(error_response))
        context.send(size + error_response)


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
    print("Here")
    model_memory_manager = get_model_memory_manager()
    model_ids = model_memory_manager.get_all_model_ids()
    warmup_models = model_memory_manager.get_warmup_models()

    logger.info(f"Starting to load and warm up {len(warmup_models)} models")

    print(f"Warmup models: {warmup_models}")

    for model_id in model_ids:
        if model_id in warmup_models:
            try:
                logger.info(f"Loading and warming up model: {model_id}")
                # await model_memory_manager.load(model_id, None)
                await model_memory_manager.warm_up_pipeline(model_id)
            except Exception as e:
                logger.error(f"Error loading or warming up model {model_id}: {e}")

    logger.info("Finished loading and warming up models")


async def main_async():
    run_parser = argparse.ArgumentParser(description="Cozy Creator")
    config = init_config(run_parser, parse_known_args_wrapper)

    startup_extensions()

    # Verify and download models
    # await verify_and_download_models()

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

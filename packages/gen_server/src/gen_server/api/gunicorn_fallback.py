# This is used as a fallback for Windows systems in which Gunicorn is not available;
# it's not appropriate for production use due to being a single-threaded server.

import asyncio
import multiprocessing
from threading import Event
from aiohttp.web import AppRunner, TCPSite
from typing import Optional, Any
from gen_server.config import get_server_url, set_config

from ..globals import (
    RouteDefinition,
    CheckpointMetadata,
    update_api_endpoints,
    update_checkpoint_files,
)
from ..base_types.pydantic_models import RunCommandConfig
from .api_routes import create_aiohttp_app


def start_api_server(
    job_queue: multiprocessing.Queue,
    cancel_registry: dict[str, Event],
    config: RunCommandConfig,
    checkpoint_files: dict[str, CheckpointMetadata],
    node_defs: dict[str, Any],
    extra_routes: Optional[dict[str, RouteDefinition]] = None,
):
    set_config(config)
    if extra_routes is not None:
        update_api_endpoints(extra_routes)
    update_checkpoint_files(checkpoint_files)

    aiohttp_app = create_aiohttp_app(job_queue, cancel_registry, node_defs)

    async def run_app():
        runner = AppRunner(aiohttp_app)
        await runner.setup()
        site = TCPSite(runner, config.host, config.port)
        await site.start()

        # This will keep the server running
        await asyncio.Event().wait()

    # Run the async app using asyncio
    print(f"Server is running at {get_server_url()}")
    asyncio.run(run_app())

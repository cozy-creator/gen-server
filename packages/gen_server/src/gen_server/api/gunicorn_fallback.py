# This is used as a fallback for Windows systems in which Gunicorn is not available;
# it's not appropriate for production use due to being a single-threaded server.

import asyncio
import multiprocessing
from threading import Event
from aiohttp.web import AppRunner, TCPSite
from typing import Optional, Any, Type
from ..utils.paths import get_server_url
from ..config import set_config
from ..globals import (
    RouteDefinition,
    CheckpointMetadata,
    update_api_endpoints,
    update_checkpoint_files,
    set_download_manager,
)
from ..base_types.pydantic_models import RunCommandConfig
from .api_routes import create_aiohttp_app
from ..base_types import ApiAuthenticator
from ..utils.download_manager import DownloadManager


def start_api_server(
    job_queue: multiprocessing.Queue,
    cancel_registry: dict[str, Event],
    config: RunCommandConfig,
    checkpoint_files: dict[str, CheckpointMetadata],
    node_defs: dict[str, Any],
    extra_routes: Optional[dict[str, RouteDefinition]] = None,
    api_authenticator: Optional[Type[ApiAuthenticator]] = None,
    download_manager: Optional[DownloadManager] = None,
):
    set_config(config)
    if extra_routes is not None:
        update_api_endpoints(extra_routes)
    if download_manager is not None:
        set_download_manager(download_manager)
    update_checkpoint_files(checkpoint_files)

    aiohttp_app = create_aiohttp_app(
        job_queue, cancel_registry, node_defs, api_authenticator
    )

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

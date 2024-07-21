# This is used as a fallback for Windows systems in which Gunicorn is not available;
# it's not appropriate for production use due to being a single-threaded server.

import asyncio
import multiprocessing
from aiohttp.web import AppRunner, TCPSite
from aiohttp.web_app import Application
from typing import Optional
from ..globals import RouteDefinition, CheckpointMetadata
from ..base_types.pydantic_models import RunCommandConfig
from .api_routes import create_aiohttp_app


def start_api_server(
    config: RunCommandConfig,
    job_queue: multiprocessing.Queue,
    checkpoint_files: Optional[dict[str, CheckpointMetadata]] = None,
    extra_routes: Optional[dict[str, RouteDefinition]] = None
):
    aiohttp_app = create_aiohttp_app(job_queue, config, checkpoint_files, extra_routes)
    
    async def run_app():
        runner = AppRunner(aiohttp_app)
        await runner.setup()
        site = TCPSite(runner, config.host, config.port)
        await site.start()
        
        print(f"Serving on http://{config.host}:{config.port}")
        
        # This will keep the server running
        await asyncio.Event().wait()

    # Run the async app using asyncio
    asyncio.run(run_app())


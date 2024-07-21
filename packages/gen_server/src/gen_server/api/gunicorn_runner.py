import os
import multiprocessing
import aiohttp.web
from typing import Optional, Any
from gunicorn.app.base import BaseApplication

from ..globals import RouteDefinition, CheckpointMetadata
from ..base_types.pydantic_models import RunCommandConfig
from .api_routes import create_aiohttp_app


class Application(BaseApplication):
    def __init__(self, app: aiohttp.web.Application, options: dict[str, Any] = {}):
        self.application = app
        self.options = options
        super().__init__()

    def load_config(self):
        if self.cfg is not None:
            for key, value in self.options.items():
                if (
                    hasattr(self.cfg, 'settings')
                    and key in self.cfg.settings
                    and value is not None
                ):
                    if hasattr(self.cfg, 'set'):
                        self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def get_workers_count(config: RunCommandConfig) -> int:
    if (config.environment == "prod"):
        return min((multiprocessing.cpu_count() * 2), 12)
    else:
        return 2


def start_api_server(
    config: RunCommandConfig,
    job_queue: multiprocessing.Queue,
    checkpoint_files: dict[str, CheckpointMetadata],
    extra_routes: Optional[dict[str, RouteDefinition]] = None
):
    aiohttp_app = create_aiohttp_app(job_queue, config, checkpoint_files, extra_routes)

    options = {
        "bind": f"{config.host}:{config.port}",
        "workers": get_workers_count(config),
        "worker_class": "aiohttp.GunicornWebWorker",
    }
    
    # Use gunicorn process manager as a wrapper for spawning aiohttp processes
    application = Application(aiohttp_app, options)
    application.run()
    


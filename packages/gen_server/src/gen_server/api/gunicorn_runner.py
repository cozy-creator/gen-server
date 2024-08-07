import multiprocessing
from threading import Event
import aiohttp.web
from typing import Optional, Any, Type
from ..utils.paths import get_server_url
from ..config import set_config
from gunicorn.app.base import BaseApplication

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


class Application(BaseApplication):
    def __init__(self, app: aiohttp.web.Application, options: dict[str, Any] = {}):
        self.application = app
        self.options = options
        super().__init__()

    def load_config(self):
        if self.cfg is not None:
            for key, value in self.options.items():
                if (
                    hasattr(self.cfg, "settings")
                    and key in self.cfg.settings
                    and value is not None
                ):
                    if hasattr(self.cfg, "set"):
                        self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def get_workers_count(config: RunCommandConfig) -> int:
    if config.environment == "prod":
        return min((multiprocessing.cpu_count() * 2), 12)
    else:
        return 2


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

    options = {
        "bind": f"{config.host}:{config.port}",
        "workers": get_workers_count(config),
        "worker_class": "aiohttp.GunicornWebWorker",
    }

    # Use gunicorn process manager as a wrapper for spawning aiohttp processes
    print(f"Server is running at {get_server_url()}")
    application = Application(aiohttp_app, options)
    application.run()

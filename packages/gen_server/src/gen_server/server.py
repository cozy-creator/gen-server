import multiprocessing
from typing import Optional

from gen_server.api import create_app
from gunicorn.app.base import BaseApplication


class Application(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def workers_count():
    return (multiprocessing.cpu_count() * 2) + 1


def run_server(
    host: str = "localhost",
    port: int = 8881,
    queue: Optional[multiprocessing.Queue] = None,
):
    options = {
        "bind": f"{host}:{port}",
        "workers": workers_count(),
        "worker_class": "aiohttp.GunicornWebWorker",
    }

    application = Application(create_app(queue), options)
    application.run()

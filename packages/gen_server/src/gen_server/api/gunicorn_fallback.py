# This is used as a fallback for Windows systems in which Gunicorn is not available;
# it's not appropriate for production use due to being a single-threaded server.

import multiprocessing
from waitress import serve
from ..globals import RunCommandConfig
from . import create_aiohttp_app


def start_server(
    config: RunCommandConfig,
    job_queue: multiprocessing.Queue,
):
    aiohttp_app = create_aiohttp_app(job_queue)
    
    serve(aiohttp_app, host=config.host, port=config.port)

import platform
from .api_routes import api_routes_validator, create_aiohttp_app


if platform.system() == 'Windows':
    # Windows
    from .gunicorn_fallback import start_api_server
else:
    # Unix-like systems (Linux, macOS, BSD, etc.)
    from .gunicorn_runner import start_api_server


__all__ = [
    "api_routes_validator",
    "create_aiohttp_app",
    "start_api_server",
]

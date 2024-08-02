from typing import Optional, Any

from aiohttp import web


class ApiAuthenticator:
    def __init__(self):
        pass

    def authenticate(self, _request: web.Request) -> Optional[dict]:
        pass

    def is_authenticated(self, _request: web.Request) -> bool:
        pass


def api_authenticator_validator(plugin: Any) -> bool:
    try:
        return issubclass(plugin, ApiAuthenticator)

    except TypeError:
        print(f"Invalid plugin type: {plugin}")
        return False

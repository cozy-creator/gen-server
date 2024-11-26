from typing import Optional, Any

from aiohttp import web


class ApiAuthenticator:
    def __init__(self):
        pass

    def authenticate(self, _request: web.Request) -> Optional[Any]:
        pass


class AuthenticationError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


def api_authenticator_validator(plugin: Any) -> bool:
    try:
        return issubclass(plugin, ApiAuthenticator)

    except TypeError:
        print(f"Invalid plugin type: {plugin}")
        return False

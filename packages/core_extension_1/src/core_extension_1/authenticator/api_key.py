import sqlite3
from typing import Optional

from aiohttp import web

from gen_server import ApiAuthenticator


class ApiKeyAuthenticator(ApiAuthenticator):
    def __init__(self):
        super().__init__()
        self._db = sqlite3.connect("main.db")

    def authenticate(self, request: web.Request) -> Optional[bool]:
        header = request.headers.get("x-api-key")
        if header is None:
            return None

    #
    # def _verify_token_with_db(self, raw_decrypted_token: str):
    #     cursor = self._db.cursor()
    #     cursor.execute(
    #         "SELECT * FROM api_keys WHERE api_key = ?", (raw_decrypted_token,)
    #     )
    #     result = cursor.fetchone()
    #     cursor.close()
    #     return result

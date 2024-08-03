import logging
import sqlite3
import hashlib
import os
from typing import Any, Dict, Optional

from aiohttp import web

from gen_server.base_types import ApiAuthenticator
from gen_server.base_types.authenticator import AuthenticationError

logger = logging.Logger(__name__)


class ApiKeyAuthenticator(ApiAuthenticator):
    def __init__(self):
        super().__init__()
        self._conn = sqlite3.connect(os.environ.get("DATABASE_URL", "main.db"))
        self._setup_table()

    def authenticate(self, request: web.Request) -> Optional[bool]:
        print("-----")
        key = request.headers.get("x-api-key")
        if key is None:
            raise AuthenticationError("API key is missing in request")

        key_hash = self._hash_key(key)
        result = self._retrieve_key(key_hash)

        if result is None:
            raise AuthenticationError("API key is invalid")
        if result["is_revoked"]:
            raise AuthenticationError("API key has been revoked")

        return True

    # ========== Internal (non-standard) methods ==========

    def _retrieve_key(self, key_hash: str) -> Optional[Dict[str, Any]]:
        query = """
            SELECT id, key_hash, is_revoked, created_at 
            FROM api_keys 
            WHERE key_hash = ?;
        """
        try:
            with self._conn as conn:
                cursor = conn.execute(query, (key_hash,))
                result = cursor.fetchone()
        except sqlite3.Error as e:
            logger.error("Failed to retrieve key from database", e)
            return None

        if result:
            return {
                "id": result[0],
                "key_hash": result[1],
                "is_revoked": result[2],
                "created_at": result[3],
            }

        return None

    def _insert_api_key(self, key: str) -> None:
        key_hash = self._hash_key(key)
        query = """
            INSERT INTO api_keys (key_hash) 
            VALUES (?);
        """
        try:
            with self._conn as conn:
                conn.execute(query, (key_hash,))
        except sqlite3.Error as e:
            logger.error("Failed to insert key into the database", e)

    def _setup_table(self) -> None:
        try:
            with self._conn as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key_hash TEXT NOT NULL,
                        is_revoked BOOLEAN NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_key_hash ON api_keys(key_hash);"
                )
        except sqlite3.Error as e:
            logger.error("Failed to setup the database", e)

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def __del__(self):
        if self._conn:
            self._conn.close()

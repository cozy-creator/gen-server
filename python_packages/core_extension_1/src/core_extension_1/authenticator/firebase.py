import json
import os
from typing import Optional

from aiohttp import web
from firebase_admin import credentials, initialize_app, firestore, auth

from gen_server.base_types import ApiAuthenticator
from dotenv import load_dotenv

load_dotenv()


class FirebaseAuthenticator(ApiAuthenticator):
    def __init__(self):
        service_account = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        if service_account is None:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT env var is required.")

        credential = credentials.Certificate(json.loads(service_account))
        self.app = initialize_app(credential=credential)
        self.database = firestore.client()

    def authenticate(self, request: web.Request) -> Optional[dict]:
        authorization = request.headers.get("Authorization")
        if authorization is None:
            return None

        token = authorization.split("Bearer ")[-1]
        return self._verify_token(token)

    def is_authenticated(self, request: web.Request) -> bool:
        return request.get("user") is not None

    def _verify_token(self, token: str):
        try:
            raw_user = auth.verify_id_token(token)

            return {
                "uid": raw_user["uid"],
                "name": raw_user["name"],
                "email": raw_user["email"],
            }
        except Exception:
            return None

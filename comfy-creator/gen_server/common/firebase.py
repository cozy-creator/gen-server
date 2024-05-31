from pprint import pprint
from typing import Optional, Union

from firebase_admin import firestore, credentials, initialize_app, App, auth
from pydantic import BaseModel

app: Optional[App] = None
database: Optional[firestore.firestore.Client] = None


class FirebaseIdentity(BaseModel):
    identities: dict[str, list[str]]
    sign_in_provider: str


class FirebaseUser(BaseModel):
    aud: str
    exp: int
    iat: int
    iss: str
    sub: str
    uid: str
    name: str
    email: str
    user_id: str
    auth_time: int
    email_verified: bool
    firebase: FirebaseIdentity


def initialize(service_account: str):
    global app, database

    credential = credentials.Certificate(service_account)
    app = initialize_app(credential=credential)
    database = firestore.client()


def verify_token(token: str):
    try:
        raw_user = auth.verify_id_token(token)
        user = FirebaseUser(**raw_user)
        return user
    except Exception as e:
        return None


class Firebase:
    def __init__(self, service_account: Union[str, dict]):
        self.app = initialize_app(credential=credentials.Certificate(service_account))
        self.database = firestore.client()


class Collection:
    def __init__(self, name: str):
        self.collection = database.collection(name)

    def get(self, document_id: str):
        return self.collection.document(document_id).get()

    def create(self, document_id: Optional[str], data: dict):
        if document_id is None:
            return self.collection.add(data)
        return self.collection.document(document_id).set(data)

    def update(self, document_id: str, data: dict):
        return self.collection.document(document_id).update(data)

    def delete(self, document_id: str):
        return self.collection.document(document_id).delete()

    def query(self, field: str, operator: str, value: any):
        return self.collection.where(field, operator, value).stream()

    def all(self):
        return self.collection.stream()

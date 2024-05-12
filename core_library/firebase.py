from typing import Optional

from firebase_admin import firestore, credentials, initialize_app, App, auth

app: Optional[App] = None
database: Optional[firestore.firestore.Client] = None

"""
Example user:
{'aud': 'capsules-dev',
 'auth_time': 1715372240,
 'email': 'abdulrahmanyusuf125@gmail.com',
 'email_verified': True,
 'exp': 1715442622,
 'firebase': {'identities': {'apple.com': ['000916.fc58f0024cde4ebc961e48f4268de25b.1534'],
                             'email': ['abdulrahmanyusuf125@gmail.com'],
                             'google.com': ['113032764133656015970']},
              'sign_in_provider': 'google.com'},
 'iat': 1715439022,
 'iss': 'https://securetoken.google.com/capsules-dev',
 'name': 'Abdulrahman Yusuf',
 'sub': 'T1SDzZztgPgHQKcGROe6BFqy9jJ2',
 'uid': 'T1SDzZztgPgHQKcGROe6BFqy9jJ2',
 'user_id': 'T1SDzZztgPgHQKcGROe6BFqy9jJ2'}
"""


def initialize(service_account: str):
    global app, database

    credential = credentials.Certificate(service_account)
    app = initialize_app(credential=credential)
    database = firestore.client()


def verify_token(token: str):
    try:
        user = auth.verify_id_token(token)
        return user
    except Exception as e:
        return None

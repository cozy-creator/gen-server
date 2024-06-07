import os
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")


class S3Config(BaseModel):
    access_key: str
    region_name: str
    bucket_name: str
    endpoint_url: str
    secret_access_key: str


class PulsarConfig(BaseModel):
    tenant: str
    rest_url: str
    service_url: str
    job_queue_namespace: str
    job_cancel_namespace: str
    user_history_namespace: str
    job_snapshot_namespace: str
    auth_token: Optional[str] = None


class FirebaseConfig(BaseModel):
    service_account: str


class Settings(BaseSettings):
    environment: str
    s3: S3Config
    pulsar: PulsarConfig
    firebase: FirebaseConfig

    read_chunk_size: int = 5 * 1024 * 1024
    upload_chunk_size: int = 5 * 1024 * 1024

    def is_production(self):
        return self.environment == "production"

    model_config = SettingsConfigDict(env_file=ENV_PATH, env_nested_delimiter="__")


settings = Settings()

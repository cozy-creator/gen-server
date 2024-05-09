from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class S3Config(BaseModel):
    access_key: str
    region_name: str
    bucket_name: str
    endpoint_url: str
    secret_access_key: str


class Settings(BaseSettings):
    environment: str
    s3: S3Config

    read_chunk_size: int = 5 * 1024 * 1024
    upload_chunk_size: int = 5 * 1024 * 1024

    def is_production(self):
        return self.environment == "production"

    model_config = SettingsConfigDict(env_file="../.env", env_nested_delimiter="__")


settings = Settings()

from . import CustomNode
from .types import Architecture
from typing import Type, Dict
from dotenv import load_dotenv
import os
import boto3

API_ENDPOINTS: dict = {}
"""
Something
"""

ARCHITECTURES: dict[str, Type[Architecture]] = {}
"""
Global class containing all architecture definitions
"""

CUSTOM_NODES: dict[str, Type[CustomNode]] = {}
"""
Something
"""

WIDGETS: dict = {}
"""
TO DO
"""

env = None
s3 = None
s3_bucket_name = None
s3_endpoint_fqdn = None
s3_access_key = None
s3_secret_key = None

def configure_environment(env_file: str = None, config_dict: Dict[str, str] = None):
    """
    Configures the environment based on provided parameters.
    """
    global env, s3, s3_bucket_name, s3_endpoint_fqdn, s3_access_key, s3_secret_key

    # Priority: 1. config_dict, 2. env_file, 3. default .env
    if config_dict:
        print("Loading from configuration dictionary")
        env = config_dict.get('env', 'local').lower()
        if 's3' in config_dict:
            s3_bucket_name = config_dict['s3'].get('bucket_name', '')
            s3_endpoint_fqdn = config_dict['s3'].get('endpoint_url', '')
            s3_access_key = config_dict['s3'].get('access_key', '')
            s3_secret_key = config_dict['s3'].get('secret_access_key', '')
    elif env_file:
        print(f"Loading from {env_file}")
        load_dotenv(dotenv_path=env_file, override=True)
        env = os.getenv('ENVIRONMENT', 'local').lower()
        s3_bucket_name = os.getenv('S3__BUCKET_NAME', '')
        s3_endpoint_fqdn = os.getenv('S3__ENDPOINT_URL', '')
        s3_access_key = os.getenv('S3__ACCESS_KEY', '')
        s3_secret_key = os.getenv('S3__SECRET_ACCESS_KEY', '')
    else:
        print("Loading default .env file")
        load_dotenv(override=True)
        env = os.getenv('ENVIRONMENT', 'local').lower()
        s3_bucket_name = os.getenv('S3__BUCKET_NAME', '')
        s3_endpoint_fqdn = os.getenv('S3__ENDPOINT_URL', '')
        s3_access_key = os.getenv('S3__ACCESS_KEY', '')
        s3_secret_key = os.getenv('S3__SECRET_ACCESS_KEY', '')

    print(f"Environment: {env}")
    print(f"S3 Bucket: {s3_bucket_name}")

    # Set up S3 client if needed
    if env == 'production':
        if not all([s3_bucket_name, s3_endpoint_fqdn, s3_access_key, s3_secret_key]):
            raise ValueError('Missing S3 configuration in production mode')
        s3 = boto3.client('s3',
                            endpoint_url=f'https://{s3_endpoint_fqdn}',
                            aws_access_key_id=s3_access_key,
                            aws_secret_access_key=s3_secret_key,
                            region_name=os.getenv("S3__REGION_NAME"))

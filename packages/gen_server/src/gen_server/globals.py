import os
import json
import boto3
from typing import Type, Dict, Optional, List, Any
from dotenv import load_dotenv
from dataclasses import dataclass, field
from . import CustomNode
from .base_types import Architecture


DEFAULT_WORKSPACE_DIR = '~/.comfy-creator/models'
DEFAULT_MODELS_DIRS = ['~/.comfy-creator/models']

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

@dataclass
class ComfyConfig:
    filesystem_type: Optional[str] = None
    workspace_dir: Optional[str] = None
    models_dirs: List[str] = field(default_factory=list)
    s3: dict = field(default_factory=dict) 

comfy_config = ComfyConfig()
"""
Global configuration dataclass for the comfy-creator application
"""

def initialize_config(env_path: Optional[str] = None, config_path: Optional[str] = None):
    """
    Load the .env file and config file specified into the global configuration dataclass
    """
    global comfy_config
    
    # These env variables can be access using os.environ or os.getenv
    if env_path:
        print(f"Loading from {env_path}")
        load_dotenv(dotenv_path=env_path, override=True)
    
    # Parse the config-file
    config_dict = {}
    if config_path:
        try:
            with open(config_path, 'r') as file:
                config_dict = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file {config_path} was not found.")
        except json.JSONDecodeError:
            print("Invalid JSON format in the configuration file.")
    
    # Find our directories
    comfy_config.filesystem_type = config_dict.get("filesystem_type", 'LOCAL')
    comfy_config.workspace_dir = os.path.expanduser(config_dict.get("workspace_dir", DEFAULT_WORKSPACE_DIR))
    comfy_config.models_dirs = [os.path.expanduser(path) for path in config_dict.get("models_dirs", DEFAULT_MODELS_DIRS)]
    
    #  Load S3 credentials and instantiate the S3 client
    s3_credentials = config_dict.get("s3_credentials")
    if s3_credentials:
        s3_endpoint_fqdn = s3_credentials.get("endpoint_fqdn")
        s3_access_key = s3_credentials.get("access_key")
        bucket_name = s3_credentials.get("bucket_name")
        folder = s3_credentials.get("folder")
        s3_secret_key = os.getenv("S3_SECRET_KEY")
        
        required_fields = [s3_endpoint_fqdn, s3_access_key, s3_secret_key, bucket_name, folder]
        if None in required_fields:
            print("Error: Missing required S3 configuration fields.")
        else:
            comfy_config.s3["bucket_name"] = bucket_name
            comfy_config.s3["folder"] = folder
            comfy_config.s3["url"] = f"https://{bucket_name}.{s3_endpoint_fqdn}"
            
            try:
                comfy_config.s3["client"] = boto3.client(
                    "s3",
                    region_name=s3_endpoint_fqdn.split('.')[0],
                    endpoint_url=f"https://{s3_endpoint_fqdn}",
                    aws_access_key_id=s3_access_key,
                    aws_secret_access_key=s3_secret_key,
                )
                print("S3 client configured successfully.")
            except Exception as e:
                print(f"Error configuring S3 client: {e}")


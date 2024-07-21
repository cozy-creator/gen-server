ENV_TEMPLATE = """
# This is an example .env file; it provides global configurations for the gen-server.
# Warning: DO NOT share this file or commit it to your repository as it may contain
# sensitive fields.

# Location where the gen-server will be hosted
HOST=localhost
PORT=8881

# Not currently in use; may remove
FILESYSTEM_TYPE=LOCAL

# Assets will be loaded from and saved into this workspace directory.
WORKSPACE_PATH=~/.cozy-creator
ASSETS_PATH=~/.cozy-creator/assets
MODELS_PATH=~/.cozy-creator/models

# Where the gen-server will search for model-files (state-dictionaries
# such as .safetensors).
AUX_MODELS_PATHS='["C:/git/ComfyUI/models"]'

# Remote S3 bucket to read from / write to. Useful in production
# Note that it can be a JSON-format string...
S3='{"endpoint_url": "https://nyc3.digitaloceanspaces.com", "access_key": "DO00W9N964WMQC2MV6JK", "secret_key": "***********", "region_name": "nyc3", "bucket_name": "voidtech-storage-dev", "folder": "public", "public_url": "https://storage.cozy.dev"}'

# ...or written out as individual variables
COZY_S3__ENDPOINT_URL=https://nyc3.digitaloceanspaces.com
COZY_S3__ACCESS_KEY=DO00W9N964WMQC2MV6JK
COZY_S3__SECRET_KEY=***********
COZY_S3__REGION_NAME=nyc3
COZY_S3__BUCKET_NAME=voidtech-storage-dev
COZY_S3__FOLDER=public
COZY_S3__PUBLIC_URL=https://storage.cozy.dev

# You can also include custom environment variables not built into our gen-server
MY_API_KEY=this-also-works!
"""

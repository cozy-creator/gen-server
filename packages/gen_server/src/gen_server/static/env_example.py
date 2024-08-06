ENV_TEMPLATE = """
# This is an example .env file; it provides global configurations for the gen-server.
# Warning: DO NOT share this file or commit it to your repository as it may contain
# sensitive fields.

# To configure where cozy-creator will locally store data.
# In particular, generated images, videos, audio, and downloaded models will be stored in this folder.
# Defaults to ~/.cozy-creator unless XDG_DATA_HOME is specified.
COZY_HOME=~/.cozy-creator
ASSETS_PATH=~/.cozy-creator/assets
MODELS_PATH=~/.cozy-creator/models

# Where the gen-server will search for extra model-files (state-dictionaries
# such as .safetensors), other than the MODELS_PATH.
AUX_MODELS_PATHS='["C:/git/ComfyUI/models"]'

# HTTP-location where the gen-server will be accessible from
HOST=localhost
PORT=8881

# enum for where files should be served from: 'LOCAL' or 'S3'
# LOCAL urls look like http://localhost:8881/media/<file-name>
# S3 urls look like https://storage.cozy.dev/<file-name>
FILESYSTEM_TYPE=LOCAL

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


### Hugging Face Variables

# Used in place of any hugging-face token stored locally
HF_TOKEN=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
HF_HUB_CACHE=~/.cache/huggingface/hub


# You can also include custom environment variables not built into our gen-server
MY_API_KEY=this-also-works!
"""

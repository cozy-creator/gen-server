### Dev Usage

To run the main application package, navigate to the `/packages/gen_server` folder:

- Run `pip install -e .`; this adds the `comfy-creator` command to your path.
- Run `comfy-creator --config=config.json`

This will generate a `.egg-info` folder. The code will be installed in editable format, which means any changes to the code will be immediately reflected the next time you run it.

Repeat this for all packages you want to install. The other packages extend functionality of the gen-server by specifying an entry-point group-name; they will be dynamically imported at runtime by the main gen-server application.


### Building for Distribution

If you don't already have `build`, you can use pip to install it. Then navigate into the package directory you want to build, and run:

- `python -m build` to build the repository.

This will produce a `/dist` folder, containing a `.whl` (wheel) file and a `.tar.gz` file. The wheel file packages the library in a format that can be easily installed using pip, while the `.tar.gz` file contains the source code. To install the wheel file, use the command `pip install file_name.whl`.

None of our packages currently use any C-APIs, and hence do not need to be recompiled for different environments.


### Running in production

`comfy-creator` flags:

--config path/to/config.json
--env path/to/.env

If these are not specified, comfy-creator will use default values.

Example config.json:
```
    {
        "filesystem_type": "S3",
        "workspace_dir": "~/.comfy-creator",
        "models_dirs": [
            "~/.comfy-creator/models",
            "~/.comfy-creator/models/stable-diffusion"
        ],
        "s3_credentials": {
            "bucket_name": "voidtech-storage-dev",
            "endpoint_fqdn": "nyc3.digitaloceanspaces.com",
            "folder": "public",
            "access_key": "DO00W9N964WMQC2MV6JK"
        }
    }
```

### Configuration Details

- **filesystem_type**: Specifies the type of file system to use. Options are `LOCAL` or `S3`.
- **workspace_dir**: The default directory where files will be saved and loaded from. Defaults to your home directory at `~/.comfy-creator`.
- **models_dirs**: Directories where `comfy-creator` will search for checkpoint files. Includes paths to general models and specific models like stable diffusion. Defaults to `~/.comfy-creator/models`.
- **s3_credentials**: Contains the credentials for reading files from and writing files to an S3 bucket; only used if filesystem_type is set to S3.
  - **bucket_name**: The name of the S3 bucket.
  - **endpoint_fqdn**: The fully qualified domain name of the S3 endpoint.
  - **folder**: The specific folder within the S3 bucket where files are stored.
  - **access_key**: The access key for S3 bucket authentication. Note: The secret key should be stored inside of the .env file as `S3_SECRET_KEY`.

> **Note:** The gen-server currently does not check any form of authentication on requests. Use another server to authenticate requests prior to forwarding them to the gen-server, or we need to implement authentication still.


### Docker Build

In the root of this repo, run:

`docker build -t comfy-creator/gen-server:0.0.4 .`

### Docker Run

`docker run -p 8080:8080 comfy-creator/gen-server:0.0.4`
`docker run --env-file ./.env --volume ./config.json:/app/config.json`


### Old dependencies:

I'm keeping these here for notes:

dependencies = [
    "aiohttp>=3.9.5",
    "blake3>=0.4.1",
    "boto3>=1.34.99",
    "diffusers",
    "firebase-admin>=6.5.0",
    "grpcio",
    "protobuf",
    "pulsar-client>=3.5.0",
    "pydantic>=2.7.1",
    "pydantic-settings>=2.2.1",
    "requests",
    "safetensors",
    "spandrel",
    "transformers",
    "uuid~=1.30"
]



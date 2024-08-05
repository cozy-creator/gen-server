### Installation

To install the gen-server, navigate to the `/packages/gen_server` folder:

- Run `pip install -e .`; this adds the `cozy` command to your path.

This will generate a `.egg-info` folder. The code will be installed in editable format, which means any changes to the code will be immediately reflected the next time you run it. 
> **Note:** New modules or renamed modules will not be reflected until you run `pip install -e .` again.

Repeat this for all packages you want to install. The other packages extend functionality of the gen-server by adding plugins to the current environment; on startup the gen-server will automatically load all compatible plugins in its environment.

You'll also want to install PyTorch; the specific installation instructions vary depending upon your system; see here for details [here](https://pytorch.org/get-started/locally/). Be sure to use a CUDA installation if your system supports it.

### Running Gen-Server Locally

Cozy Creator has the following sub-commands:

- `cozy web-build` to build the web-bundle.
- `cozy run` to start the gen-server.

On startup, the Cozy Gen Server loads its configuration variables from several possible sources; this is the order of precedence:

1. Command-line arguments (ex; `cozy run --s3_bucket_name=cozy-storage`)
2. Environment variables (ex; `export COZY_S3_BUCKET_NAME=cozy-storage`)
3. .env file (ex; `cozy run --env-file "./.env"`)
4. Secrets directory (ex; `cozy run --secrets-dir="/run/secrets"`)
5. Default settings

Run `cozy run --help` and `cozy web-build --help` to see the full list of available cli-flags.

Note:
- Underscores and dashes are equivalent for all commands and flags (e.g., `cozy web-build` and `cozy web_build` both work).
- CLI variable-names are case-insensitive.
- CLI variables can be specified in two ways:
  - With an equals sign: `cozy run port=3000`
  - With a space: `cozy run port 3000`
- Objects can be specified as JSON strings, example:

```sh
cozy run s3='{"endpoint_url": "https://nyc3.digitaloceanspaces.com", "access_key": "DO00W9N964WMQC2MV6JK", "secret_key": "*******", "region_name": "nyc3", "bucket_name": "storage", "folder": "public"}'
```

### Using Environment Variables

Environment variables must have the `COZY_` prefix added in order to prevent collisions with other environment variables. For example `export COZY_PORT=9000` will work.

See our .env.example file for all available environment variables. By default, when you run `cozy run` without the `--env-file` flag,it will attempt to load the `.env` file in the workspace path (which may also be specified or could be default). You can specify its location using the `--env-file` flag; 

```sh
cozy run --env-file="~/.cozy-creator/.env"
```

For the secrets-dir, use:

```sh
cozy run --secrets-dir="/run/secrets"
```

### Building Cozy Creator For Distribution

If you don't already have `build`, you can use pip to install it. Then navigate into the package directory you want to build, and run:

- `python -m build` to build the repository.

This will produce a `/dist` folder, containing a `.whl` (wheel) file and a `.tar.gz` file. The wheel file packages the library in a format that can be easily installed using pip, while the `.tar.gz` file contains the source code. To install the wheel file, use the command `pip install file_name.whl`.

None of our packages currently use any C-APIs, and hence do not need to be recompiled for different environments.


### Docker Build

Build the Cozy Graph editor, and place it inside of `/web/`, like `cozy-graph-editor-0.0.1.tgz`. This will be used as a dependency when building the front-end. You can install it by running `yarn add file:cozy-graph-editor-0.0.1.tgz` in the `/web` directory, or wherever you placed it the file. In your package.json file you'll see a dependency like this:

`"@cozy-creator/graph-editor": "./cozy-graph-editor-0.0.1.tgz",`

Then in the root of this repo, run:

```sh
docker build -t cozycreator/gen-server:0.2.5 .
```

### Docker Run

Note that in windows, we add the `MSYS_NO_PATHCONV=1` flag because Windows dcommand line doesn't know how to interpet paths. Note that in Docker, destination route paths must be absolute, so `~/.cozy-creator` won't work; you must use `/root/.cozy-creator` or whever your user's home directory is. This command below assumes the user is inside of the container is running as root:

WSL2 version:
In Windows, run `wsl` to enter Windows Subsystem for Linux. And then run:

```sh
docker run \
  --env-file=/root/.cozy-creator/.env.local \
  -p 8881:8881 -p 8888:8888 \
  -v /root/.cozy-creator:/root/.cozy-creator \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --gpus=all \
  cozycreator/gen-server:0.2.5
```

Windows version:
By default, docker runs in WSL2, and WSL2 is extremely slow at reading from the windows filesystem. Meaning that when you mount a windows directory into the docker container, the load-time for SDXL's 5.1GB Unet goes from 12 seconds -> 90 seconds.

To fix this, you can access your WSL installation by typing in `\\wsl$` into the windows file explorer, and then navigating to your Linux installation (probably Ubuntu). Then in the root directory of your Ubuntu installation, place your models inside of /root/.cozy-creator/models, and then load from there. Your docker container will load models directly from your Linux filesystem, bypassing Windows and speeding up the process.
```sh
MSYS_NO_PATHCONV=1 docker run \
  --env-file=.env.local \
  -p 8881:8881 -p 8888:8888 \
  -v ~/.cozy-creator:/root/.cozy-creator \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --gpus=all \
  cozycreator/gen-server:0.2.5
```

You can set environment variables manually by using `-e`; just remember to prepend them with `COZY_` first. When you specify an .env-file in Docker's run command, Docker inserts them as environment variables into the container, meaning all of the keys inside of your `.env` file should be prefixed with `COZY_` to have them work as expected. Some other flag-usage examples:

```sh
docker run \
  -e COZY_HOST=0.0.0.0 \
  -p 9000:9000 \
  -e COZY_PORT=9000 \
  -v "C:/git/ComfyUI/models":/models \
  -e COZY_AUX_MODELS_PATHS='["/models"]' \
  cozycreator/gen-server:0.2.2
```

### Hugging Face Hub Model Caching

By default Cozy Creator stores models inside of `~/.cozy-creator/models`, but we also use Hugging Face Hub. If you have models downloaded from Hugging Face Hub by another program, we'll be able to load them in from your local filesystem when they're needed, and if they're not present we'll use Hugging Face Hub to download them.

Hugging Face by default caches all models in `~/.cache/huggingface/hub`. You can change this default location by setting the environment variable `HF_HOME` to a new path. You can also set the environment variable flag `HF_HUB_OFFLINE=1` if you don't want to download models from the internet at all.

See [here for more details.](https://huggingface.co/docs/transformers/main/en/installation#cache-setup)


### Gen-Server Spec

- Plugin system: add aiohttp routes, architecture-definitions, custom-nodes (frontend and backend), and widgets (frontend)
- Searches filesystem for models (checkpoints) available and matches them to architecture definitions
- Endpoint to receive inference requests (prebuilt workflows for now)
- Location to save files to (locally or remotely)
- Endpoint to serve files (not a great use for aiohttp but it should work)

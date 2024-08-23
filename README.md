### Cozy Creator
Cozy Creator is a generative AI engine that allows you to create and run generative AI models on your own computer or in the cloud. It is designed to be easy to use and accessible to everyone, regardless of their technical expertise.

This repository contains the source code for Cozy Creator, which includes the following packages:

- **gen-server**: The core engine that runs the generative AI models.
- **go-cozy**: A Go based API server for interacting with the gen-server.
- **core_extensions**: A collection of extensions that add new functionality to the gen-server.
- **web**: A web-based user interface for managing and running models and workflows.

### Installation

To install Cozy Creator, you need to get the source code and install the gen-server. The gen-server is the core engine that runs the generative AI models, while the other packages extend functionality of the gen-server by adding plugins to the current environment; on startup the gen-server will automatically load all compatible plugins in its environment.

Before you can install the gen-server, you need to install Python and the required dependencies. Most of the dependencies needed will be installed automatically when you install the gen-server, but you may need to install pytorch and other dependencies manually.

Installing Pytorch is a bit more involved, as it depends on your system. See [here](https://pytorch.org/get-started/locally/) for instructions specific to your system.

#### Gen-server

To install the gen-server, navigate to the `/packages/gen_server` folder and run:
```
pip install -e .
```

This adds the `cozy` command to your path and generates a `.egg-info` folder. The code will be installed in editable format, which means any changes to the code will be immediately reflected the next time you run it.

#### Core Extensions

To install the core extensions, navigate to the `/packages/core_extensions` folder and run:
```
pip install -e .
```
This will install the extensions in editable format and they will be automatically loaded when you start the gen-server.

> **Note:** New modules or renamed modules will not be reflected until you run `pip install -e .` again.


#### Go-Cozy
<!-- TODO: add instructions for installing go-cozy -->

#### Web
The web package is a web-based user interface for managing and running models and workflows. To install it, navigate to the `/web` folder and run:
```
yarn install
```
You can also install the dependencies with `npm install` or any other package manager of your choice.


Then, to build the web bundle, run:
```
yarn build
```


### Logging into Hugging Face

To access gated repositories and models on Hugging Face, you must log in using the Hugging Face CLI. Follow these steps:

1. Ensure you have the Hugging Face `cli` library installed:
   ```
   pip install -U "huggingface_hub[cli]"
   ```

2. Open a terminal or command prompt.

3. Run the following command:
   ```
   huggingface-cli login
   ```

4. You will be prompted to enter your Hugging Face access token. If you don't have one:
   - Go to https://huggingface.co/
   - Log in to your account (or create one if you haven't already)
   - Click on your profile picture and go to "Settings"
   - In the left sidebar, click on "Access Tokens"
   - Create a new token with the necessary permissions

5. Copy your access token and paste it into the terminal when prompted.

6. Press Enter. You should see a message confirming successful login.

After logging in, you'll be able to access gated repositories and models in your Cozy Creator workflows that require authentication.

Note: Your login credentials will be stored securely on your machine. You only need to perform this login process once, unless you need to change accounts or your token expires.


### Running Gen-Server Locally

After installing Cozy Creator, you want to run the gen-server and get it running. To do this, you need to start the gen-server with the command below:

```
cozy run
```

On startup, the Cozy Gen Server loads its configuration variables from several possible sources; this is the order of precedence:

1. Command-line arguments (ex; `cozy run --environment=development`)
2. Environment variables (ex; `export ENVIRONMENT=development`)
3. .env file (ex; `cozy run --env-file "./.env"`)
4. Secrets directory (ex; `cozy run --secrets-dir="/run/secrets"`)
5. Yaml config file (ex; `cozy run --config-file ./config.yaml`)
5. Default settings


### Cozy CLI

As mentioned above, cozy-creator has a CLI that can be used to run the gen-server. But it also has a few other useful commands, they include:

- `cozy download` to download models onto the local machine (currently only supports HuggingFace models).
- `cozy build-web` to build the web-bundle.

To see the full list of available commands and flags, run:
```
cozy --help
```

and to see the full list of available flags for a specific command, run: … `cozy <COMMAND> --help` e.g:
```
cozy download --help
```

Note:
- Underscores and dashes are equivalent for all commands and flags (e.g., `cozy build-web` and `cozy build_web` both work).
- CLI variable-names are case-insensitive.
- CLI variables can be specified in two ways:
  - With an equals sign: `cozy run --port=3000`
  - With a space: `cozy run --port 3000`
- Objects can be specified as JSON strings, example:

```sh
cozy run s3='{"endpoint_url": "https://nyc3.digitaloceanspaces.com", "access_key": "DO00W9N964WMQC2MV6JK", "secret_key": "*******", "region_name": "nyc3", "bucket_name": "storage", "folder": "public"}'
```

### Configuration

Cozy Creator supports multiple configuration formats:

1. Environment variables: These can be set directly or using a `.env` file.
2. YAML config file: For setting configuration variables.
3. Secrets directory: Used to load secrets from a specified location.

The most commonly used format is the `.env` file for setting environment variables. However, you can also use a YAML config file for more structured configuration settings.

By default, both of these files are loaded from the `~/.cozy-creator` directory. But you can specify a different location using the, `--env-file`, `--config-file` flag or the `--secrets-dir` flag.

#### Environment Variables

The `.env` file is used to set environment variables. As previously mentioned, they are loaded from the `~/.cozy-creator` directory unless you specify a different location using the `--env-file` flag.

Environment variables must have the `COZY_` prefix added in order to prevent collisions with other environment variables. For example `export COZY_PORT=9000` will work.

See our .env.example file for all available environment variables. Example:

```
cozy run --env-file="~/.cozy-creator/.env"
```

#### Yaml Config File

The yaml config file is used to set configuration variables. It is loaded from the `~/.cozy-creator` directory unless you specify a different location using the `--config-file` flag.

Example:
```
cozy run --config-file="~/.cozy-creator/config.yaml"
```

#### Secrets Directory

The secrets directory is used to load secrets from a directory. It is useful for storing sensitive information.

For the secrets-dir, use:

```sh
cozy run --secrets-dir="/run/secrets"
```

### Building Cozy Creator For Distribution

If you don't already have `build`, you can use pip to install it. Then navigate into the package directory you want to build, and run:

- `python -m build` to build the repository.

This will produce a `/dist` folder, containing a `.whl` (wheel) file and a `.tar.gz` file. The wheel file packages the library in a format that can be easily installed using pip, while the `.tar.gz` file contains the source code. To install the wheel file, use the command `pip install file_name.whl`.

None of our packages currently use any C-APIs, and hence do not need to be recompiled for different environments.


### Building for Docker

#### Cozy Graph Editor
To build the Cozy Creator Docker image, you need to build the the Cozy graph editor and place it inside of `/web/`, like `cozy-graph-editor-0.0.1.tgz`. This will be used as a dependency when building the front-end.

The Cozy graph editor is a separate component that is not part of the Cozy Creator packages but you can find it [here](https://github.com/cozy-creator/graph-editor). 

You can clone and install it and then generate the `tgz` file by running:
```
yarn pack
```
You can also use any other package manager of your choice to install the graph editor if you prefer.

#### Building the Image

After you have built the Cozy graph editor and generated the `tgz` file, you can build the Docker image by running:

```
docker build -t cozycreator/gen-server:<VERSION> .
```
You can replace `<VERSION>` with the version of the Cozy Creator you want to build, e.g:

```
docker build -t cozycreator/gen-server:0.2.5 .
```
This will build the image and tag it as `cozycreator/gen-server:0.2.5`.

> **Note:** If you're building the image on a Mac, you should add the --platform flag to the docker build command, like this: `docker build --platform=linux/amd64 -t cozycreator/gen-server:<VERSION> .`

<!-- 
Build the Cozy Graph editor, and place it inside of `/web/`, like `cozy-graph-editor-0.0.1.tgz`. This will be used as a dependency when building the front-end. You can install it by running `yarn add file:cozy-graph-editor-0.0.1.tgz` in the `/web` directory, or wherever you placed it the file. In your package.json file you'll see a dependency like this:

`"@cozy-creator/graph-editor": "./cozy-graph-editor-0.0.1.tgz",`

Then in the root of this repo, run: -->

<!-- ```sh
docker build -t cozycreator/gen-server:0.2.5 .
``` -->

### Running the Docker Container

#### Running Locally
After building the image, you can then proceed to run the Docker container.

> **Note:** On windows, we add the `MSYS_NO_PATHCONV=1` flag because Windows command line doesn't know how to interpet paths. Note that in Docker, destination route paths must be absolute, so `~/.cozy-creator` won't work; you must use `/root/.cozy-creator` or whever your user's home directory is. This command below assumes the user is inside of the container is running as root:

WSL2 version:
In Windows, run `wsl` to enter Windows Subsystem for Linux, and then run:

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

#### Running on Runpod

You can also run the Docker container on [Runpod](https://runpod.io/). To do this, you need to create a Runpod account and then follow the instructions [here](https://docs.runpod.io/pods/overview) to learn how to run a pod.

When creating a pod, you will be asked to specify the pod's docker image name among other things. You can use the image name `cozycreator/gen-server:<VERSION>`, where `<VERSION>` is the version of the Cozy Creator you want to use.

Once you have created the pod, runpod will automatically pull the image from Docker Hub and start the pod. You can then access the pod by visiting the URL provided by Runpod.

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

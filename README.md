### Installation

To install the gen-server, navigate to the `/packages/gen_server` folder:

- Run `pip install -e .`; this adds the `cozy` command to your path.

This will generate a `.egg-info` folder. The code will be installed in editable format, which means any changes to the code will be immediately reflected the next time you run it. 
> **Note:** New modules or renamed modules will not be reflected until you run `pip install -e .` again.

Repeat this for all packages you want to install. The other packages extend functionality of the gen-server by adding plugins to the current environment; on startup the gen-server will automatically load all compatible plugins in its environment.

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

Environment variables must have the `COZY` prefix added in order to prevent collisions with other environment variables. For example `export COZY_PORT=9000` will work.

See our .env.example file for all available environment variables. By default, when you run `cozy` it will attempt to load the `.env` file in your current working directory. If the .env file can be found elsewhere, you can specify its location using the `--env-file` flag; 

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

Build the Cozy Graph editor, and place it inside of `/web/`, like `cozy-graph-editor-0.0.1.tgz`. This will be used as a dependency when building the front-end. If you place the file somewhere else, be sure that the package.json dependency points to the right file-location, such as:

`"@cozy-creator/graph-editor": "./cozy-graph-editor-0.0.1.tgz",`

Then in the root of this repo, run:

```sh
docker build -t cozy-creator/gen-server:0.1.1 .
```

### Docker Run

```sh
docker run --env-file=.env -e COZY_HOST=0.0.0.0 -p 8881:8881 -v ~/.cozy_creator:/workspace -e COZY_WORKSPACE_DIR=/workspace --gpus=all cozy-creator/gen-server:0.1.1
```

You can set environment variables manually by using `-e`; just remember to prepend them with `COZY_` first. Some other flag-usage examples:

```sh
docker run -e COZY_HOST=0.0.0.0 -p 9000:9000 -e COZY_PORT=9000  -v "C:/git/comfyui/models":/models -e COZY_MODELS_DIRS='["/models"]' cozy-creator/gen-server:0.1.0
```

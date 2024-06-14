### Dev Usage

To run the main application package, navigate to the `/packages/gen_server` folder:

- Run `pip install -e .`; this adds the `comfy-creator` command to your path.
- Run `comfy-creator`

This will generate a `.egg-info` folder. The code will be installed in editable format, which means any changes to the code will be immediately reflected the next time you run it.

Repeat this for all packages you want to install. The other packages extend functionality of the gen-server by specifying an entry-point group-name; they will be dynamically imported at runtime by the main gen-server application.


### Building for Distrubtion

If you don't already have `build`, you can use pip to install it. Then navigate into the package directory you want to build, and run:

- `python -m build` to build the repository.

This will produce a `/dist` folder, containing a `.whl` (wheel) file and a `.tar.gz` file. The wheel file packages the library in a format that can be easily installed using pip, while the `.tar.gz` file contains the source code. To install the wheel file, use the command `pip install file_name.whl`.

None of our packages currently use any C-APIs, and hence do not need to be recompiled for different environments.


### Running in production

The gen-server currently does not currently check any form of authentication on requests. Use another server to authenticate requests prior to forwarding them to the gen-server.


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



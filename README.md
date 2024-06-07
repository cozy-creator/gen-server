### Dev Commands

To install the main package, navigate to the `/packages/gen_server` folder:
- Run `pip install -e .`; this adds the `comfy-creator` command to your path.
- Run `comfy-creator`

This will generate a `.egg-info` folder.

Repeat this for all packages you want to install. The other packages extend functionality of the gen-server by specifying an entry-point group-name; they will be dynamically imported at runtime by the main gen-server application.

If you have `build` installed, you can also run:
- `python -m build` to build the repo.
This will produce a /dist folder.


### Notes

- 'torch' is required, but is not currently listed as a dependency in our pyproject.toml. Please install the version of torch your environment needs in order to run.


### Running in production

The gen-server currently does not currently check any form of authentication on requests. Use another server to authenticate requests prior to forwarding them to the gen-server.


### Old dependencies:

I'm keeping these here for notes

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

### more notes for later:

[project.entry-points."comfy_creator.api"]
endpoint1 = "core_extension_1.api.endpoint1:Endpoint1"
endpoint2 = "core_extension_1.api.endpoint2:Endpoint2"

[project.entry-points."comfy_creator.architectures"]
architecture1 = "core_extension_1.architectures.sd1_unet.SD1UNet"
architecture2 = "core_extension_1.arch2:Architecture2"
architecture3 = "core_extension_1.text_encoder:SD15TextEncoderArch"
architecture4 = "core_extension_1.unet:SD15UNetArch"
architecture5 = "core_extension_1.VAE:SD15VAEArch"

[project.entry-points."comfy_creator.custom_nodes"]
node1 = "core_extension_1.custom_nodes.node1:get_nodes"
node2 = "core_extension_1.custom_nodes.node1:Node1"

[project.entry-points."comfy_creator.widgets"]
widget1 = "core_extension_1.widgets:Widget1"
widget2 = "core_extension_1.widgets:Widget2"

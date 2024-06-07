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


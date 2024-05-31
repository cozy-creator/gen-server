### Instructions for using in development

At the root of this folder:
- Run `pip install -e .` to install the `comfy-creator` command.
- Run `comfy-creator`

Also:
- `python -m build` builds the repo.



### Running in production

The gen-server currently does not currently check any form of authentication on requests. Use another server to authenticate requests prior to forwarding them to the gen-server.

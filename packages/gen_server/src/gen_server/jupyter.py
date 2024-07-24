import logging
import os
import sys
import traceback
from typing import Optional
import subprocess


logger = logging.getLogger(__name__)


def start_jupyter_lab(
    ip: str = "*",
    port: int = 8888,
    allow_origin="*",
    token: Optional[str] = None,
):
    cmd_args = [
        "jupyter",
        "lab",
        f"--ip={ip}",
        f"--port={port}",
        "--no-browser",
        f"--ServerApp.allow_origin={allow_origin}",
        "--FileContentsManager.delete_to_trash=False",
        # f"--ServerApp.preferred_dir={config.workspace_path}",
        '--ServerApp.terminado_settings={"shell_command":["/bin/bash"]}',
    ]

    token = os.getenv("JUPYTER_PASSWORD") if token is None else token
    if token is None:
        logger.warning("No Jupyter Lab token provided")
    else:
        cmd_args.append(f"--ServerApp.token={token}")

    try:
        subprocess.run(cmd_args, check=True)
    except FileNotFoundError:
        print("Jupyter Lab not found. Please install it with 'pip install jupyterlab'")
        sys.exit(1)
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred while starting Jupyter Lab: {e}")
        sys.exit(1)

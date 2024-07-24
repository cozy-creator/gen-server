#!/bin/bash

set -e  # Exit the script if any statement returns a non-true return value

# Start JupyterLab
start_jupyter() {
    echo "Starting JupyterLab..."
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.allow_origin=* --ServerApp.token=$JUPYTER_PASSWORD \
        --FileContentsManager.delete_to_trash=False --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' &
    echo "JupyterLab started"
}

cozy run & # Start the Cozy server
start_jupyter

sleep infinity  # This will keep the container running

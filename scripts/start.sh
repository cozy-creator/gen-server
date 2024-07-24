#!/bin/bash

set -e  # Exit the script if any statement returns a non-true return value

# Set up Jupyter runtime directory with correct permissions
setup_jupyter_runtime() {
    JUPYTER_RUNTIME_DIR="/root/.local/share/jupyter/runtime"
    mkdir -p "$JUPYTER_RUNTIME_DIR"
    chmod 700 "$JUPYTER_RUNTIME_DIR"
}

# Start JupyterLab
start_jupyter() {
    echo "Starting JupyterLab..."
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token=$JUPYTER_PASSWORD &
    echo "JupyterLab started"
}

setup_jupyter_runtime
cozy run & # Start the Cozy Creator Gen-Server
start_jupyter

sleep infinity  # This will keep the container running

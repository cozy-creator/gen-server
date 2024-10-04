#!/bin/bash

set -e  # Exit the script if any statement returns a non-true return value

# Set up Jupyter runtime directory with correct permissions
setup_jupyter_runtime() {
    JUPYTER_RUNTIME_DIR="/app/.local/share/jupyter/runtime"  # Customize this path
    mkdir -p "$JUPYTER_RUNTIME_DIR"
    chmod 700 "$JUPYTER_RUNTIME_DIR"
    export JUPYTER_RUNTIME_DIR="$JUPYTER_RUNTIME_DIR"  # Export so JupyterLab knows where to find it
}

# Start JupyterLab
start_jupyter() {
    echo "Starting JupyterLab..."
    jupyter lab --ip=0.0.0.0 \
                --port=8888 \
                --no-browser \
                --allow-root \
                --ServerApp.allow_origin=* \
                --IdentityProvider.token=$JUPYTER_PASSWORD \
                --FileContentsManager.delete_to_trash=False \
                --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
                --ServerApp.root_dir='/app' \
                --JupyterApp.runtime_dir="$JUPYTER_RUNTIME_DIR" &
    echo "JupyterLab started"
}

download_test_db() {
    wget https://github.com/user-attachments/files/17260326/test.db.zip
    unzip -o test.db.zip
}

setup_jupyter_runtime
download_test_db

cozy-server run --config-file /workspace/.cozy-creator/config.yaml --db-dsn test.db & # Start the Cozy server
start_jupyter

sleep infinity  # This will keep the container running

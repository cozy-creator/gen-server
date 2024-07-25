#!/bin/bash

set -e  # Exit the script if any statement returns a non-true return value



DOWNLOAD_URLS=(
    "https://civitai.com/api/download/models/290640?type=Model&format=SafeTensor&size=pruned&fp=fp16"
    "https://civitai.com/api/download/models/164360?type=Model&format=SafeTensor&size=pruned&fp=fp16"
    "https://civitai.com/api/download/models/637156?type=Model&format=SafeTensor&size=pruned&fp=fp16"
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true"
    "https://huggingface.co/schirrmacher/ormbg/resolve/main/models/ormbg.pth?download=true"
)

MODEL_FILENAMES=(
    "pony_diffusion_v6.safetensors"
    "break_domain_xl_v05g.safetensors"
    "real_cartoon_3d_v17.safetensors"
    "sd_xl_base_1.0.safetensors"
    "ormbg.pth"
)

MODEL_DIR="/root/.cozy-creator/models"


# Set up Jupyter runtime directory with correct permissions
setup_jupyter_runtime() {
    JUPYTER_RUNTIME_DIR="${JUPYTER_RUNTIME_DIR:-/root/.local/share/jupyter/runtime}"
    mkdir -p "$JUPYTER_RUNTIME_DIR"
    chmod 700 "$JUPYTER_RUNTIME_DIR"
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
                --ServerApp.runtime_dir='/tmp/jupyter-runtime' \
                &
    echo "JupyterLab started"
}


function download_models() {
    mkdir -p "${MODEL_DIR}"
    for i in "${!DOWNLOAD_URLS[@]}"; do
        URL="${DOWNLOAD_URLS[$i]}"
        FILENAME="${MODEL_FILENAMES[$i]}"
        MODEL_PATH="${MODEL_DIR}/${FILENAME}"
        if [ ! -f "${MODEL_PATH}" ]; then
            echo "Downloading ${FILENAME}..."
            wget -O "${MODEL_PATH}" "${URL}"
        else
            echo "${FILENAME} already exists."
        fi
    done
}


download_models &
setup_jupyter_runtime

cozy run & # Start the Cozy server
start_jupyter

sleep infinity  # This will keep the container running

#!/bin/bash

# Setup script for core-extension-1

# Define the path for ai-toolkit
AI_TOOLKIT_PATH="src/core_extension_1/custom_nodes/ai-toolkit"

# Create the directory if it doesn't exist
mkdir -p "src/core_extension_1/custom_nodes"

# Clone ai-toolkit if it doesn't exist
if [ ! -d "$AI_TOOLKIT_PATH" ]; then
    git clone https://github.com/ostris/ai-toolkit.git "$AI_TOOLKIT_PATH"
    cd "$AI_TOOLKIT_PATH"
    git submodule update --init --recursive
    pip install -r requirements.txt
    cd -
else
    echo "ai-toolkit already exists, updating..."
    cd "$AI_TOOLKIT_PATH"
    git pull
    git submodule update --init --recursive
    pip install -r requirements.txt
    cd -
fi

# Install the custom nodes
# pip install -e .

echo "Setup complete. core-extension-1 and ai-toolkit are now installed."
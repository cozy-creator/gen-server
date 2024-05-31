from setuptools import setup, find_packages
from pathlib import Path



requirements_path = Path(__file__).parent / "requirements.txt"
with requirements_path.open("r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="image_nodes",
    version="0.1.0",
    description="Image manipulation utilities for ComfyUI",
    packages=find_packages(),  # Automatically find packages within the directory
    entry_points={
        "comfyui_custom_nodes": [
            "image_nodes = image_nodes.nodes:get_nodes",
            "image_nnodes = image_nodes.nnodes:get_nodes",
        ]
    },
    install_requires=requirements,
)
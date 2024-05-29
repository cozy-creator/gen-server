# this is a meta package for distribution

from setuptools import setup

setup(
    name="ComfyCreator",
    version="0.0.1",
    description="Install gen_server and its extensions",
    install_requires=[
        'gen_server==1.0.0',
        'ComfyCreator-extension1==0.1.0',
        'ComfyCreator-extension2==0.1.0',
        'ComfyCreator-extension3==0.1.0',
        # Add other extensions as needed
    ]
)
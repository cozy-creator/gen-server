from setuptools import setup, find_packages

# Note that if there are any dev / test packages specified in the requirements.txt
# file, you should exclude them from being added here.
with open('requirements.txt') as file:
    requirements = file.read().splitlines()

setup(
    name="ComfyCreator",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'comfy-creator = gen_server.main:main'
        ],
        'comfy_creator.extensions': [
            'extension1 = core:SomeClass',
            'extension2 = core2.main:main',
            'extension3 = image_nodes:AnotherClass',
        ]
    },
    install_requires=requirements
)

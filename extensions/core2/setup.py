from setuptools import setup, find_packages


setup(
    name="core2",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'comfy_creator.api': [
            'endpoint1 = core2.api.endpoint1:Endpoint1',
            'endpoint2 = core2.api.endpoint2:Endpoint2',
        ],
        'comfy_creator.architectures': [
            'architecture1 = core2.arch1:Architecture1',
            'architecture2 = core2.arch2:Architecture2',
        ],

        'comfy_creator.widgets': [
            'widget1 = core2.widgets:Widget1',
            'widget2 = core2.widgets:Widget2',
        ],
        'comfy_creator.custom_nodes': [
            'node1 = core2.custom_nodes.node1:Node1',
            'node2 = core2.custom_nodes.node2:Node2',
        ]
    }
)

from typing import Type

from .base import BaseNode


class NodeRegistry:
    def __init__(self):
        self._nodes = {}

    def register(self, node: Type[BaseNode]):
        if node.type in self._nodes:
            raise ValueError(f"Node {node.type} already registered")

        self._nodes[node.type] = node

    def get(self, node_type: str) -> Type[BaseNode]:
        node = self._nodes.get(node_type)
        if not node:
            raise ValueError(f"Node {node_type} not found")
        return node

    def __iter__(self):
        return iter(self._nodes.values())


node_registry = NodeRegistry()

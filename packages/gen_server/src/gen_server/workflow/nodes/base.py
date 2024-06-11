from enum import Enum


class NodeType(Enum):
    CheckpointLoader = "CheckpointLoader"


class BaseNode:
    inputs: dict = {}
    outputs: dict = {}

    type: NodeType

    def run(self, state):
        raise NotImplementedError

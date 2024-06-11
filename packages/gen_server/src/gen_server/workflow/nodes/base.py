from enum import Enum


class NodeType(Enum):
    CheckpointLoader = "CheckpointLoader"


class BaseNode:
    id: str
    inputs: dict = {}
    outputs: dict = {}

    type: NodeType

    def __init__(self, node_id: str):
        self.id = node_id

    def __call__(self, state):
        raise NotImplementedError

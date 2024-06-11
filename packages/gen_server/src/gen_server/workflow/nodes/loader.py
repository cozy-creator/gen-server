from packages.gen_server.src.gen_server.workflow.nodes.base import BaseNode, NodeType
from diffusers import DiffusionPipeline

from .registry import node_registry
from ..state import RunnerState


class CheckpointLoader(BaseNode):
    inputs = {"checkpoint": {"type": "STRING", "required": True}}
    outputs = {"pipe": {'type': 'PIPE'}}

    type = NodeType.CheckpointLoader

    def __call__(self, state: RunnerState):
        pipe = DiffusionPipeline.from_pretrained("")
        if pipe:
            pipe.to("cuda")

        return {'pipe': pipe}


node_registry.register(CheckpointLoader)

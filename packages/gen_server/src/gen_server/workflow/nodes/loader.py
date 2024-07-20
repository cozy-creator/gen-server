from gen_server.workflow.nodes.base import BaseNode, NodeType
from diffusers import DiffusionPipeline

from .registry import node_registry
from ..state import RunnerState
from ...utils.device import get_torch_device


class CheckpointLoader(BaseNode):
    inputs = {"checkpoint": {"type": "STRING", "required": True}}
    outputs = {"pipe": {"type": "PIPE"}}

    type = NodeType.CheckpointLoader

    def __call__(self, state: RunnerState):
        pipe = DiffusionPipeline.from_pretrained("")
        if pipe:
            pipe.to(get_torch_device())

        return {"pipe": pipe}


node_registry.register(CheckpointLoader)

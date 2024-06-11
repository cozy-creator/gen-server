from packages.core_library.workflow.nodes.base import BaseNode, NodeType
from diffusers import DiffusionPipeline


class CheckpointLoader(BaseNode):
    inputs = {"checkpoint": {"type": "STRING", "required": True}}
    outputs = {"pipe": {'type': 'PIPE'}}

    type = NodeType.CheckpointLoader

    def run(self, data: dict):
        pipe = DiffusionPipeline.from_pretrained("")
        if pipe:
            pipe.to("cuda")

        return {'pipe': pipe}

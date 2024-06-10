from packages.core_library.workflow.nodes.base import BaseNode
from diffusers import DiffusionPipeline


class CheckpointLoader(BaseNode):
    inputs = [{
        'name': 'checkpoint',
        'type': 'STRING',
        'required': True
    }]

    outputs = [{'name': 'pipe', 'type': 'PIPE'}]

    type = "CheckpointLoader"

    def __call__(self, data: dict):
        pipe = DiffusionPipeline.from_pretrained("")
        if pipe:
            pipe.to("cuda")

        return {'pipe': pipe}

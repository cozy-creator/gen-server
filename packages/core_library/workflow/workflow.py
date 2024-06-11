from typing import TypedDict, Union, Literal

from packages.core_library.workflow.nodes.base import NodeType


class Workflow:
    def __init__(self, data: dict[str, any]):
        self.data = data

from typing import TypedDict, Union

from .nodes.base import NodeType
from .nodes.registry import node_registry
from .state import RunnerState

InputData = Union[str, int, float, bool, list[str, str]]
OutputData = list[str, str]


class WorkflowData(TypedDict):
    type: NodeType
    inputs: dict[str, InputData]
    outputs: dict[str, OutputData]


class WorkflowExecutor:
    def __init__(self, workflow: dict[str, WorkflowData], inputs):
        self.workflow = workflow
        self.state = RunnerState(inputs)

    def sort_workflow_nodes(self):
        visited = set[str]()
        sorted_nodes = []

        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            for neighbour in self.workflow:
                visit(neighbour)
            sorted_nodes.insert(0, node_id)

        for node_id in self.workflow:
            if node_id not in visited:
                visit(node_id)

        return sorted_nodes

    def _get_node(self, data: WorkflowData):
        if not data.get("type"):
            raise ValueError('node type is required')

        Node = node_registry.get(data.get("type"))
        return Node(data)

    async def execute_node(self, node):
        try:
            await node.execute(self.state)
        except Exception as error:
            # raise WorkflowNodeRunFailedError(node)
            raise error
        finally:
            self.state.last_executed_node = node.id

    async def run(self):
        if not self.workflow.data:
            raise ValueError('workflow missing in workflow')

        node_ids = self.sort_workflow_nodes()

        for node_id in node_ids:
            try:
                node = self._get_node(self.workflow[node_id])
                if not node:
                    break
                await self.execute_node(node)
            except Exception as error:
                print(error)
                break

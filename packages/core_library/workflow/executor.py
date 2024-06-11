import asyncio
from typing import TypedDict, Union

from packages.core_library.workflow.nodes.base import BaseNode, NodeType
from packages.core_library.workflow.workflow import Workflow

InputData = Union[str, int, float, bool, list[str, str]]
OutputData = list[str, str]


class WorkflowData(TypedDict):
    type: NodeType
    inputs: dict[str, InputData]
    outputs: dict[str, OutputData]


class ExecutionState:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = {}

        self.errors = {}
        self.outputs = {}
        self.inputs = inputs
        self.last_executed_node = None

        self.end_time = None
        self.start_time = asyncio.get_event_loop().time()

    def set_output(self, node_id, output):
        self.outputs[node_id] = output

    def set_error(self, node_id, error):
        self.errors[node_id] = error

    def complete(self):
        self.end_time = asyncio.get_event_loop().time()

    def get_input(self, node_id: str, name: str):
        inputs = self.inputs.get(node_id, {})
        if not isinstance(inputs, dict):
            raise ValueError(f"Invalid inputs type for {node_id}")

        value = inputs.get(name)
        if value is None:
            raise ValueError(f"Cannot find input {name} for node {node_id}")

        return value

    def get_output(self, node_id: str, name: str):
        outputs = self.outputs.get(node_id, {})
        if not isinstance(outputs, dict):
            raise ValueError(f"Invalid outputs type for {node_id}")

        value = outputs.get(name)
        if value is None:
            raise ValueError(f"Cannot find output {name} for node {node_id}")

        return value


class WorkflowExecutor:
    def __init__(self, workflow: dict[str, WorkflowData], inputs):
        self.workflow = workflow
        self.state = ExecutionState(inputs)

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

    async def execute_node(self, node):
        try:
            inputs = {**self.state.inputs[node.id]}

            await node.execute(inputs)
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
            node = self.workflow.data[node_id]
            try:
                await self.execute_node(node)
            except Exception as error:
                print(error)
                break

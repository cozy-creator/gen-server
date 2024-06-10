import asyncio

from packages.core_library.workflow.workflow import Workflow


class ExecutionState:
    def __init__(self, inputs):
        self.errors = {}
        self.results = {}
        self.inputs = inputs
        self.last_executed_node = None

        self.end_time = None
        self.start_time = asyncio.get_event_loop().time()

    def set_result(self, node_id, result):
        self.results[node_id] = result

    def set_error(self, node_id, error):
        self.errors[node_id] = error

    def complete(self):
        self.end_time = asyncio.get_event_loop().time()


class WorkflowExecutor:
    def __init__(self, workflow: Workflow, inputs):
        self.workflow = workflow
        self.state = ExecutionState(inputs)

    async def execute_node(self, node):
        try:
            input = {
                **self.state.results[self.state.last_executed_node],
                **self.state.inputs[node.id]
            }

            await node.execute(input)
        except Exception as error:
            # raise WorkflowNodeRunFailedError(node.id, node.type, node.title, str(error))
            raise error
        finally:
            self.state.last_executed_node = node.id

    async def execute(self):
        graph = self.workflow.graph
        if not graph:
            raise ValueError('graph missing in workflow')

        if 'nodes' not in graph or 'edges' not in graph:
            raise ValueError('nodes or edges missing in workflow graph')

        sorted_node_ids = self.workflow.sort_nodes()

        for node_id in sorted_node_ids:
            node = graph['nodes'][node_id]
            try:
                await self.execute_node(node)
            except Exception as error:
                print(error)
                break

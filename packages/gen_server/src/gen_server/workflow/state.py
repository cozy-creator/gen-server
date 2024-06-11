class RunnerState:
    def __init__(self, inputs):
        if inputs is None:
            inputs = {}

        self._inputs = inputs
        self._return_values = {}
        self.previous_node = None

        self.start_time = None
        self.finish_time = None

    def set_return_value(self, node_id: str, value):
        if node_id in self._return_values:
            raise ValueError(f"Return value for node {node_id} already exists")
        if value is None:
            raise ValueError(f"Return value for node {node_id} cannot be None")
        if not isinstance(value, dict):
            raise ValueError(f"Return value for node {node_id} must be a dict")

        self._return_values[node_id] = value

    def set_input(self, node_id: str, name: str, value):
        if not isinstance(value, dict):
            raise ValueError(f"Input value for node {node_id} must be a dict")
        if value is None:
            raise ValueError(f"Input value for node {node_id} cannot be None")
        if node_id not in self._inputs:
            self._inputs[node_id] = {}

        self._inputs[node_id][name] = value

    def get_node_inputs(self, node_id: str):
        if node_id not in self._inputs:
            raise ValueError(f"Cannot find inputs for node {node_id}")
        if not isinstance(self._inputs[node_id], dict):
            raise ValueError(f"Inputs for node {node_id} must be a dict")

        inputs = self._inputs[node_id]
        if not inputs:
            raise ValueError(f"Inputs for node {node_id} cannot be empty")

        return inputs

    def get_input(self, node_id: str, name: str):
        inputs = self.get_node_inputs(node_id)
        value = inputs.get(name)
        if value is None:
            raise ValueError(f"Cannot find input {name} for node {node_id}")

        return value

    def get_node_return_values(self, node_id: str):
        if node_id not in self._return_values:
            raise ValueError(f"Cannot find return value for node {node_id}")
        if not isinstance(self._return_values[node_id], dict):
            raise ValueError(f"Return value for node {node_id} must be a dict")

        return_values = self._return_values[node_id]
        if not return_values:
            raise ValueError(f"Return value for node {node_id} cannot be empty")

        return return_values

    def get_return_value(self, node_id: str, name: str):
        return_values = self.get_node_return_values(node_id)
        value = return_values.get(name)
        if value is None:
            raise ValueError(f"Cannot find return value {name} for node {node_id}")

        return value

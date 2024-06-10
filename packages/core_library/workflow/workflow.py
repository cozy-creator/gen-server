class Workflow:
    def __init__(self, graph):
        self.graph = graph

    def sort_nodes(self):
        visited = set()
        sorted_nodes = []

        def visit(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            for neighbour in self.graph.nodes[node_id]:
                visit(neighbour)
            sorted_nodes.insert(0, node_id)

        for node_id in self.graph.nodes:
            if node_id not in visited:
                visit(node_id)

        return sorted_nodes

from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state_graph.state import State


class StateGraphBuilder:
    def __init__(self):
        self.builder = StateGraph(State)

    def add_node(self, name, func):
        self.builder.add_node(name, func)

    def add_edge(self, from_node, to_node):
        self.builder.add_edge(from_node, to_node)

    def add_conditional_edges(self, from_node, func, conditions=None):
        self.builder.add_conditional_edges(from_node, func, conditions)

    def set_entry_point(self, node_name):
        self.builder.set_entry_point(node_name)

    def compile_graph(self):
        memory = SqliteSaver.from_conn_string(":memory:")
        return self.builder.compile(checkpointer=memory)

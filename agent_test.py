from typing import TypedDict
from langgraph.graph import StateGraph

class State(TypedDict):
    counter: int
    alphabet: list[str]

graph_builder = StateGraph(State)

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
import operator

class State(TypedDict):
    counter: int
    alphabet: list[str]

def node_a(state: State):
    state['counter'] += 1
    state['alphabet']= ["Hello"]
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", node_a)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

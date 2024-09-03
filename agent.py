from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing import Annotated, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, MessagesState

@tool
def get_weather(location: str):
    """Call to get the restaurant"""
    if location.lower() in ["서울", "인천"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."

@tool
def get_coolest_cities():
    """Get a list of coolest cities"""
    return "서울, 고성"

tools = [get_weather, get_coolest_cities]
tool_node = ToolNode(tools)

model_with_tools = ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).bind_tools(tools)

def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tools", 'agent')

graph = workflow.compile()

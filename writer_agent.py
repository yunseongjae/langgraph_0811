from typing import Annotated, List, Dict
from langchain_openai import ChatOpenAI
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain.prompts import ChatPromptTemplate

class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    topic: str
    outline: List[Dict[str, str]]
    current_section: int
    content: List[str]
    research_data: List[Dict[str, str]]

@tool
def arxiv_search(query: str) -> str:
    """arXiv에서 논문을 검색합니다."""
    arxiv = ArxivAPIWrapper()
    return arxiv.run(query)

tavily_tool = TavilySearchResults(max_results=3)

tools = [arxiv_search, tavily_tool]

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

def create_outline(state: ResearchState) -> ResearchState:
    print(f"\n--- 개요 작성 시작 (주제: {state['topic']}) ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "주어진 주제에 대한 연구 보고서의 개요를 작성해주세요. 각 섹션은 번호와 제목을 포함해야 합니다."),
        ("human", "주제: {topic}\n개요를 작성해주세요:"),
    ])
    chain = prompt | llm
    result = chain.invoke({"topic": state["topic"]})
    
    outline = [{"title": line.strip()} for line in result.content.split('\n') if line.strip()]
    
    print("생성된 개요:")
    for item in outline:
        print(f"- {item['title']}")
    
    return {
        **state,
        "outline": outline,
        "messages": state["messages"] + [AIMessage(content=result.content)]
    }

def collect_information(state: ResearchState) -> ResearchState:
    current_section = state["outline"][state["current_section"]]
    print(f"\n--- 정보 수집 시작 (섹션: {current_section['title']}) ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "주어진 주제와 섹션에 대한 정보를 수집해주세요. arXiv 검색과 웹 검색 도구를 사용할 수 있습니다."),
        ("human", "주제: {topic}\n섹션: {section}\n관련 정보를 수집해주세요:"),
    ])
    chain = prompt | llm_with_tools
    result = chain.invoke({"topic": state["topic"], "section": current_section["title"]})
    
    print(f"수집된 정보 (요약): {result.content[:100]}...")
    
    return {
        **state,
        "research_data": state["research_data"] + [{"section": current_section["title"], "data": result.content}],
        "messages": state["messages"] + [AIMessage(content=result.content)]
    }

def write_report(state: ResearchState) -> ResearchState:
    current_section = state["outline"][state["current_section"]]
    print(f"\n--- 보고서 작성 시작 (섹션: {current_section['title']}) ---")
    research_data = next((item["data"] for item in state["research_data"] if item["section"] == current_section["title"]), "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "수집된 정보를 바탕으로 연구 보고서의 섹션을 작성해주세요."),
        ("human", "주제: {topic}\n섹션: {section}\n수집된 정보: {research_data}\n이 섹션을 작성해주세요:"),
    ])
    chain = prompt | llm
    result = chain.invoke({"topic": state["topic"], "section": current_section["title"], "research_data": research_data})
    
    print(f"작성된 섹션 (요약): {result.content[:100]}...")
    
    return {
        **state,
        "content": state["content"] + [result.content],
        "current_section": state["current_section"] + 1,
        "messages": state["messages"] + [AIMessage(content=result.content)]
    }

def should_continue(state: ResearchState):
    if state["current_section"] >= len(state["outline"]):
        print("\n--- 모든 섹션 작성 완료 ---")
        return "end"
    print(f"\n--- 다음 섹션으로 진행 (현재 진행도: {state['current_section']}/{len(state['outline'])}) ---")
    return "collect_information"

graph_builder = StateGraph(ResearchState)

graph_builder.add_node("create_outline", create_outline)
graph_builder.add_node("collect_information", collect_information)
graph_builder.add_node("write_report", write_report)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("create_outline")

graph_builder.add_edge("create_outline", "collect_information")
graph_builder.add_edge("collect_information", "write_report")
graph_builder.add_edge("collect_information", "tools")
graph_builder.add_edge("tools", "collect_information")

graph_builder.add_conditional_edges(
    "write_report",
    should_continue,
    {
        "collect_information": "collect_information",
        "end": END
    }
)

app = graph_builder.compile()

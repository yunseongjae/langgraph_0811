import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
import nest_asyncio
nest_asyncio.apply()

from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
wolfram = WolframAlphaAPIWrapper()

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    focus: Literal["web", "academic", "video"]

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
import re

web_tool = TavilySearchResults(max_results=2)

@tool
def academic_tool(query:str):
    """
    academic paper search tool
    """
    arxiv = ArxivAPIWrapper()
    docs = arxiv.run(query)
    return docs

@tool
def math_tool(query:str):
    """
    math tool
    """
    wolfram = WolframAlphaAPIWrapper()
    result = wolfram.run(query)
    pattern = r'Answer:\s*(.*)'
    match = re.search(pattern, result, re.IGNORECASE)
    if match:
        # Extract the answer and remove any leading/trailing whitespace
        answer = match.group(1).strip()
    else:
        answer = result
    return answer

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.tools import YouTubeSearchTool
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
import ast

youtube_search_tool = YouTubeSearchTool()


@tool
def video_tool(query:str) -> str:
    """
    Retriever tool for the transcript of a YouTube video. query should be given in string format.
    """
    #query에 해당하는 Youtube 비디오 URL 가져오기
    urls = youtube_search_tool.run(query)
    urls = ast.literal_eval(urls)
    #URL 순회하면서 Document 객체에 내용 담기
    docs = []
    for url in urls:
        loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=True,
        language=["en", "ko"]
        )
        scripts = loader.load()
        script_content = scripts[0].page_content
        title=scripts[0].metadata['title']
        author=scripts[0].metadata['author']
        doc = Document(page_content=script_content, metadata={"source": url, "title":title, "author":author})
        docs.append(doc)

    #모든 비디오의 내용을 벡터DB에 담기
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    retrieved_docs = retriever.invoke(query)

    video_results = []

    for doc in retrieved_docs:
        title = doc.metadata.get('title', 'No title available')
        author = doc.metadata.get('author', 'No author available')
        script_content = doc.page_content

        video_info = f"""
        Video Information:
        ------------------
        Title: {title}
        Author: {author}
        Transcript:
        {script_content}
        ------------------
        """
        video_results.append(video_info)

    # Join all video results into a single string
    all_video_results = "\n\n".join(video_results)

    return all_video_results

tools = {
    "web": [web_tool],
    "academic": [academic_tool],
    "video": [video_tool],
    "math":[math_tool]
}

tool_nodes = {focus: ToolNode(tools[focus]) for focus in tools}

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State):
    llm_with_tools = llm.bind_tools(tools[state["focus"]])
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}

from langgraph.graph import StateGraph, END

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
for focus, tool_node in tool_nodes.items():
    graph_builder.add_node(f"{focus}_tools", tool_node)

def tools_condition(state):
    if state["messages"][-1].tool_calls:
        return f"{state['focus']}_tools"
    return END  # END 상수 사용

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "web_tools": "web_tools",
        "academic_tools": "academic_tools",
        "video_tools": "video_tools",
        "math_tools": "math_tools",
        END: END
    }
)

for focus in tools:
    graph_builder.add_edge(f"{focus}_tools", "chatbot")

graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

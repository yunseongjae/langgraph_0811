from typing import Annotated, TypedDict, List, Dict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field, create_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from docx import Document
from docx.shared import Inches

# OpenAI client initialization
client = OpenAI()

# Schema definition for DALL-E image generation
class GenImageSchema(BaseModel):
    prompt: str = Field(description="The prompt for image generation")

# DALL-E image generation function definition
@tool(args_schema=GenImageSchema)
def generate_image(prompt: str) -> str:
    """Generate an image using DALL-E based on the given prompt."""
    if not prompt:
        raise ValueError("Image generation prompt cannot be empty")
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        print(f"Error in image generation: {e}")
        return "Image generation failed"

# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    outline: Dict[str, str]
    current_section: int
    section_content: str
    section_image: str
    image_prompt: str
    total_sections: int
    full_report: List[Dict[str, str]]

# Graph builder initialization
graph_builder = StateGraph(State)

# LLM setup
llm = ChatOpenAI(model="gpt-4o-mini")
search = TavilySearchResults(max_results=3)

# Function to create dynamic Outline model
def create_outline_model(section_count: int):
    fields = {f"section{i}": (str, Field(description=f"Title for section {i}")) for i in range(1, section_count + 1)}
    return create_model("DynamicOutline", **fields)

# Outline generation node
def create_outline(state: State):
    DynamicOutline = create_outline_model(state["total_sections"])
    outline_parser = JsonOutputParser(pydantic_object=DynamicOutline)

    outline_prompt = PromptTemplate(
        template="Create an outline for a detailed report with exactly {section_count} main sections.\n{format_instructions}\nThe topic is: {topic}\n",
        input_variables=["section_count", "topic"],
        partial_variables={"format_instructions": outline_parser.get_format_instructions()},
    )
    
    chain = outline_prompt | llm | outline_parser
    
    try:
        outline = chain.invoke({"section_count": state["total_sections"], "topic": state["messages"][-1].content})
        return {"outline": outline, "current_section": 1, "full_report": []}
    except Exception as e:
        print(f"Error in create_outline: {e}")
        return {"error": str(e)}

# Writer node
def write_section(state: State):
    if "error" in state:
        return {"messages": [AIMessage(content=f"An error occurred: {state['error']}")]}
    
    if state["current_section"] > state["total_sections"]:
        return {"messages": [AIMessage(content="Report completed.")]}
    
    current_section_key = f"section{state['current_section']}"
    current_topic = state["outline"][current_section_key]
    search_results = search.invoke(current_topic)
    
    previous_sections = "\n".join([f"Section {i}: {state['outline'][f'section{i}']}" for i in range(1, state['current_section'])])
    
    section_prompt = PromptTemplate(
        template="""Write a detailed section for the topic: {topic}. 
        
        Use the following search results for information: {search_results}
        
        Previous sections:
        {previous_sections}
        
        Write only the content for this section, 
        do not include any image prompts or suggestions.
        Detailed statistics or information is needed, 
        so you should include collected information from search result.""",
        input_variables=["topic", "search_results", "previous_sections"],
    )
    section_content = llm.invoke(section_prompt.format(
        topic=current_topic,
        search_results=search_results,
        previous_sections=previous_sections
    ))

    return {
        "section_content": section_content.content,
        "current_section": state["current_section"]
    }


# Image generator node
def generate_image_prompt(state: State):

    image_prompt_generator = PromptTemplate(
        template="""Based on the following section content, 
        create a prompt for generating an infographic that represents this section.
        Section content: {section_content}
        \n\n
        Image generation prompt(under 500 characters):""",
        input_variables=["section_content"],
    )
    
    try:
        image_prompt = llm.invoke(image_prompt_generator.format(section_content=state["section_content"]))
        image_url = generate_image(image_prompt.content)
    except Exception as e:
        print(f"Error in generate_image_prompt: {e}")
        image_prompt = "Error generating image prompt"
        image_url = "Image generation failed"

    current_section = {
        "title": state['outline'][f"section{state['current_section']}"],
        "content": state['section_content'],
        "image_url": image_url,
        "image_prompt": "DO NOT CONTAIN ANY METRIC OR TEXT ON THE IMAGE. \n"+image_prompt.content if isinstance(image_prompt, AIMessage) else image_prompt
    }

    updated_full_report = state.get("full_report", []) + [current_section]

    print(f"Completed section {state['current_section']} of {state['total_sections']}")

    return {
        "image_prompt": image_prompt.content if isinstance(image_prompt, AIMessage) else image_prompt,
        "section_image": image_url,
        "current_section": state["current_section"] + 1,
        "full_report": updated_full_report
    }
from docx import Document
from docx.shared import Inches
import requests
from io import BytesIO

def finalize_report(state: State):
    doc = Document()
    doc.add_heading(f"Report: {state['messages'][0].content}", 0)

    for section in state['full_report']:
        doc.add_heading(section['title'], level=1)
        doc.add_paragraph(section['content'])
        
        # 이미지 추가
        if section['image_url'] != "Image generation failed":
            try:
                response = requests.get(section['image_url'])
                image = BytesIO(response.content)
                doc.add_picture(image, width=Inches(6))
                doc.add_paragraph(f"Image prompt: {section['image_prompt']}")
            except Exception as e:
                doc.add_paragraph(f"Failed to add image: {str(e)}")

        doc.add_page_break()

    # 보고서 저장
    filename = f"report_{state['messages'][0].content}.docx".replace(" ", "_")
    doc.save(filename)

    return {
        "messages": [AIMessage(content=f"Report finalized and saved as {filename}.")],
        "report_file": filename
    }

# Add nodes
graph_builder.add_node("create_outline", create_outline)
graph_builder.add_node("write_section", write_section)
graph_builder.add_node("generate_image", generate_image_prompt)
graph_builder.add_node("finalize_report", finalize_report)

# Add edges
graph_builder.add_edge(START, "create_outline")
graph_builder.add_edge("create_outline", "write_section")
graph_builder.add_edge("write_section", "generate_image")
graph_builder.add_edge("finalize_report", END)
# Add conditional edges
def should_continue_writing(state: State):
    if state["current_section"] <= state["total_sections"]:
        return "write_section"
    else:
        return "finalize_report"

graph_builder.add_conditional_edges(
    "generate_image",
    should_continue_writing,
    {
        "write_section": "write_section",
        "finalize_report": "finalize_report"
    }
)

# Compile graph
graph = graph_builder.compile()

from typing import TypedDict, Annotated, List, Union
from enum import Enum
from dataclasses import dataclass
import operator
import uuid
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.types import Send

from langchain_google_genai import ChatGoogleGenerativeAI
# 모델 로드 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_output_tokens=8192)

# 지원 언어 정의
class Language(str, Enum):
    KOREAN = "korean"
    ENGLISH = "english"
    JAPANESE = "japanese"
    CHINESE = "chinese"

# 번역 설정을 위한 데이터클래스
@dataclass
class TranslationConfig:
    source_lang: Language
    target_lang: Language
    preserve_format: bool = True

# 번역 상태를 위한 TypedDict
class TranslationState(TypedDict):
    input_text: str
    text_chunks: List[str]
    translations: Annotated[List[str], operator.add]
    final_translation: str
    config: TranslationConfig
    style_analysis: dict
    translation_prompt: str
    is_parallel: bool

# 톤 분석 노드
async def analyze_tone(state: TranslationState):
    analyzer = llm
    
    analysis_prompt = """Analyze the writing style and tone of the following text. Consider:
    1. Formality level (formal, semi-formal, casual)
    2. Technical level (technical, semi-technical, general)
    3. Emotional tone (objective, subjective, persuasive, etc.)
    4. Writing style characteristics (academic, journalistic, conversational, etc.)
    5. Special language features (idioms, metaphors, technical terms)
    6. Sentence structure patterns
    
    Text to analyze:
    {text}
    
    Provide a detailed analysis that can be used to create translation guidelines."""
    
    messages = [
        {"role": "system", "content": "You are a tone analysis expert."},
        {"role": "user", "content": analysis_prompt.format(text=state["input_text"])}
    ]
    
    analysis = await analyzer.ainvoke(messages)
    return {"style_analysis": analysis.content}

# 번역 프롬프트 생성 노드
async def create_translation_prompt(state: TranslationState):
    prompt_creator = llm
    
    prompt_template = """Based on the following style analysis, create a detailed translation prompt 
    that will help maintain the same style in the target language ({target_lang}).

    Style Analysis:
    {style_analysis}

    Create a translation prompt that includes:
    1. Specific instructions for maintaining the identified tone and style
    2. Guidelines for sentence structure and pattern matching
    3. Specific examples of how certain elements should be translated
    4. Any special considerations for the target language"""
    
    messages = [
        {"role": "system", "content": "You are a translation prompt expert."},
        {"role": "user", "content": prompt_template.format(
            target_lang=state["config"].target_lang.value,
            style_analysis=state["style_analysis"]
        )}
    ]
    
    translation_prompt = await prompt_creator.ainvoke(messages)
    return {"translation_prompt": translation_prompt.content}

def determine_processing_strategy(state: TranslationState):
    """텍스트 길이에 따라 처리 전략 결정"""
    text_length = len(state["input_text"])
    return {"is_parallel": text_length > 5000}

def process_text(state: TranslationState):
    """텍스트 처리 전략에 따라 분할 또는 단일 처리"""
    if state["is_parallel"]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100
        )
        chunks = splitter.split_text(state["input_text"])
        return {"text_chunks": chunks}
    else:
        return {"text_chunks": [state["input_text"]]}

def should_parallelize(state: TranslationState):
    """병렬 처리 여부에 따라 다른 노드로 라우팅"""
    if state["is_parallel"]:
        return "parallel_translate"
    return "single_translate"

# 단일 텍스트 번역 노드
async def translate_single(state: TranslationState):
    """전체 텍스트를 한 번에 번역"""
    translator = llm
    
    messages = [
        {"role": "system", "content": state["translation_prompt"]},
        {"role": "user", "content": f"Translate the following text maintaining the specified style: \n\n{state['text_chunks'][0]}"}
    ]
    
    translation = await translator.ainvoke(messages)
    return {
        "translations": [translation.content],
        "final_translation": translation.content
    }

# 병렬 번역 노드
async def translate_chunk(state: TranslationState):
    """각 청크를 번역"""
    translator = llm
    
    messages = [
        {"role": "system", "content": state["translation_prompt"]},
        {"role": "user", "content": f"Translate the following text maintaining the specified style: \n\n{state['text_chunks']}"}
    ]
    
    translation = await translator.ainvoke(messages)
    return {"translations": [translation.content]}

# 리뷰어 노드
async def review_translations(state: TranslationState):
    """번역된 청크들을 검토하고 통합"""
    reviewer = llm
    
    combined = "\n".join(state["translations"])
    review_prompt = f"""Review and improve the following translation. 
    Please print the translated text only.
    Original Style Analysis: {state['style_analysis']}
    
    This text was translated in parallel segments. Ensure:
    1. Consistent style and tone throughout
    2. Smooth connections between segments
    3. Natural flow while maintaining the original style
    4. Proper adaptation of style-specific elements in {state['config'].target_lang.value}
    
    Text to review:
    {combined}
    """
    
    messages = [
        {"role": "system", "content": state["translation_prompt"]},
        {"role": "user", "content": review_prompt}
    ]
    
    final = await reviewer.ainvoke(messages)
    return {"final_translation": final.content}

# 병렬 번역용 매핑 함수
def map_translations(state: TranslationState):
    """각 청크에 대해 번역 작업을 매핑"""
    return [
        Send(
            "parallel_translate",
            {"chunk": chunk}
        ) for chunk in state["text_chunks"]
    ]

# 그래프 구성
workflow = StateGraph(TranslationState)
# 그래프 구성

# 노드 추가
workflow.add_node("strategy_determiner", determine_processing_strategy)
workflow.add_node("text_processor", process_text)
workflow.add_node("tone_analyzer", analyze_tone)
workflow.add_node("prompt_creator", create_translation_prompt)
workflow.add_node("single_translate", translate_single)
workflow.add_node("parallel_translate", translate_chunk)
workflow.add_node("reviewer", review_translations)

# 엣지 연결
workflow.add_edge(START, "strategy_determiner")
workflow.add_edge("strategy_determiner", "tone_analyzer")
workflow.add_edge("tone_analyzer", "prompt_creator")
workflow.add_edge("prompt_creator", "text_processor")

# text_processor에서 조건부 분기를 위한 추가 노드 설정
workflow.add_conditional_edges(
    "text_processor",
    should_parallelize,
    {
        "single_translate": "single_translate",
        "parallel_translate": "parallel_translate"
    }
)

# 병렬 처리를 위한 conditional edges 추가
workflow.add_conditional_edges(
    "parallel_translate",
    lambda x: "reviewer" if x.get("translations") else "parallel_translate",
    {
        "parallel_translate": "parallel_translate",
        "reviewer": "reviewer"
    }
)

# 나머지 엣지 연결
workflow.add_edge("single_translate", END)
workflow.add_edge("reviewer", END)

# 컴파일
app = workflow.compile()

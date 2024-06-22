from typing import Any

# Example implementation using LiteLLM
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src.external.tool_calling_llm import ToolCallingLLM


class LiteLLMFunctions(ToolCallingLLM, ChatNVIDIA):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "litellm_functions"


## Core LC Chat Interface

llm = LiteLLMFunctions(model="mistralai/mixtral-8x7b-instruct-v0.1")

if 0:
    result = llm.invoke("Write a ballad about LangChain.")
    print(result.content)

from langchain_core.pydantic_v1 import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])

ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)
print(ai_msg)

"""Copied from LangChain PR #23193: experimental: Mixin to allow tool calling features for non tool calling chat models
https://github.com/langchain-ai/langchain/pull/23193
"""

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA

from src.external.tool_calling_llm import ToolCallingLLM


class LiteLLMFunctions(ToolCallingLLM, ChatNVIDIA):

    @property
    def _llm_type(self) -> str:
        return "litellm_functions"

from .base import BaseLLM
from config import settings

_llm_instance: BaseLLM | None = None


def get_llm() -> BaseLLM:
    global _llm_instance
    if _llm_instance is None:
        provider = settings.LLM_PROVIDER.lower()
        if provider == "openai":
            from .openai_provider import OpenAIProvider
            _llm_instance = OpenAIProvider()
        elif provider == "claude":
            from .claude_provider import ClaudeProvider
            _llm_instance = ClaudeProvider()
        elif provider == "gemini":
            from .gemini_provider import GeminiProvider
            _llm_instance = GeminiProvider()
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: {provider}. Choose openai, claude, or gemini."
            )
    return _llm_instance

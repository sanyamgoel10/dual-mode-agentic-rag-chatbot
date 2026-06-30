from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseLLM(ABC):
    @abstractmethod
    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """
        Returns one of:
          {"type": "tool_calls", "calls": [{"id": str, "name": str, "input": dict}]}
          {"type": "text", "content": str}
        """

    @abstractmethod
    async def stream_response(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """Yields string tokens for the final answer. No tools passed."""

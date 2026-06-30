import json
from typing import AsyncGenerator
import anthropic
from langsmith import traceable
from .base import BaseLLM
from config import settings


def _to_claude_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in tools
    ]


def _to_claude_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Returns (system_prompt, messages_list). Claude takes system prompt separately."""
    system = ""
    result = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "tool":
            # Tool results go as user messages in Claude
            if result and result[-1]["role"] == "user" and isinstance(result[-1]["content"], list):
                result[-1]["content"].append({
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                })
            else:
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m["tool_call_id"],
                            "content": m["content"],
                        }
                    ],
                })
        elif m["role"] == "assistant" and "tool_calls" in m:
            result.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"],
                    }
                    for tc in m["tool_calls"]
                ],
            })
        else:
            result.append({"role": m["role"], "content": m["content"]})
    return system, result


class ClaudeProvider(BaseLLM):
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.LLM_MODEL

    @traceable(name="claude_chat_with_tools")
    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        system, native_messages = _to_claude_messages(messages)
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=native_messages,
            tools=_to_claude_tools(tools),
        )
        if response.stop_reason == "tool_use":
            calls = [
                {"id": block.id, "name": block.name, "input": block.input}
                for block in response.content
                if block.type == "tool_use"
            ]
            return {"type": "tool_calls", "calls": calls}
        text = next(
            (block.text for block in response.content if block.type == "text"), ""
        )
        return {"type": "text", "content": text}

    @traceable(name="claude_stream_response")
    async def stream_response(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        system, native_messages = _to_claude_messages(messages)
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=native_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

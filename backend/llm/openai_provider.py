import json
from typing import AsyncGenerator
from openai import AsyncOpenAI
from langsmith import traceable
from .base import BaseLLM
from config import settings


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    result = []
    for m in messages:
        if m["role"] == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": m["tool_call_id"],
                "content": m["content"],
            })
        elif m["role"] == "assistant" and "tool_calls" in m:
            result.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["input"]),
                        },
                    }
                    for tc in m["tool_calls"]
                ],
            })
        else:
            result.append({"role": m["role"], "content": m["content"]})
    return result


class OpenAIProvider(BaseLLM):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.LLM_MODEL

    @traceable(name="openai_chat_with_tools")
    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(messages),
            tools=_to_openai_tools(tools),
            tool_choice="auto",
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            return {
                "type": "tool_calls",
                "calls": [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments),
                    }
                    for tc in msg.tool_calls
                ],
            }
        return {"type": "text", "content": msg.content or ""}

    @traceable(name="openai_stream_response")
    async def stream_response(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(messages),
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

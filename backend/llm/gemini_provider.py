from typing import AsyncGenerator
from google import genai
from google.genai import types
from langsmith import traceable
from .base import BaseLLM
from config import settings

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return _client


def _to_genai_tools(tools: list[dict]) -> list[types.Tool]:
    declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
        )
        for t in tools
    ]
    return [types.Tool(function_declarations=declarations)]


def _to_genai_contents(messages: list[dict]) -> tuple[str, list]:
    """Returns (system_instruction, contents_list)."""
    system = ""
    contents = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif m["role"] == "assistant" and "tool_calls" in m:
            # Use the original Gemini content object if available so that
            # thought_signatures (required by thinking models) are preserved.
            if "_gemini_content" in m:
                contents.append(m["_gemini_content"])
            else:
                parts = [
                    {"function_call": {"name": tc["name"], "args": tc["input"]}}
                    for tc in m["tool_calls"]
                ]
                contents.append({"role": "model", "parts": parts})
        elif m["role"] == "assistant":
            contents.append({"role": "model", "parts": [{"text": m["content"]}]})
        elif m["role"] == "tool":
            contents.append({
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": m["name"],
                            "response": {"result": m["content"]},
                        }
                    }
                ],
            })
    return system, contents


class GeminiProvider(BaseLLM):
    def __init__(self):
        self.model_name = settings.LLM_MODEL

    @traceable(name="gemini_chat_with_tools")
    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        system, contents = _to_genai_contents(messages)
        client = _get_client()

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=_to_genai_tools(tools),
        )
        response = await client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        content = response.candidates[0].content
        function_calls = [
            part.function_call
            for part in content.parts
            if part.function_call is not None
        ]

        if function_calls:
            return {
                "type": "tool_calls",
                "calls": [
                    {"id": fc.name, "name": fc.name, "input": dict(fc.args)}
                    for fc in function_calls
                ],
                # Preserve full content so thought_signatures survive the round-trip
                "_gemini_content": content,
            }

        text = "".join(
            part.text
            for part in content.parts
            if part.text is not None and not getattr(part, "thought", False)
        )
        return {"type": "text", "content": text}

    @traceable(name="gemini_stream_response")
    async def stream_response(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        system, contents = _to_genai_contents(messages)
        client = _get_client()

        config = types.GenerateContentConfig(system_instruction=system)

        async for chunk in await client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

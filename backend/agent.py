import json
from typing import AsyncGenerator
from langsmith import traceable
from config import settings
from tools.definitions import TOOLS
from tools.search_docs import search_docs
from tools.query_orders import query_orders
from llm.base import BaseLLM

SYSTEM_PROMPT = f"""You are a helpful assistant for the Dual Mode Agentic RAG Chatbot.
Answer questions using only your available tools. Do not make up information.

Rules:
- For questions about company policies, returns, warranty, HR leave, or product FAQ: use search_docs. Do NOT include source or page references in your answer — citations are shown separately in the UI.
- For questions about orders, revenue, customers, or order data: use query_orders with a SELECT statement.
- For questions that require both policy and order data: use both tools.
- If a question is outside your scope, respond exactly: "I don't have that information."
- Never invent policy text, order data, or SQL columns.

Today's date is {settings.CURRENT_DATE}.

The orders table has exactly these columns:
  order_id (TEXT), customer (TEXT), product (TEXT), amount (REAL, rupees),
  status (TEXT: pending/shipped/delivered/cancelled), order_date (TEXT: YYYY-MM-DD)
"""


def _build_messages(history: list[dict], user_message: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def _execute_tool(name: str, input_args: dict) -> tuple[str, list[dict]]:
    """Execute a tool and return (serialized_result, metadata_events)."""
    metadata_events = []
    if name == "search_docs":
        results = search_docs(input_args["query"])
        for chunk in results:
            metadata_events.append({
                "type": "citation",
                "source": chunk["source"],
                "page": chunk["page"],
            })
        return json.dumps(results), metadata_events
    elif name == "query_orders":
        try:
            result = query_orders(input_args["sql"])
            metadata_events.append({"type": "sql", "query": result["sql"]})
            return json.dumps(result["rows"]), metadata_events
        except ValueError as e:
            return json.dumps({"error": str(e)}), metadata_events
    return json.dumps({"error": f"Unknown tool: {name}"}), metadata_events


@traceable(name="run_agent")
async def run_agent(
    message: str,
    history: list[dict],
    llm: BaseLLM,
) -> AsyncGenerator[dict, None]:
    messages = _build_messages(history, message)
    accumulated_tool_events: list[dict] = []

    for _ in range(3):  # max iterations guard
        response = await llm.chat_with_tools(messages, TOOLS)

        if response["type"] == "text":
            break

        for call in response["calls"]:
            yield {"type": "tool_use", "tool": call["name"], "input": str(call["input"])}

            result_str, metadata_events = _execute_tool(call["name"], call["input"])
            accumulated_tool_events.extend(metadata_events)

            # Append assistant tool call to messages.
            # _gemini_content carries thought_signatures required by Gemini thinking models.
            assistant_msg: dict = {"role": "assistant", "tool_calls": [call]}
            if "_gemini_content" in response:
                assistant_msg["_gemini_content"] = response["_gemini_content"]
            messages.append(assistant_msg)
            # Append tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": result_str,
            })

    # Yield citation / SQL metadata events before streaming the answer
    for event in accumulated_tool_events:
        yield event

    # Stream final answer
    async for token in llm.stream_response(messages):
        yield {"type": "token", "content": token}

    yield {"type": "done"}

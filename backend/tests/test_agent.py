import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agent import run_agent


@pytest.fixture
def mock_llm_tool_then_text():
    llm = MagicMock()
    llm.chat_with_tools = AsyncMock(side_effect=[
        {
            "type": "tool_calls",
            "calls": [{"id": "call1", "name": "query_orders", "input": {"sql": "SELECT COUNT(*) as total FROM orders"}}],
        },
        {"type": "text", "content": "unused"},
    ])

    async def fake_stream(messages):
        for token in ["There ", "are ", "5 ", "orders."]:
            yield token

    llm.stream_response = fake_stream
    return llm


@pytest.fixture
def mock_llm_text_only():
    llm = MagicMock()
    llm.chat_with_tools = AsyncMock(return_value={"type": "text", "content": "unused"})

    async def fake_stream(messages):
        for token in ["I ", "don't ", "know."]:
            yield token

    llm.stream_response = fake_stream
    return llm


async def collect_events(gen):
    return [event async for event in gen]


@pytest.mark.asyncio
async def test_agent_sql_tool_call(mock_llm_tool_then_text):
    mock_result = {"rows": [{"total": 5}], "sql": "SELECT COUNT(*) as total FROM orders"}
    with patch("agent.query_orders", return_value=mock_result):
        events = await collect_events(
            run_agent("How many orders?", [], mock_llm_tool_then_text)
        )

    types = [e["type"] for e in events]
    assert "tool_use" in types
    assert "sql" in types
    assert "token" in types
    assert events[-1]["type"] == "done"

    sql_event = next(e for e in events if e["type"] == "sql")
    assert "SELECT" in sql_event["query"]


@pytest.mark.asyncio
async def test_agent_no_tools_streams_response(mock_llm_text_only):
    events = await collect_events(
        run_agent("What is the weather?", [], mock_llm_text_only)
    )
    types = [e["type"] for e in events]
    assert "tool_use" not in types
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) > 0
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_agent_rag_tool_call():
    llm = MagicMock()
    llm.chat_with_tools = AsyncMock(side_effect=[
        {
            "type": "tool_calls",
            "calls": [{"id": "c1", "name": "search_docs", "input": {"query": "refund window"}}],
        },
        {"type": "text", "content": "unused"},
    ])

    async def fake_stream(messages):
        yield "30 days."

    llm.stream_response = fake_stream

    mock_chunks = [{"text": "30 day return window.", "source": "returns_policy.pdf", "page": 1}]
    with patch("agent.search_docs", return_value=mock_chunks):
        events = await collect_events(run_agent("What is the refund window?", [], llm))

    citation_events = [e for e in events if e["type"] == "citation"]
    assert len(citation_events) == 1
    assert citation_events[0]["source"] == "returns_policy.pdf"

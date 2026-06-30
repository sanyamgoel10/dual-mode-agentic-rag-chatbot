import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_health_endpoint():
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_chat_endpoint_streams():
    async def fake_agent(message, history, llm):
        yield {"type": "token", "content": "Hello"}
        yield {"type": "done"}

    from main import app
    with patch("main.run_agent", side_effect=fake_agent), \
         patch("main.get_llm", return_value=MagicMock()), \
         patch("main.init_db"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            async with client.stream(
                "POST", "/chat", json={"message": "hi", "history": []}
            ) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]
                body = await response.aread()

    lines = [l for l in body.decode().split("\n") if l.startswith("data: ")]
    events = [json.loads(l[6:]) for l in lines]
    assert any(e["type"] == "token" for e in events)
    assert events[-1]["type"] == "done"

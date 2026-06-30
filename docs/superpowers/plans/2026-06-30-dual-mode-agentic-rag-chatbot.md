# Dual-Mode Agentic RAG Chatbot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI + Next.js chatbot that autonomously routes questions to vector RAG (Pinecone) or text-to-SQL (SQLite) using bare LLM tool-calling, with token-level SSE streaming.

**Architecture:** A single agent loop in `agent.py` calls the LLM with two tool definitions; the LLM decides which tool(s) to invoke per turn. Tools return structured results that the LLM uses to stream a final answer. The frontend renders tool badges, citations, and SQL alongside streaming text.

**Tech Stack:** Python 3.11, FastAPI, Pinecone (v3+), Google Generative AI SDK (`google-generativeai`), OpenAI SDK, Anthropic SDK, SQLite (stdlib), LangSmith, PyMuPDF, pytest, pytest-asyncio. Frontend: Next.js 14, TypeScript, Tailwind CSS.

## Global Constraints

- Python 3.11+, Node.js 18+
- Pinecone index must have dimension=768 (text-embedding-004 output)
- All credentials via environment variables only — never hardcoded
- SSE stream uses `text/event-stream` with `data: <json>\n\n` format
- `query_orders` tool accepts SELECT statements only — reject any other statement
- Agent loop max iterations: 3 per turn
- `CURRENT_DATE` env var defaults to `"2026-06-15"` (assignment grading requirement)
- FastAPI must have CORS enabled for the Vercel frontend origin
- LangSmith `@traceable` on all tool functions and the agent loop

---

## File Map

```
dual_mode_agntic_rag_chatbot/
├── database/                          (existing — do not modify)
│   ├── orders.csv
│   ├── hr_leave_policy.pdf
│   ├── pricing_discounts_policy.pdf
│   ├── product_faq.pdf
│   ├── returns_policy.pdf
│   └── warranty_policy.pdf
├── backend/
│   ├── config.py                      Task 1 — env var loading
│   ├── startup.py                     Task 2 — CSV → SQLite on app start
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── definitions.py             Task 3 — canonical tool schemas
│   │   ├── query_orders.py            Task 3 — SQL safety + execution
│   │   └── search_docs.py             Task 4 — Pinecone search + citations
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                    Task 5 — abstract interface
│   │   ├── openai_provider.py         Task 5
│   │   ├── claude_provider.py         Task 5
│   │   ├── gemini_provider.py         Task 5
│   │   └── factory.py                 Task 5 — provider factory
│   ├── agent.py                       Task 6 — agent loop + SSE event generator
│   ├── main.py                        Task 7 — FastAPI app + /chat endpoint
│   ├── ingest.py                      Task 8 — manual PDF → Pinecone indexing
│   ├── requirements.txt               Task 1
│   ├── .env.example                   Task 1
│   └── Dockerfile                     Task 10
├── backend/tests/
│   ├── conftest.py                    Task 1
│   ├── test_startup.py                Task 2
│   ├── test_query_orders.py           Task 3
│   ├── test_search_docs.py            Task 4
│   ├── test_agent.py                  Task 6
│   └── test_main.py                   Task 7
├── frontend/
│   ├── app/
│   │   ├── layout.tsx                 Task 9
│   │   └── page.tsx                   Task 9
│   ├── components/
│   │   ├── ChatWindow.tsx             Task 9
│   │   ├── MessageBubble.tsx          Task 9
│   │   └── StreamingDot.tsx           Task 9
│   ├── lib/
│   │   └── api.ts                     Task 9 — SSE stream reader
│   ├── package.json                   Task 9
│   ├── tailwind.config.ts             Task 9
│   └── Dockerfile                     Task 10
├── docker-compose.yml                 Task 10
└── README.md                          Task 11
```

---

### Task 1: Project Scaffolding & Configuration

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/config.py`
- Create: `backend/.env.example`
- Create: `backend/tests/conftest.py`
- Create: `backend/tools/__init__.py`
- Create: `backend/llm/__init__.py`

**Interfaces:**
- Produces: `config.settings` object consumed by all backend modules

- [ ] **Step 1: Create `backend/requirements.txt`**

```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
python-dotenv>=1.0.0
pinecone>=3.2.0
google-generativeai>=0.7.0
openai>=1.30.0
anthropic>=0.28.0
PyMuPDF>=1.24.0
pandas>=2.2.0
langsmith>=0.1.77
pydantic>=2.0.0
pytest>=8.2.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

- [ ] **Step 2: Create `backend/config.py`**

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "emb-chatbot")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "emb-chatbot")
    CURRENT_DATE: str = os.getenv("CURRENT_DATE", "2026-06-15")
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "/app/data/orders.db")
    ORDERS_CSV_PATH: str = os.getenv("ORDERS_CSV_PATH", "/app/database/orders.csv")
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")

settings = Settings()
```

- [ ] **Step 3: Create `backend/.env.example`**

```bash
# Chat LLM — set one provider
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Embeddings (always Google text-embedding-004)
# GOOGLE_API_KEY already set above

# Vector store — create index with dimension=768, metric=cosine
PINECONE_API_KEY=...
PINECONE_INDEX=emb-chatbot

# Tracing
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=emb-chatbot

# App
CURRENT_DATE=2026-06-15
CORS_ORIGINS=http://localhost:3000,https://your-app.vercel.app
```

- [ ] **Step 4: Create `backend/tests/conftest.py`**

```python
import pytest
import os

os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("LLM_MODEL", "gpt-4o")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_INDEX", "test-index")
os.environ.setdefault("CURRENT_DATE", "2026-06-15")
os.environ.setdefault("DATABASE_PATH", "/tmp/test_orders.db")
os.environ.setdefault("ORDERS_CSV_PATH", "tests/fixtures/orders_sample.csv")
```

- [ ] **Step 5: Create test fixture CSV at `backend/tests/fixtures/orders_sample.csv`**

```
order_id,customer,product,amount,status,order_date
ORD-0001,Alice Smith,Laptop Stand,1899,delivered,2026-05-10
ORD-0002,Bob Jones,USB-C Hub,7497,pending,2026-05-20
ORD-0003,Carol White,Mechanical Keyboard,4999,shipped,2026-06-01
ORD-0004,Alice Smith,Laptop Stand,1899,cancelled,2026-04-15
ORD-0005,Dave Brown,USB-C Hub,7497,delivered,2026-05-25
```

- [ ] **Step 6: Create empty `__init__.py` files**

```bash
touch backend/tools/__init__.py backend/llm/__init__.py
```

- [ ] **Step 7: Install dependencies and verify**

```bash
cd backend && pip install -r requirements.txt
python -c "import fastapi, pinecone, google.generativeai, openai, anthropic, fitz, langsmith; print('OK')"
```

Expected output: `OK`

- [ ] **Step 8: Commit**

```bash
git init
git add backend/requirements.txt backend/config.py backend/.env.example backend/tests/ backend/tools/__init__.py backend/llm/__init__.py
git commit -m "feat: project scaffolding and configuration"
```

---

### Task 2: SQLite Orders Database

**Files:**
- Create: `backend/startup.py`
- Create: `backend/tests/test_startup.py`

**Interfaces:**
- Produces: `startup.init_db(csv_path, db_path)` — called at app startup and in tests
- Produces: SQLite `orders` table with columns: `order_id TEXT, customer TEXT, product TEXT, amount REAL, status TEXT, order_date TEXT`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_startup.py
import sqlite3
import os
import pytest
from startup import init_db

DB_PATH = "/tmp/test_startup.db"
CSV_PATH = "tests/fixtures/orders_sample.csv"

def teardown_function():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def test_init_db_creates_table():
    init_db(CSV_PATH, DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orders'")
    assert cursor.fetchone() is not None
    conn.close()

def test_init_db_loads_rows():
    init_db(CSV_PATH, DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert count == 5
    conn.close()

def test_init_db_correct_columns():
    init_db(CSV_PATH, DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT order_id, customer, product, amount, status, order_date FROM orders LIMIT 1").fetchone()
    assert row is not None
    assert row[0] == "ORD-0001"
    assert row[3] == 1899.0
    conn.close()

def test_init_db_idempotent():
    init_db(CSV_PATH, DB_PATH)
    init_db(CSV_PATH, DB_PATH)  # second call should not error or duplicate
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert count == 5
    conn.close()
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd backend && python -m pytest tests/test_startup.py -v
```

Expected: `ImportError: No module named 'startup'`

- [ ] **Step 3: Implement `backend/startup.py`**

```python
import sqlite3
import pandas as pd
import os
from config import settings

def init_db(csv_path: str = None, db_path: str = None) -> None:
    csv_path = csv_path or settings.ORDERS_CSV_PATH
    db_path = db_path or settings.DATABASE_PATH

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    df.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Orders DB initialised at {settings.DATABASE_PATH}")
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd backend && python -m pytest tests/test_startup.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add backend/startup.py backend/tests/test_startup.py backend/tests/fixtures/
git commit -m "feat: orders CSV to SQLite startup initialisation"
```

---

### Task 3: `query_orders` Tool & Tool Definitions

**Files:**
- Create: `backend/tools/definitions.py`
- Create: `backend/tools/query_orders.py`
- Create: `backend/tests/test_query_orders.py`

**Interfaces:**
- Produces: `query_orders(sql: str, db_path: str) -> dict` → `{"rows": list[dict], "sql": str}`
- Produces: `TOOLS: list[dict]` — canonical tool schemas consumed by all LLM providers
- Raises: `ValueError` if SQL is not a SELECT statement

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_query_orders.py
import pytest
import os
from startup import init_db
from tools.query_orders import query_orders

DB_PATH = "/tmp/test_query.db"
CSV_PATH = "tests/fixtures/orders_sample.csv"

def setup_module():
    init_db(CSV_PATH, DB_PATH)

def teardown_module():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def test_select_returns_rows():
    result = query_orders("SELECT * FROM orders WHERE status='pending'", DB_PATH)
    assert result["sql"] == "SELECT * FROM orders WHERE status='pending'"
    assert len(result["rows"]) == 1
    assert result["rows"][0]["order_id"] == "ORD-0002"

def test_count_query():
    result = query_orders("SELECT COUNT(*) as total FROM orders", DB_PATH)
    assert result["rows"][0]["total"] == 5

def test_sum_query():
    result = query_orders("SELECT SUM(amount) as revenue FROM orders WHERE status='delivered'", DB_PATH)
    assert result["rows"][0]["revenue"] == pytest.approx(1899.0 + 7497.0)

def test_non_select_raises():
    with pytest.raises(ValueError, match="Only SELECT"):
        query_orders("DROP TABLE orders", DB_PATH)

def test_delete_raises():
    with pytest.raises(ValueError, match="Only SELECT"):
        query_orders("DELETE FROM orders WHERE order_id='ORD-0001'", DB_PATH)

def test_empty_result():
    result = query_orders("SELECT * FROM orders WHERE status='nonexistent'", DB_PATH)
    assert result["rows"] == []
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd backend && python -m pytest tests/test_query_orders.py -v
```

Expected: `ImportError: No module named 'tools.query_orders'`

- [ ] **Step 3: Create `backend/tools/definitions.py`**

```python
TOOLS = [
    {
        "name": "search_docs",
        "description": (
            "Search company documents using semantic search. Use for questions about "
            "return policy, warranty, HR leave policy, product FAQ, pricing and discounts. "
            "Returns relevant text with source document name and page number for citation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "query_orders",
        "description": (
            "Query the orders database. Use for questions about orders, revenue, customers, "
            "products sold, order counts, or order status. "
            "Write a SELECT SQL query. Table name: orders. "
            "Columns: order_id (TEXT), customer (TEXT), product (TEXT), "
            "amount (REAL, in rupees), status (TEXT: pending/shipped/delivered/cancelled), "
            "order_date (TEXT: YYYY-MM-DD format)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid SELECT SQL query against the orders table"
                }
            },
            "required": ["sql"]
        }
    }
]
```

- [ ] **Step 4: Implement `backend/tools/query_orders.py`**

```python
import sqlite3
from langsmith import traceable
from config import settings

@traceable(name="query_orders")
def query_orders(sql: str, db_path: str = None) -> dict:
    db_path = db_path or settings.DATABASE_PATH

    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        raise ValueError(f"Only SELECT statements are allowed. Got: {sql[:50]}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

    return {"rows": rows, "sql": sql}
```

- [ ] **Step 5: Run tests to confirm pass**

```bash
cd backend && python -m pytest tests/test_query_orders.py -v
```

Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add backend/tools/definitions.py backend/tools/query_orders.py backend/tests/test_query_orders.py
git commit -m "feat: query_orders tool with SELECT-only safety guard"
```

---

### Task 4: `search_docs` Tool

**Files:**
- Create: `backend/tools/search_docs.py`
- Create: `backend/tests/test_search_docs.py`

**Interfaces:**
- Produces: `search_docs(query: str) -> list[dict]` → `[{"text": str, "source": str, "page": int}, ...]`
- Produces: `embed_text(text: str, task_type: str) -> list[float]` — also used by `ingest.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_search_docs.py
from unittest.mock import patch, MagicMock
from tools.search_docs import search_docs, embed_text

def make_mock_match(text, source, page, score=0.9):
    match = MagicMock()
    match.metadata = {"text": text, "source": source, "page": page}
    match.score = score
    return match

def test_search_docs_returns_formatted_results():
    mock_embedding = [0.1] * 768

    mock_matches = [
        make_mock_match("Returns within 30 days are accepted.", "returns_policy.pdf", 1),
        make_mock_match("Products must be unused.", "returns_policy.pdf", 2),
    ]

    mock_query_result = MagicMock()
    mock_query_result.matches = mock_matches

    with patch("tools.search_docs.embed_text", return_value=mock_embedding), \
         patch("tools.search_docs._get_index") as mock_index_fn:

        mock_index = MagicMock()
        mock_index.query.return_value = mock_query_result
        mock_index_fn.return_value = mock_index

        results = search_docs("return policy")

    assert len(results) == 2
    assert results[0]["text"] == "Returns within 30 days are accepted."
    assert results[0]["source"] == "returns_policy.pdf"
    assert results[0]["page"] == 1

def test_search_docs_empty_results():
    mock_query_result = MagicMock()
    mock_query_result.matches = []

    with patch("tools.search_docs.embed_text", return_value=[0.1] * 768), \
         patch("tools.search_docs._get_index") as mock_index_fn:

        mock_index = MagicMock()
        mock_index.query.return_value = mock_query_result
        mock_index_fn.return_value = mock_index

        results = search_docs("something obscure")

    assert results == []
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd backend && python -m pytest tests/test_search_docs.py -v
```

Expected: `ImportError: No module named 'tools.search_docs'`

- [ ] **Step 3: Implement `backend/tools/search_docs.py`**

```python
import google.generativeai as genai
from pinecone import Pinecone
from langsmith import traceable
from config import settings

genai.configure(api_key=settings.GOOGLE_API_KEY)

_pinecone_index = None

def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _pinecone_index = pc.Index(settings.PINECONE_INDEX)
    return _pinecone_index

def embed_text(text: str, task_type: str = "retrieval_query") -> list[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type=task_type,
    )
    return result["embedding"]

@traceable(name="search_docs")
def search_docs(query: str) -> list[dict]:
    query_embedding = embed_text(query, task_type="retrieval_query")
    index = _get_index()
    response = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
    )
    return [
        {
            "text": match.metadata.get("text", ""),
            "source": match.metadata.get("source", "unknown"),
            "page": match.metadata.get("page", 1),
        }
        for match in response.matches
    ]
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd backend && python -m pytest tests/test_search_docs.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add backend/tools/search_docs.py backend/tests/test_search_docs.py
git commit -m "feat: search_docs tool with Pinecone vector search and Google embeddings"
```

---

### Task 5: LLM Provider Abstraction

**Files:**
- Create: `backend/llm/base.py`
- Create: `backend/llm/openai_provider.py`
- Create: `backend/llm/claude_provider.py`
- Create: `backend/llm/gemini_provider.py`
- Create: `backend/llm/factory.py`

**Interfaces:**
- Produces: `BaseLLM.chat_with_tools(messages, tools) -> dict` — `{"type": "tool_calls", "calls": [...]}` or `{"type": "text", "content": str}`
- Produces: `BaseLLM.stream_response(messages) -> AsyncGenerator[str, None]` — yields string tokens
- Produces: `get_llm() -> BaseLLM` — factory function consumed by `agent.py`

**Normalized internal message format** (used by `agent.py`, translated by each provider):
```python
{"role": "system",    "content": str}
{"role": "user",      "content": str}
{"role": "assistant", "content": str}                                          # text response
{"role": "assistant", "tool_calls": [{"id": str, "name": str, "input": dict}]}  # tool call
{"role": "tool",      "tool_call_id": str, "name": str, "content": str}       # tool result
```

- [ ] **Step 1: Create `backend/llm/base.py`**

```python
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
```

- [ ] **Step 2: Implement `backend/llm/openai_provider.py`**

```python
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
```

- [ ] **Step 3: Implement `backend/llm/claude_provider.py`**

```python
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
    """Returns (system_prompt, messages_list). Claude system prompt is separate."""
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
                    "content": [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}],
                })
        elif m["role"] == "assistant" and "tool_calls" in m:
            result.append({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
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
        text = next((block.text for block in response.content if block.type == "text"), "")
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
```

- [ ] **Step 4: Implement `backend/llm/gemini_provider.py`**

```python
import json
import asyncio
from typing import AsyncGenerator
import google.generativeai as genai
from langsmith import traceable
from .base import BaseLLM
from config import settings

genai.configure(api_key=settings.GOOGLE_API_KEY)

def _to_gemini_tools(tools: list[dict]) -> list:
    from google.generativeai.types import Tool, FunctionDeclaration
    declarations = [
        FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
        )
        for t in tools
    ]
    return [Tool(function_declarations=declarations)]

def _to_gemini_contents(messages: list[dict]) -> tuple[str, list[dict]]:
    """Returns (system_instruction, contents_list)."""
    system = ""
    contents = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif m["role"] == "assistant" and "tool_calls" in m:
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
                "parts": [{"function_response": {"name": m["name"], "response": {"result": m["content"]}}}],
            })
    return system, contents

class GeminiProvider(BaseLLM):
    def __init__(self):
        self.model_name = settings.LLM_MODEL

    @traceable(name="gemini_chat_with_tools")
    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        system, contents = _to_gemini_contents(messages)
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
            tools=_to_gemini_tools(tools),
        )
        response = await asyncio.to_thread(model.generate_content, contents)
        part = response.candidates[0].content.parts[0]
        if hasattr(part, "function_call") and part.function_call.name:
            fc = part.function_call
            return {
                "type": "tool_calls",
                "calls": [{"id": fc.name, "name": fc.name, "input": dict(fc.args)}],
            }
        return {"type": "text", "content": part.text}

    @traceable(name="gemini_stream_response")
    async def stream_response(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        system, contents = _to_gemini_contents(messages)
        model = genai.GenerativeModel(model_name=self.model_name, system_instruction=system)

        def _stream():
            return model.generate_content(contents, stream=True)

        response = await asyncio.to_thread(_stream)
        for chunk in response:
            if chunk.text:
                yield chunk.text
```

- [ ] **Step 5: Create `backend/llm/factory.py`**

```python
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
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Choose openai, claude, or gemini.")
    return _llm_instance
```

- [ ] **Step 6: Verify imports are clean**

```bash
cd backend && python -c "from llm.factory import get_llm; print('LLM factory OK')"
```

Expected: `LLM factory OK`

- [ ] **Step 7: Commit**

```bash
git add backend/llm/
git commit -m "feat: multi-provider LLM abstraction (OpenAI, Claude, Gemini)"
```

---

### Task 6: Agent Loop

**Files:**
- Create: `backend/agent.py`
- Create: `backend/tests/test_agent.py`

**Interfaces:**
- Consumes: `get_llm() -> BaseLLM` from `llm/factory.py`
- Consumes: `search_docs(query) -> list[dict]` from `tools/search_docs.py`
- Consumes: `query_orders(sql) -> dict` from `tools/query_orders.py`
- Consumes: `TOOLS: list[dict]` from `tools/definitions.py`
- Produces: `run_agent(message, history, llm) -> AsyncGenerator[dict, None]`
  - Yields events: `{"type": "tool_use", "tool": str, "input": str}`, `{"type": "citation", "source": str, "page": int}`, `{"type": "sql", "query": str}`, `{"type": "token", "content": str}`, `{"type": "done"}`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_agent.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from agent import run_agent

@pytest.fixture
def mock_llm_tool_then_text():
    llm = MagicMock()
    llm.chat_with_tools = AsyncMock(side_effect=[
        {
            "type": "tool_calls",
            "calls": [{"id": "call1", "name": "query_orders", "input": {"sql": "SELECT COUNT(*) as total FROM orders"}}]
        },
        {"type": "text", "content": "unused - we stream separately"}
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
        events = await collect_events(run_agent("How many orders?", [], mock_llm_tool_then_text))

    types = [e["type"] for e in events]
    assert "tool_use" in types
    assert "sql" in types
    assert "token" in types
    assert events[-1]["type"] == "done"

    sql_event = next(e for e in events if e["type"] == "sql")
    assert "SELECT" in sql_event["query"]

@pytest.mark.asyncio
async def test_agent_no_tools_streams_response(mock_llm_text_only):
    events = await collect_events(run_agent("What is the weather?", [], mock_llm_text_only))

    types = [e["type"] for e in events]
    assert "tool_use" not in types
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) > 0
    assert events[-1]["type"] == "done"

@pytest.mark.asyncio
async def test_agent_rag_tool_call():
    llm = MagicMock()
    llm.chat_with_tools = AsyncMock(side_effect=[
        {"type": "tool_calls", "calls": [{"id": "c1", "name": "search_docs", "input": {"query": "refund window"}}]},
        {"type": "text", "content": "unused"}
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd backend && python -m pytest tests/test_agent.py -v
```

Expected: `ImportError: No module named 'agent'`

- [ ] **Step 3: Implement `backend/agent.py`**

```python
import json
import os
from typing import AsyncGenerator
from langsmith import traceable
from config import settings
from tools.definitions import TOOLS
from tools.search_docs import search_docs
from tools.query_orders import query_orders
from llm.base import BaseLLM

SYSTEM_PROMPT = f"""You are a helpful company assistant for EMB Global.
Answer questions using only your available tools. Do not make up information.

Rules:
- For questions about company policies, returns, warranty, HR leave, or product FAQ: use search_docs and always cite the source document and page.
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
    tools_were_used = False

    for _ in range(3):  # max iterations
        response = await llm.chat_with_tools(messages, TOOLS)

        if response["type"] == "text":
            break

        tools_were_used = True
        for call in response["calls"]:
            yield {"type": "tool_use", "tool": call["name"], "input": str(call["input"])}

            result_str, metadata_events = _execute_tool(call["name"], call["input"])
            accumulated_tool_events.extend(metadata_events)

            # Append assistant tool call to messages
            messages.append({"role": "assistant", "tool_calls": [call]})
            # Append tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": result_str,
            })

    # Yield citation/SQL metadata events
    for event in accumulated_tool_events:
        yield event

    # Stream final answer
    async for token in llm.stream_response(messages):
        yield {"type": "token", "content": token}

    yield {"type": "done"}
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd backend && python -m pytest tests/test_agent.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/agent.py backend/tests/test_agent.py
git commit -m "feat: agent loop with LLM tool-calling, SSE event generation, and LangSmith tracing"
```

---

### Task 7: FastAPI Application & SSE Endpoint

**Files:**
- Create: `backend/main.py`
- Create: `backend/tests/test_main.py`

**Interfaces:**
- Produces: `POST /chat` — accepts `{"message": str, "history": [{"role": str, "content": str}]}`, returns `text/event-stream`
- Produces: `GET /health` — returns `{"status": "ok"}`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_main.py
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
         patch("main.get_llm", return_value=MagicMock()):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            async with client.stream("POST", "/chat", json={"message": "hi", "history": []}) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]
                body = await response.aread()

    lines = [l for l in body.decode().split("\n") if l.startswith("data: ")]
    events = [json.loads(l[6:]) for l in lines]
    assert any(e["type"] == "token" for e in events)
    assert events[-1]["type"] == "done"
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd backend && python -m pytest tests/test_main.py -v
```

Expected: `ImportError: No module named 'main'`

- [ ] **Step 3: Implement `backend/main.py`**

```python
import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from config import settings
from startup import init_db
from agent import run_agent
from llm.factory import get_llm

app = FastAPI(title="EMB Global RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/health")
async def health():
    return {"status": "ok"}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []

@app.post("/chat")
async def chat(request: ChatRequest):
    llm = get_llm()
    history = [{"role": m.role, "content": m.content} for m in request.history]

    async def event_stream():
        async for event in run_agent(request.message, history, llm):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd backend && python -m pytest tests/test_main.py -v
```

Expected: 2 passed

- [ ] **Step 5: Run all tests to confirm no regressions**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 6: Start server locally and verify**

```bash
cd backend && uvicorn main:app --reload --port 8000
# In another terminal:
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 7: Commit**

```bash
git add backend/main.py backend/tests/test_main.py
git commit -m "feat: FastAPI app with SSE /chat endpoint and startup DB initialisation"
```

---

### Task 8: PDF Ingestion Script

**Files:**
- Create: `backend/ingest.py`

Note: This script has external dependencies (Pinecone, Google API). It is not unit-tested — run it manually with real credentials.

- [ ] **Step 1: Implement `backend/ingest.py`**

```python
import os
import fitz  # PyMuPDF
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from tools.search_docs import embed_text
from config import settings

DATABASE_DIR = Path(__file__).parent.parent / "database"
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 30

def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Returns list of (page_num, text) for each page with content."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append((i, text))
    return pages

def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE_WORDS])
        chunks.append(chunk)
        i += CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
    return chunks

def get_or_create_index(pc: Pinecone) -> object:
    existing = [idx.name for idx in pc.list_indexes()]
    if settings.PINECONE_INDEX not in existing:
        pc.create_index(
            name=settings.PINECONE_INDEX,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index: {settings.PINECONE_INDEX}")
    return pc.Index(settings.PINECONE_INDEX)

def ingest_all():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = get_or_create_index(pc)

    pdf_files = list(DATABASE_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files: {[f.name for f in pdf_files]}")

    vectors = []
    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)
        for page_num, page_text in pages:
            chunks = chunk_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                embedding = embed_text(chunk, task_type="retrieval_document")
                vector_id = f"{pdf_path.stem}-p{page_num}-c{chunk_idx}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_num,
                        "text": chunk,
                    },
                })
                print(f"  Embedded: {vector_id}")

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])
        print(f"Upserted batch {i // batch_size + 1}")

    print(f"\nDone. {len(vectors)} vectors indexed to '{settings.PINECONE_INDEX}'.")

if __name__ == "__main__":
    ingest_all()
```

- [ ] **Step 2: Run ingestion with real credentials**

```bash
cd backend
cp .env.example .env   # fill in real API keys
python ingest.py
```

Expected output:
```
Found 5 PDF files: [...]
  Embedded: hr_leave_policy-p1-c0
  ...
Done. XX vectors indexed to 'emb-chatbot'.
```

- [ ] **Step 3: Commit**

```bash
git add backend/ingest.py
git commit -m "feat: PDF ingestion script — chunks PDFs and upserts to Pinecone"
```

---

### Task 9: Next.js Frontend

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/app/layout.tsx`
- Create: `frontend/app/page.tsx`
- Create: `frontend/lib/api.ts`
- Create: `frontend/components/ChatWindow.tsx`
- Create: `frontend/components/MessageBubble.tsx`
- Create: `frontend/components/StreamingDot.tsx`

**Interfaces:**
- Consumes: `POST {NEXT_PUBLIC_API_URL}/chat` — the FastAPI SSE endpoint
- Produces: Chat UI at `/` with streaming responses, tool badges, citations, SQL display

- [ ] **Step 1: Scaffold Next.js app**

```bash
cd frontend
npx create-next-app@14 . --typescript --tailwind --app --no-src-dir --import-alias "@/*" --eslint
```

Answer prompts: TypeScript=Yes, Tailwind=Yes, App Router=Yes, `src/` dir=No.

- [ ] **Step 2: Create `frontend/lib/api.ts`**

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

export interface ChatEvent {
  type: "tool_use" | "citation" | "sql" | "token" | "done" | "error"
  tool?: string
  input?: string
  source?: string
  page?: number
  query?: string
  content?: string
}

export async function* streamChat(
  message: string,
  history: ChatMessage[]
): AsyncGenerator<ChatEvent> {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  })

  if (!response.ok || !response.body) {
    yield { type: "error", content: "Failed to connect to backend" }
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6)) as ChatEvent
        } catch {}
      }
    }
  }
}
```

- [ ] **Step 3: Create `frontend/components/StreamingDot.tsx`**

```tsx
export default function StreamingDot() {
  return (
    <span className="inline-flex items-center gap-1 ml-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </span>
  )
}
```

- [ ] **Step 4: Create `frontend/components/MessageBubble.tsx`**

```tsx
import { ChatEvent } from "@/lib/api"

interface AssistantMessage {
  content: string
  tools: string[]
  citations: { source: string; page: number }[]
  sqlQueries: string[]
  isStreaming: boolean
}

interface Props {
  role: "user" | "assistant"
  content: string
  assistantMeta?: Omit<AssistantMessage, "content" | "isStreaming">
  isStreaming?: boolean
}

function ToolBadge({ tools }: { tools: string[] }) {
  if (tools.length === 0) return null
  const hasRag = tools.includes("search_docs")
  const hasSql = tools.includes("query_orders")
  const label = hasRag && hasSql ? "RAG + SQL" : hasRag ? "RAG" : "SQL"
  const colour = hasRag && hasSql
    ? "bg-purple-100 text-purple-700"
    : hasRag
    ? "bg-blue-100 text-blue-700"
    : "bg-green-100 text-green-700"
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${colour}`}>
      {label}
    </span>
  )
}

export default function MessageBubble({ role, content, assistantMeta, isStreaming }: Props) {
  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] bg-blue-600 text-white px-4 py-2 rounded-2xl rounded-tr-sm text-sm">
          {content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] space-y-2">
        <div className="bg-white border border-gray-200 px-4 py-3 rounded-2xl rounded-tl-sm shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium text-gray-500">Assistant</span>
            {assistantMeta && <ToolBadge tools={assistantMeta.tools} />}
          </div>
          <p className="text-sm text-gray-800 whitespace-pre-wrap">
            {content}
            {isStreaming && (
              <span className="inline-flex items-center gap-0.5 ml-1">
                {[0,1,2].map(i => (
                  <span key={i} className="w-1 h-1 rounded-full bg-gray-400 animate-bounce"
                    style={{ animationDelay: `${i*0.15}s` }} />
                ))}
              </span>
            )}
          </p>
        </div>

        {assistantMeta?.citations?.length > 0 && (
          <div className="flex flex-col gap-1 pl-2">
            {assistantMeta.citations.map((c, i) => (
              <div key={i} className="text-xs text-gray-500 flex items-center gap-1">
                <span>📄</span>
                <span className="font-medium">{c.source}</span>
                <span>· page {c.page}</span>
              </div>
            ))}
          </div>
        )}

        {assistantMeta?.sqlQueries?.length > 0 && (
          <div className="pl-2">
            {assistantMeta.sqlQueries.map((sql, i) => (
              <pre key={i} className="text-xs bg-gray-900 text-green-400 px-3 py-2 rounded-lg overflow-x-auto">
                {sql}
              </pre>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Create `frontend/components/ChatWindow.tsx`**

```tsx
"use client"
import { useState, useRef, useEffect } from "react"
import { streamChat, ChatMessage } from "@/lib/api"
import MessageBubble from "./MessageBubble"

interface DisplayMessage {
  role: "user" | "assistant"
  content: string
  tools: string[]
  citations: { source: string; page: number }[]
  sqlQueries: string[]
  isStreaming: boolean
}

export default function ChatWindow() {
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  async function handleSend() {
    if (!input.trim() || isLoading) return

    const userText = input.trim()
    setInput("")
    setIsLoading(true)

    const userMsg: DisplayMessage = {
      role: "user", content: userText,
      tools: [], citations: [], sqlQueries: [], isStreaming: false
    }
    setMessages(prev => [...prev, userMsg])

    const assistantMsg: DisplayMessage = {
      role: "assistant", content: "",
      tools: [], citations: [], sqlQueries: [], isStreaming: true
    }
    setMessages(prev => [...prev, assistantMsg])

    const history: ChatMessage[] = messages.map(m => ({ role: m.role, content: m.content }))

    try {
      for await (const event of streamChat(userText, history)) {
        setMessages(prev => {
          const updated = [...prev]
          const last = { ...updated[updated.length - 1] }
          if (event.type === "token") {
            last.content += event.content ?? ""
          } else if (event.type === "tool_use" && event.tool) {
            if (!last.tools.includes(event.tool)) last.tools = [...last.tools, event.tool]
          } else if (event.type === "citation" && event.source) {
            last.citations = [...last.citations, { source: event.source, page: event.page ?? 1 }]
          } else if (event.type === "sql" && event.query) {
            last.sqlQueries = [...last.sqlQueries, event.query]
          } else if (event.type === "done") {
            last.isStreaming = false
          }
          updated[updated.length - 1] = last
          return updated
        })
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto">
      <header className="px-6 py-4 border-b bg-white">
        <h1 className="text-lg font-semibold text-gray-900">EMB Global Assistant</h1>
        <p className="text-xs text-gray-500">Ask about company policies or order data</p>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4 bg-gray-50">
        {messages.length === 0 && (
          <p className="text-center text-sm text-gray-400 mt-20">
            Ask a question about company policies or orders...
          </p>
        )}
        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            role={msg.role}
            content={msg.content}
            assistantMeta={msg.role === "assistant" ? {
              tools: msg.tools,
              citations: msg.citations,
              sqlQueries: msg.sqlQueries
            } : undefined}
            isStreaming={msg.isStreaming}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="px-6 py-4 border-t bg-white">
        <div className="flex gap-2">
          <input
            className="flex-1 border border-gray-300 rounded-xl px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Type your message..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && !e.shiftKey && handleSend()}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="bg-blue-600 text-white px-4 py-2 rounded-xl text-sm font-medium disabled:opacity-50 hover:bg-blue-700"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 6: Create `frontend/app/layout.tsx`**

```tsx
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "EMB Global Assistant",
  description: "Dual-mode agentic RAG chatbot",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}
```

- [ ] **Step 7: Create `frontend/app/page.tsx`**

```tsx
import ChatWindow from "@/components/ChatWindow"

export default function Home() {
  return <ChatWindow />
}
```

- [ ] **Step 8: Create `frontend/.env.local`**

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

- [ ] **Step 9: Run frontend locally and verify**

```bash
cd frontend && npm run dev
```

Open `http://localhost:3000`. Verify chat UI loads. With backend running, send a test message.

- [ ] **Step 10: Commit**

```bash
git add frontend/
git commit -m "feat: Next.js chat UI with SSE streaming, tool badges, citations, and SQL display"
```

---

### Task 10: Docker & docker-compose

**Files:**
- Create: `backend/Dockerfile`
- Create: `frontend/Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create `backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN mkdir -p /app/data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python startup.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
```

- [ ] **Step 2: Create `frontend/Dockerfile`**

```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
```

- [ ] **Step 3: Add `output: "standalone"` to `frontend/next.config.ts`**

```typescript
import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  output: "standalone",
}

export default nextConfig
```

- [ ] **Step 4: Create `docker-compose.yml`**

```yaml
version: "3.9"

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./database:/app/database:ro
    env_file:
      - ./backend/.env
    environment:
      - ORDERS_CSV_PATH=/app/database/orders.csv
      - DATABASE_PATH=/app/data/orders.db

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
```

- [ ] **Step 5: Build and test locally**

```bash
# From repo root
docker compose build
docker compose up
```

Open `http://localhost:3000` — should load the chat UI. Test a question.

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 6: Commit**

```bash
git add backend/Dockerfile frontend/Dockerfile frontend/next.config.ts docker-compose.yml
git commit -m "feat: Docker build for backend and frontend with docker-compose for local dev"
```

---

### Task 11: Deployment & README

**Files:**
- Create: `README.md`
- Create: `render.yaml` (optional Render IaC)

- [ ] **Step 1: Deploy backend to Render**

1. Push repo to GitHub (must be public)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect the repo, set root directory to `backend/`
4. Runtime: Docker
5. Set all environment variables from `.env.example`:
   - `LLM_PROVIDER`, `LLM_MODEL`, relevant API key
   - `GOOGLE_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX`
   - `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`
   - `CURRENT_DATE=2026-06-15`
   - `ORDERS_CSV_PATH=/app/database/orders.csv`
   - `DATABASE_PATH=/app/data/orders.db`
   - `CORS_ORIGINS=https://your-app.vercel.app`
6. Add a persistent disk at `/app/data` (optional — SQLite rebuilds on restart anyway)
7. Note the Render URL (e.g. `https://emb-chatbot-backend.onrender.com`)

- [ ] **Step 2: Deploy frontend to Vercel**

1. Go to [vercel.com](https://vercel.com) → New Project → import the repo
2. Set root directory to `frontend/`
3. Framework: Next.js (auto-detected)
4. Environment variable: `NEXT_PUBLIC_API_URL=https://emb-chatbot-backend.onrender.com`
5. Deploy
6. Note the Vercel URL

- [ ] **Step 3: Update backend CORS_ORIGINS on Render**

Set `CORS_ORIGINS=https://your-app.vercel.app` (use actual Vercel URL), then redeploy.

- [ ] **Step 4: Write `README.md`**

```markdown
# Dual-Mode Agentic RAG Chatbot

**Live demo:** https://your-app.vercel.app

## Architecture

A single FastAPI backend runs an agent loop that calls the configured LLM with two tool definitions. The LLM autonomously decides which tool(s) to use per question. Tools return structured results; the LLM synthesizes a streamed final answer delivered via Server-Sent Events.

- **search_docs** → embeds the query with Google text-embedding-004, searches Pinecone, returns top-5 chunks with source+page citations
- **query_orders** → accepts a SELECT SQL query generated by the LLM, executes against SQLite, returns rows

A Next.js frontend renders streaming tokens and shows which tool(s) were used, document citations, and the executed SQL.

## Model & Embedding Choices

| Layer | Choice | Reason |
|---|---|---|
| LLM | Configurable (OpenAI / Claude / Gemini) | Each provider's native tool-calling API is used directly — no framework |
| Embeddings | Google text-embedding-004 (dim=768) | Free tier, high quality, consistent dimension |
| Vector store | Pinecone | Managed, no infra, persists across deployments |
| Structured data | SQLite | Fixed 200-row dataset, rebuilt from CSV on each container start |

## How Routing Works

There is no explicit router. Two tools are defined in the LLM's system context with descriptions that clearly delineate when to use each. The LLM reads the user question and calls `search_docs`, `query_orders`, both, or neither — based on its own reasoning. A system prompt guard prevents hallucinated columns or policy text.

## Setup

### Prerequisites
- Pinecone account (free tier) with an index named `emb-chatbot` (dimension=768, cosine)
- Google API key (for text-embedding-004 + optionally Gemini)
- LLM API key (OpenAI / Anthropic / Google)
- LangSmith account (free tier)

### One-time ingestion (index PDFs into Pinecone)
```bash
cd backend
cp .env.example .env  # fill in API keys
pip install -r requirements.txt
python ingest.py
```

### Local development
```bash
cp backend/.env.example backend/.env  # fill in API keys
docker compose up --build
# frontend: http://localhost:3000
# backend:  http://localhost:8000
```

### Environment variables
See `backend/.env.example` for all required variables.

## Known Limitations

- Render free tier spins down after 15 min inactivity — first request has ~30–50s cold start
- Routing relies on LLM judgment; edge-case questions may route to the wrong tool
- Conversation history stores text only — tool call details from previous turns are not replayed
- Claude has no embeddings API; Google text-embedding-004 is always used for embeddings regardless of chat LLM provider
```

- [ ] **Step 5: Final end-to-end verification**

Test each question type against the live URL:

```
Document: "What is the return window?"
  → Expect: [RAG] badge, citation from returns_policy.pdf

Data: "How many orders are pending?"
  → Expect: [SQL] badge, SELECT query shown, correct count

Mixed: "Does our return policy apply to order ORD-1001?"
  → Expect: [RAG + SQL] badge, both citation and SQL shown

Out-of-scope: "What is the weather in Mumbai?"
  → Expect: "I don't have that information."
```

- [ ] **Step 6: Final commit**

```bash
git add README.md
git commit -m "docs: README with architecture, setup, routing explanation, and known limitations"
git push origin main
```
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task covering it |
|---|---|
| Unstructured RAG with citations | Task 4 (search_docs), Task 6 (agent emits citation events), Task 9 (frontend renders citations) |
| Text-to-SQL with SELECT safety | Task 3 (query_orders), Task 6 (agent emits sql event), Task 9 (frontend renders SQL) |
| Autonomous routing | Task 6 (agent loop — no explicit router, LLM decides) |
| Mixed questions (both tools) | Task 6 (loop handles multiple tool calls) |
| Out-of-scope fallback | Task 6 (system prompt instruction) + Task 11 (verified in e2e test) |
| Multi-provider LLM | Task 5 (OpenAI + Claude + Gemini providers) |
| Google text-embedding-004 | Task 4 (search_docs.py embed_text), Task 8 (ingest.py) |
| Pinecone | Task 4, Task 8 |
| SQLite from CSV on startup | Task 2 (startup.py), Task 7 (FastAPI startup event) |
| SSE token streaming | Task 7 (/chat endpoint), Task 9 (frontend SSE reader) |
| LangSmith tracing | Tasks 3/4/6 (@traceable decorators) |
| CURRENT_DATE=2026-06-15 | Task 1 (config.py default), Task 6 (system prompt) |
| Docker + docker-compose | Task 10 |
| Render + Vercel deployment | Task 11 |
| README with required sections | Task 11 |

No gaps found.

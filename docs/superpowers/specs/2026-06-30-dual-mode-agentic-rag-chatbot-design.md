# Dual-Mode Agentic RAG Chatbot — Design Spec

**Date:** 2026-06-30
**Assignment:** EMB Global AI Engineer Technical Assessment

---

## 1. Problem Statement

Build a single chatbot for a fictional company that answers questions from two distinct knowledge sources and autonomously decides which source to use per question:

- **Unstructured knowledge** — 5 policy/FAQ PDFs → answered via vector RAG with citations
- **Structured knowledge** — an orders table (~200 rows) → answered via LLM-generated SQL executed against SQLite
- **Mixed questions** — use both sources together
- **Out-of-scope questions** — return a safe fallback, no hallucination

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  FRONTEND (Vercel)                   │
│                    Next.js App                       │
│   Chat UI · Tool Badge · Citation/SQL display        │
└───────────────────┬─────────────────────────────────┘
                    │ HTTP SSE stream
┌───────────────────▼─────────────────────────────────┐
│                 BACKEND (Render)                     │
│                   FastAPI App                        │
│                                                      │
│  ┌─────────────┐    ┌──────────────────────────┐    │
│  │  /chat SSE  │───▶│       Agent Loop         │    │
│  │  endpoint   │    │  (LLM tool-calling loop) │    │
│  └─────────────┘    └────────┬─────────────────┘    │
│                              │                       │
│              ┌───────────────┴──────────────┐        │
│              ▼                              ▼        │
│   ┌─────────────────────┐   ┌────────────────────┐  │
│   │  search_docs tool   │   │ query_orders tool  │  │
│   │  Pinecone search    │   │ SQLite SQL execute │  │
│   └──────────┬──────────┘   └────────┬───────────┘  │
│              │                       │               │
└──────────────┼───────────────────────┼───────────────┘
               ▼                       ▼
        ┌─────────────┐        ┌──────────────┐
        │  Pinecone   │        │   SQLite DB  │
        │(cloud, free)│        │  orders.db   │
        └─────────────┘        └──────────────┘

         LangSmith traces every LLM call + tool execution
```

---

## 3. Technology Decisions

| Layer | Choice | Reasoning |
|---|---|---|
| Agent pattern | Single agent, bare LLM tool-calling | Shows understanding of underlying mechanics; no framework magic hiding the routing logic |
| Chat LLM | Configurable: OpenAI / Claude / Gemini | Multi-provider via env vars; graders can plug in any key |
| Embedding model | Google `text-embedding-004` (dim=768) | Free tier, no cost for small dataset, high quality |
| Vector store | Pinecone (cloud managed) | No infra to manage, persists across deployments, free tier sufficient for 5 docs |
| Structured data | SQLite, rebuilt from `orders.csv` on startup | Fixed dataset, no writes needed; self-contained in container |
| Backend | FastAPI + SSE streaming | Required by assignment; token-level streaming via `StreamingResponse` |
| Frontend | Next.js on Vercel | Purpose-built for Next.js; zero config deploy, no cold starts |
| Backend deploy | Render (free web service) | Docker-native, free tier, cold start acceptable for assessment |
| Tracing | LangSmith | Provider-agnostic, `@traceable` decorator; observability without LangChain |
| Packaging | Docker (backend + frontend) + docker-compose for local dev | Required by assignment |

---

## 4. Project Structure

```
dual_mode_agntic_rag_chatbot/
├── database/                        ← source data files (existing)
│   ├── orders.csv
│   ├── hr_leave_policy.pdf
│   ├── pricing_discounts_policy.pdf
│   ├── product_faq.pdf
│   ├── returns_policy.pdf
│   └── warranty_policy.pdf
├── backend/
│   ├── main.py                      ← FastAPI app, /chat SSE endpoint
│   ├── agent.py                     ← agent loop (LLM tool-calling)
│   ├── tools/
│   │   ├── search_docs.py           ← Pinecone vector search + citations
│   │   └── query_orders.py          ← SQL generation + SQLite execution
│   ├── llm/
│   │   ├── base.py                  ← abstract LLM interface
│   │   ├── openai_provider.py
│   │   ├── claude_provider.py
│   │   └── gemini_provider.py
│   ├── ingest.py                    ← manual one-time: chunk PDFs → Pinecone
│   ├── startup.py                   ← auto on start: orders.csv → SQLite
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/                         ← Next.js app router
│   ├── components/
│   │   ├── ChatWindow.tsx
│   │   ├── MessageBubble.tsx        ← tool badge + citation/SQL panel
│   │   └── StreamingDot.tsx
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml               ← local dev: runs backend + frontend
└── README.md
```

---

## 5. Environment Variables

```bash
# Chat LLM — pick one provider
LLM_PROVIDER=openai          # "openai" | "claude" | "gemini"
LLM_MODEL=gpt-4o             # e.g. gpt-4o | claude-sonnet-4-6 | gemini-2.0-flash
OPENAI_API_KEY=...           # if LLM_PROVIDER=openai
ANTHROPIC_API_KEY=...        # if LLM_PROVIDER=claude
GOOGLE_API_KEY=...           # if LLM_PROVIDER=gemini (also used for embeddings)

# Embeddings — always Google
GOOGLE_API_KEY=...           # text-embedding-004, required regardless of LLM provider

# Vector store
PINECONE_API_KEY=...
PINECONE_INDEX=emb-chatbot   # must be created with dimension=768

# Tracing
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=emb-chatbot

# App
CURRENT_DATE=2026-06-15      # default: 15 June 2026 per assignment spec; override if needed
```

---

## 6. Agent Loop

The agent loop in `agent.py` runs on every chat turn:

```
1. Build messages:
   [system_prompt, ...conversation_history, user_message]

2. Call LLM with two tool definitions:
   - search_docs(query: str)
   - query_orders(sql: str)

3. LLM response type?

   → Tool call(s):
       Execute search_docs and/or query_orders
       Append tool results to messages
       Loop back to step 2

   → Plain text:
       Stream tokens via SSE to frontend
       Done
```

Maximum iterations per turn: 3 (guardrail to prevent infinite loops).

### System Prompt (condensed)

```
You are a company assistant. Answer questions using your tools.
- For policy/FAQ questions: use search_docs and cite the source document.
- For order data questions: use query_orders. Write only SELECT statements.
- For mixed questions: use both tools.
- If the question is out of scope: respond "I don't have that information."
- Today's date is {CURRENT_DATE}.
- Never invent policy text or SQL columns that don't exist.

The orders table has exactly these columns:
  order_id, customer, product, amount, status, order_date
```

---

## 7. Tool Definitions

### `search_docs(query: str)`

```
1. Embed query using Google text-embedding-004
2. Search Pinecone index, top_k=5
3. Each result has metadata: { source, page, text }
4. Return: [{ text, source, page }, ...]
   → LLM cites source + page in final answer
```

### `query_orders(sql: str)`

```
1. Receive SQL string written by the LLM
2. Safety check: must be a SELECT statement (reject anything else)
3. Execute against orders.db (SQLite)
4. Return: { rows: [...], sql: "SELECT ..." }
   → sql string forwarded to frontend for display
```

---

## 8. Data Ingestion

### Pinecone (manual, one-time)

Run `python backend/ingest.py` once after Pinecone index is created:

```
For each PDF in database/:
  1. Extract text with PyMuPDF
  2. Chunk into ~500 token pieces, 50 token overlap
  3. Embed each chunk with Google text-embedding-004
  4. Upsert to Pinecone with metadata: { source, page, text }
```

Pinecone vectors persist across deployments. Re-run only if documents change.

### SQLite (automatic, every startup)

Docker entrypoint runs `python backend/startup.py` before the server starts:

```
1. Read database/orders.csv
2. Create orders.db with "orders" table
3. Load all rows (order_id, customer, product, amount, status, order_date)
```

Fast (~200 rows), idempotent, ensures fresh state on every container start.

---

## 9. Streaming (FastAPI → Next.js)

FastAPI `StreamingResponse` emits newline-delimited JSON events:

```jsonc
{ "type": "tool_use",  "tool": "search_docs", "input": "refund policy" }
{ "type": "tool_use",  "tool": "query_orders", "input": "SELECT ..." }
{ "type": "citation",  "source": "returns_policy.pdf", "page": 1 }
{ "type": "sql",       "query": "SELECT COUNT(*) FROM orders WHERE status='pending'" }
{ "type": "token",     "content": "The " }
{ "type": "token",     "content": "refund window " }
{ "type": "done" }
```

Frontend reads the stream via `fetch` with `ReadableStream`, renders tokens as they arrive, and renders the tool badge + citation/SQL panel from metadata events.

---

## 10. Frontend UI

```
┌─────────────────────────────────────────┐
│  EMB Global Assistant                   │
├─────────────────────────────────────────┤
│                                         │
│  You: What is the return window?        │
│                                         │
│  Assistant:              [RAG]          │
│  The return window is 30 days from      │
│  the date of delivery.                  │
│  📄 returns_policy.pdf · page 1         │
│                                         │
│  You: How many orders are pending?      │
│                                         │
│  Assistant:              [SQL]          │
│  There are 23 pending orders.           │
│  🔍 SELECT COUNT(*) FROM orders         │
│      WHERE status = 'pending'           │
│                                         │
└─────────────────────────────────────────┤
│  [Type your message...        ] [Send]  │
└─────────────────────────────────────────┘
```

Each message bubble shows:
- Tool badge: `[RAG]`, `[SQL]`, or `[RAG + SQL]`
- Citation panel (RAG): source filename + page
- SQL panel (SQL): exact query that executed
- Tokens stream in as they arrive

---

## 11. LangSmith Tracing

Every tool function and LLM call is decorated with `@traceable`. Each conversation turn produces a trace showing:
- Which tool(s) were called and with what input
- Tool output
- Final LLM response
- Latency and token usage per step

---

## 12. Known Limitations

- Render backend has ~30–50s cold start after inactivity (free tier spin-down)
- Routing depends entirely on LLM judgment — edge cases may route incorrectly
- Claude has no embeddings API; Google text-embedding-004 is always used for embeddings regardless of chat LLM
- Pinecone index must be manually created with dimension=768 before running ingest.py
- SQL execution is SELECT-only; no aggregation hints are given to the LLM beyond the schema

---

## 13. Deployment Checklist

- [ ] Create Pinecone index (`emb-chatbot`, dimension=768, cosine metric)
- [ ] Run `python backend/ingest.py` to index PDFs
- [ ] Deploy backend to Render (set all env vars)
- [ ] Deploy frontend to Vercel (set `NEXT_PUBLIC_API_URL` to Render backend URL)
- [ ] Verify live URL works end-to-end
- [ ] Push public GitHub repo with Dockerfile and README

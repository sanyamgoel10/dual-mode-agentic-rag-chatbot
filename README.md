# Dual-Mode Agentic RAG Chatbot

**Live demo:** [Vercel App](https://dual-mode-agentic-rag-chatbot.vercel.app/)

A production-grade chatbot that answers questions about a company's knowledge base using two distinct retrieval strategies, autonomously routing between them per question.

---

## Architecture

A single FastAPI backend runs an agent loop that calls the configured LLM with two tool definitions. The LLM autonomously decides which tool(s) to use per question — no hardcoded router. Tools return structured results; the LLM synthesizes a streamed final answer delivered via Server-Sent Events.

```
Browser (Next.js on Vercel)
    │  SSE stream
    ▼
FastAPI backend (Render)
    │
    ├── search_docs tool → Google text-embedding-004 → Pinecone
    └── query_orders tool → LLM-generated SQL → SQLite
```

**Tool routing:**
- `search_docs(query)` — embeds the query, searches Pinecone top-5, returns chunks with source + page for citations
- `query_orders(sql)` — accepts a SELECT statement written by the LLM, executes against SQLite, returns rows

The LLM reads the user question and decides which tool(s) to call based on tool descriptions in the system context. A system prompt guard prevents hallucinated columns or policy text. For out-of-scope questions, the LLM answers without calling any tool.

---

## Technology Decisions

| Layer | Choice | Reason |
|---|---|---|
| LLM | Configurable: OpenAI / Claude / Gemini | Each provider's native tool-calling API used directly — no framework |
| Embeddings | Google `text-embedding-004` (dim=768) | Free tier, high quality, consistent output dimension |
| Vector store | Pinecone | Managed, no infra, persists across deployments, free tier sufficient |
| Structured data | SQLite rebuilt from CSV on startup | Fixed 200-row dataset, no writes needed; self-contained in container |
| Streaming | FastAPI `StreamingResponse` + SSE | Token-level streaming as required |
| Tracing | LangSmith `@traceable` | Provider-agnostic observability without LangChain dependency |
| Backend deploy | Render (free web service) | Docker-native, free tier |
| Frontend deploy | Vercel | Purpose-built for Next.js, zero-config, no cold starts |

---

## Setup

### Prerequisites

- Pinecone account — create a free index named `emb-chatbot` (dimension=768, metric=cosine)
- Google API key — for `text-embedding-004` (always required) and optionally Gemini chat
- LLM API key — OpenAI, Anthropic, or Google depending on `LLM_PROVIDER`
- LangSmith account — free tier at langsmith.com

### One-time PDF ingestion (index documents into Pinecone)

```bash
cd backend
cp .env.example .env   # fill in your API keys
pip install -r requirements.txt
python ingest.py
```

This chunks all PDFs in `database/`, embeds them with Google `text-embedding-004`, and upserts to Pinecone. Run once — vectors persist. Re-run only if documents change.

### Local development

```bash
# Fill in API keys
cp backend/.env.example backend/.env

# Run with Docker Compose
docker compose up --build

# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
# Health:   http://localhost:8000/health
```

### Environment variables

See `backend/.env.example` for all variables. Key ones:

```bash
LLM_PROVIDER=openai          # openai | claude | gemini
LLM_MODEL=gpt-4o
OPENAI_API_KEY=...           # or ANTHROPIC_API_KEY / GOOGLE_API_KEY
GOOGLE_API_KEY=...           # always required (used for embeddings)
PINECONE_API_KEY=...
PINECONE_INDEX=emb-chatbot
LANGSMITH_API_KEY=...
CURRENT_DATE=2026-06-15      # fixed date for grading consistency
```

---

## Deployment

### Backend → Render

1. Push repo to GitHub (public)
2. Render → New Web Service → connect repo → root: `backend/` → Runtime: Docker
3. Set all env vars from `.env.example`
4. Set `ORDERS_CSV_PATH=/app/database/orders.csv` and `DATABASE_PATH=/app/data/orders.db`
5. Note the Render URL

### Frontend → Vercel

1. Vercel → New Project → import repo → root: `frontend/`
2. Set `NEXT_PUBLIC_API_URL=https://your-render-backend.onrender.com`
3. Deploy — note the Vercel URL

### Final step

Update `CORS_ORIGINS` on Render to include your Vercel URL, then redeploy backend.

---

## Test Questions

| Type | Example | Expected |
|---|---|---|
| Document | "What is the return window?" | `[RAG]` badge, citation from `returns_policy.pdf` |
| Data | "How many orders are pending?" | `[SQL]` badge, SELECT query shown |
| Mixed | "Does our policy allow returns on order ORD-1001?" | `[RAG + SQL]` badge |
| Out-of-scope | "What is the weather in Mumbai?" | "I don't have that information." |

---

## Known Limitations

- Render free tier spins down after 15 min inactivity — first request has ~30–50s cold start
- Routing relies entirely on LLM judgment — edge-case questions may route to the wrong tool
- Conversation history stores text only — tool call details from previous turns are not replayed to the LLM
- Claude has no embeddings API; Google `text-embedding-004` is always used for embeddings regardless of chat LLM
- Pinecone index must be created manually with dimension=768 before running `ingest.py`

# Conversational AI Assistant

A production-ready customer support chatbot built with **FastAPI**, **LangChain**, **OpenAI**, and **FAISS**. The system supports multi-turn conversations, two-stage intent detection, session memory, and RAG-powered knowledge retrieval from a custom knowledge base.

---

## Problem Statement

Rule-based chatbots break down when user queries are phrased ambiguously or fall outside predefined patterns. Embedding a large language model end-to-end is expensive and slow. This project addresses that tradeoff by combining lightweight keyword-based intent detection with selective LLM calls, and grounding responses in a domain-specific knowledge base via retrieval-augmented generation.

---

## Solution Approach

The system routes each incoming message through a two-stage pipeline. A fast keyword scanner handles obvious cases (greetings, farewells, complaints) at near-zero cost. Only ambiguous messages trigger an LLM classifier. Relevant knowledge is retrieved from a FAISS vector index before generation, ensuring answers are grounded rather than hallucinated. Conversation history is held in-memory per session and asynchronously persisted to SQLite.

---

## Features

- **Two-stage intent detection** — keyword scan first, LLM JSON classifier as fallback across nine intent classes
- **RAG knowledge retrieval** — FAISS `IndexFlatIP` cosine similarity search over ingested documents
- **Multi-turn session memory** — per-session `ChatMessageHistory` with configurable turn-count TTL
- **Clarification probing** — LLM detects when a question is too ambiguous and requests clarification before answering
- **Document ingestion** — accepts raw text strings or uploaded `.txt` files at runtime
- **Conversation logging** — all exchanges persisted to SQLite via SQLAlchemy Core
- **LLM-as-judge evaluation** — `evaluator.py` scores answer quality against ground truth on a 0–1 rubric
- **Streamlit frontend** — chat UI with sidebar document upload and session controls
- **Docker Compose deployment** — single-command containerised setup
- **REST API** — full OpenAPI documentation at `/docs`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| LLM & Embeddings | OpenAI GPT-4o-mini + text-embedding-3-small (via LangChain) |
| Vector search | FAISS (faiss-cpu) |
| Session memory | LangChain `ChatMessageHistory` |
| Database | SQLite via SQLAlchemy Core |
| Configuration | Pydantic Settings |
| Frontend | Streamlit |
| Containerisation | Docker + Docker Compose |
| Testing | pytest + pytest-asyncio + httpx |

---

## Architecture

```
CLIENT LAYER
  Streamlit UI (frontend/app.py)  ·  REST Client  ·  curl / Postman
        │  HTTP JSON
        ▼
FASTAPI BACKEND  (backend/)
  POST /api/v1/chat/message          GET  /api/v1/sessions/{id}
  GET  /api/v1/chat/history/{id}     DELETE /api/v1/sessions/{id}
  POST /api/v1/ingest/text           GET  /api/v1/sessions
  POST /api/v1/ingest/file           GET  /api/v1/ingest/status
  GET  /health

  ChatOrchestrator (orchestrator.py)
    1. Load / create SessionState via SessionMemoryManager
    2. IntentDetector — keyword scan → LLM classifier (if needed)
    3. Clarification check — LLM JSON probe (optional)
    4. RAGPipeline.retrieve — FAISS top-k similarity search
    5. PromptTemplates.build_system_message — base + RAG + intent overlay
    6. LLMService.chat — OpenAI ChatCompletion with retry
    7. SessionState.add_* — append to in-memory history
    8. log_conversation — persist to SQLite
    9. Return ChatResponse (reply, intent, sources, metrics)

DATA LAYER  (data/)
  data/vector_store/    — FAISS index + chunks.pkl (auto-created on first ingest)
  data/conversations.db — SQLite conversation log
  data/documents/       — raw knowledge-base .txt files
```

---

## Project Structure

```
Conversational AI Assistant/
├── backend/
│   ├── __init__.py
│   ├── main.py               # FastAPI app factory and lifespan
│   ├── config.py             # Pydantic Settings (reads .env)
│   ├── models.py             # Pydantic request / response schemas
│   ├── orchestrator.py       # Central request handler
│   ├── intent_detector.py    # Two-stage intent classification
│   ├── intents.py            # Intent definitions and keyword sets
│   ├── memory.py             # Session memory (LangChain + TTL)
│   ├── rag_pipeline.py       # Ingest, embed, FAISS index, retrieve
│   ├── llm_service.py        # OpenAI wrapper with retry logic
│   ├── prompt_templates.py   # All prompts and template builders
│   ├── database.py           # SQLite conversation logger
│   ├── evaluator.py          # Latency stats and LLM-as-judge scoring
│   └── routes/
│       ├── chat.py           # POST /chat/message, GET /chat/history
│       ├── ingestion.py      # POST /ingest/text|file, GET /ingest/status
│       └── sessions.py       # GET / DELETE /sessions/{id}
├── frontend/
│   └── app.py                # Streamlit chat UI
├── data/
│   ├── documents/
│   │   ├── product_faq.txt
│   │   └── troubleshooting.txt
│   └── vector_store/         # Auto-created on first ingest
├── tests/
│   ├── test_chat.py
│   └── test_rag.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- An OpenAI API key

### 1. Clone the repository

```bash
git clone <repo-url>
cd "Conversational AI Assistant"
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root with the following:

```env
OPENAI_API_KEY=sk-...

# Optional — defaults shown
APP_TITLE=Conversational AI Assistant
APP_VERSION=1.0.0
DEBUG=false
MAX_HISTORY_TURNS=10
TOP_K_RETRIEVAL=3
DB_URL=sqlite:///data/conversations.db
```

---

## How to Run

### Start the backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Ingest the sample knowledge base

```bash
curl -X POST http://localhost:8000/api/v1/ingest/text \
  -H "Content-Type: application/json" \
  -d "{\"content\": \"$(cat data/documents/product_faq.txt)\", \"source_name\": \"product_faq.txt\"}"
```

Or upload files directly via the Streamlit sidebar.

### Start the Streamlit frontend

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Run with Docker Compose

```bash
cd docker
OPENAI_API_KEY=sk-... docker-compose up --build
```

### Run tests

```bash
pytest tests/ -v
```

> All OpenAI calls are mocked in the test suite — no API key required.

---

## Example Usage

**Send a message:**

```bash
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your refund policy?", "session_id": "user-123"}'
```

**Example response:**

```json
{
  "session_id": "user-123",
  "response": "You can request a full refund within 30 days of purchase...",
  "intent": "faq",
  "status": "success",
  "sources": [
    { "content": "...", "source": "product_faq.txt", "relevance_score": 0.94 }
  ],
  "clarification_question": null,
  "suggested_actions": ["Browse the help centre"],
  "metrics": {
    "latency_ms": 820,
    "retrieved_docs": 2,
    "intent_confidence": 0.88,
    "has_rag_context": true
  }
}
```

---

## Key Engineering Decisions

**Two-stage intent detection** keeps inference costs low. Keyword scanning handles the majority of routine messages instantly. Only genuinely ambiguous inputs invoke an LLM classifier, which returns structured JSON for reliable downstream routing.

**Cosine similarity via FAISS `IndexFlatIP`** is achieved by L2-normalising all embeddings before insertion so that inner-product search equals cosine similarity. Exact (non-approximate) search is appropriate at knowledge-base sizes up to approximately 100,000 chunks.

**LangChain `ChatMessageHistory`** is used for in-memory session state because it is natively compatible with LangChain's LCEL chain composition, making future migration to a Redis- or DynamoDB-backed store straightforward.

**SQLite for conversation logging** eliminates infrastructure dependencies in development. Switching to PostgreSQL requires only updating `DB_URL` in `.env`.

**LLM-as-judge evaluation** in `evaluator.py` scores answer quality against ground truth by prompting the model with a 0–1 rubric. This enables offline accuracy tracking without a labelled test set.

---

## Limitations

- The FAISS index is held in memory and persisted to disk on write. It is reloaded in full at startup; very large indexes (> 1 million chunks) should migrate to `IndexIVFFlat` or a managed vector database.
- Session memory is in-process. Restarting the server clears all active sessions. A Redis-backed memory store is required for persistence across restarts.
- Concurrent file uploads are not protected by a write lock. Production deployments with multiple workers should add a lock around `RAGPipeline.ingest()`.
- Intent detection accuracy depends on the quality of keyword sets in `intents.py` and degrades on highly domain-specific vocabulary not covered by the keyword list.

---

## Future Work

- Redis-backed session memory for multi-worker and cross-restart persistence
- Support for PDF and Markdown ingestion formats
- Streaming responses via Server-Sent Events for reduced perceived latency
- A/B testing framework to compare intent detection strategies
- User authentication and per-user session isolation

---

## Screenshots

> _Add screenshots of the Streamlit chat UI and the `/docs` API explorer here._

---

## Author

**Samik Hafeez** — BSc Computer Science Portfolio Project  
This project demonstrates production API design, retrieval-augmented generation, and multi-turn conversational AI applied to a customer support use case.

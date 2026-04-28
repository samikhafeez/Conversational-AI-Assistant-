"""
main.py — FastAPI application entry point.

Start with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.database import init_db
from backend.models import HealthResponse
from backend.memory import SessionMemoryManager
from backend.rag_pipeline import RAGPipeline

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Shared singletons (injected into routes via app.state) ────────────────────
_memory_manager: SessionMemoryManager | None = None
_rag_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up / shut-down lifecycle."""
    global _memory_manager, _rag_pipeline

    logger.info("🚀  Starting %s v%s", settings.app_title, settings.app_version)

    # Initialise SQLite conversation log
    init_db()
    logger.info("✅  Database initialised")

    # Session memory manager
    _memory_manager = SessionMemoryManager()
    app.state.memory_manager = _memory_manager
    logger.info("✅  Session memory manager ready")

    # RAG pipeline (loads existing FAISS index if available)
    _rag_pipeline = RAGPipeline()
    app.state.rag_pipeline = _rag_pipeline
    logger.info("✅  RAG pipeline ready  (index loaded: %s)", _rag_pipeline.is_loaded)

    yield  # ── application runs here ──────────────────────────────────────────

    logger.info("🛑  Shutting down — persisting vector store …")
    _rag_pipeline.persist()
    logger.info("👋  Shutdown complete")


# ── Application factory ───────────────────────────────────────────────────────
def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description=(
            "Conversational AI Assistant with multi-turn memory, "
            "intent detection, and RAG-powered knowledge retrieval."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    from backend.routes.chat import router as chat_router
    from backend.routes.ingestion import router as ingest_router
    from backend.routes.sessions import router as session_router

    application.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])
    application.include_router(ingest_router, prefix="/api/v1/ingest", tags=["Ingestion"])
    application.include_router(session_router, prefix="/api/v1/sessions", tags=["Sessions"])

    # ── Health endpoint ───────────────────────────────────────────────────────
    @application.get("/health", response_model=HealthResponse, tags=["System"])
    async def health():
        rag: RAGPipeline = application.state.rag_pipeline
        mem: SessionMemoryManager = application.state.memory_manager
        return HealthResponse(
            status="ok",
            version=settings.app_version,
            vector_store_loaded=rag.is_loaded,
            active_sessions=mem.active_session_count,
        )

    @application.get("/", tags=["System"])
    async def root():
        return JSONResponse({"message": f"Welcome to {settings.app_title}. Visit /docs for API reference."})

    return application


app = create_app()

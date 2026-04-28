"""
config.py — Centralised settings loaded from environment variables.

All configurable values live here. Import `settings` wherever needed.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────────────────
    openai_api_key: str = "your-openai-api-key-here"
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ── LLM generation params ─────────────────────────────────────────────
    temperature: float = 0.3
    max_tokens: int = 1024
    top_p: float = 1.0

    # ── Session / Memory ─────────────────────────────────────────────────
    session_ttl_seconds: int = 3600          # 1 hour inactivity timeout
    max_history_turns: int = 10             # keep last N human/AI pairs

    # ── RAG / FAISS ───────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 4                # docs returned per query
    vector_store_path: str = "data/vector_store"
    documents_path: str = "data/documents"

    # ── Intent detection ─────────────────────────────────────────────────
    intent_confidence_threshold: float = 0.5

    # ── Database (SQLite conversation log) ────────────────────────────────
    db_url: str = "sqlite:///./data/conversations.db"

    # ── Evaluation ───────────────────────────────────────────────────────
    latency_warn_threshold_ms: float = 3000.0

    # ── App ───────────────────────────────────────────────────────────────
    app_title: str = "Conversational AI Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton of Settings."""
    return Settings()


# Module-level convenience alias
settings = get_settings()

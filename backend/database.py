"""
database.py — SQLite conversation logger using SQLAlchemy Core (no ORM).

Tables:
  conversations  — one row per message exchange
  ingestion_log  — one row per document ingested into the vector store

Why SQLite?  Zero-dependency, single-file, easy to export / inspect.
Swap the `db_url` in config.py to PostgreSQL for production.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
)

from backend.config import settings

logger = logging.getLogger(__name__)

# ── Engine & metadata ─────────────────────────────────────────────────────────

_engine = create_engine(
    settings.db_url,
    connect_args={"check_same_thread": False},   # required for SQLite + threading
    echo=settings.debug,
)
_meta = MetaData()

# ── Table definitions ─────────────────────────────────────────────────────────

conversations = Table(
    "conversations",
    _meta,
    Column("id",            Integer, primary_key=True, autoincrement=True),
    Column("session_id",    String(64),  nullable=False, index=True),
    Column("message_id",    String(64),  nullable=False, unique=True),
    Column("user_message",  Text,        nullable=False),
    Column("ai_response",   Text,        nullable=False),
    Column("intent",        String(32),  nullable=False),
    Column("latency_ms",    Float,       nullable=True),
    Column("created_at",    DateTime,    default=datetime.utcnow),
)

ingestion_log = Table(
    "ingestion_log",
    _meta,
    Column("id",            Integer, primary_key=True, autoincrement=True),
    Column("source_name",   String(256), nullable=False),
    Column("chunks_added",  Integer,     nullable=False),
    Column("ingested_at",   DateTime,    default=datetime.utcnow),
)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they do not already exist."""
    import os
    # Ensure the data directory exists for SQLite file
    db_path = settings.db_url.replace("sqlite:///", "")
    if db_path and db_path != ":memory:":
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _meta.create_all(_engine)
    logger.info("Database tables ready at %s", settings.db_url)


# ── Write helpers ─────────────────────────────────────────────────────────────

def log_conversation(
    session_id: str,
    message_id: str,
    user_message: str,
    ai_response: str,
    intent: str,
    latency_ms: float,
) -> None:
    with _engine.begin() as conn:
        conn.execute(insert(conversations).values(
            session_id=session_id,
            message_id=message_id,
            user_message=user_message,
            ai_response=ai_response,
            intent=intent,
            latency_ms=latency_ms,
            created_at=datetime.utcnow(),
        ))


def log_ingestion(source_name: str, chunks_added: int) -> None:
    with _engine.begin() as conn:
        conn.execute(insert(ingestion_log).values(
            source_name=source_name,
            chunks_added=chunks_added,
            ingested_at=datetime.utcnow(),
        ))


# ── Read helpers ──────────────────────────────────────────────────────────────

def get_session_conversations(session_id: str) -> list[dict]:
    with _engine.connect() as conn:
        rows = conn.execute(
            select(conversations)
            .where(conversations.c.session_id == session_id)
            .order_by(conversations.c.created_at)
        ).fetchall()
    return [row._asdict() for row in rows]


def get_all_conversations(limit: int = 100) -> list[dict]:
    with _engine.connect() as conn:
        rows = conn.execute(
            select(conversations).order_by(conversations.c.created_at.desc()).limit(limit)
        ).fetchall()
    return [row._asdict() for row in rows]

"""
models.py — Pydantic request/response schemas for the API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class IntentType(str, Enum):
    """High-level user intent categories."""
    GREETING        = "greeting"
    FAREWELL        = "farewell"
    FAQ             = "faq"
    PRODUCT_INQUIRY = "product_inquiry"
    COMPLAINT       = "complaint"
    SUPPORT         = "support"
    SMALLTALK       = "smalltalk"
    ESCALATION      = "escalation"
    CLARIFICATION   = "clarification"
    UNKNOWN         = "unknown"


class ResponseStatus(str, Enum):
    SUCCESS       = "success"
    FALLBACK      = "fallback"
    CLARIFICATION = "clarification"
    ERROR         = "error"


# ─────────────────────────────────────────────────────────────────────────────
# Chat request / response
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096, description="User message")
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique session identifier; create one client-side and reuse it.",
    )
    user_id: str | None = Field(None, description="Optional authenticated user ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional extra context")


class SourceDocument(BaseModel):
    """A retrieved knowledge-base chunk surfaced to the client."""
    content: str
    source: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class EvaluationMetrics(BaseModel):
    """Lightweight per-response evaluation snapshot."""
    latency_ms: float
    retrieved_docs: int
    intent_confidence: float
    has_rag_context: bool


class ChatResponse(BaseModel):
    session_id: str
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    response: str
    intent: IntentType
    status: ResponseStatus
    sources: list[SourceDocument] = Field(default_factory=list)
    clarification_question: str | None = None
    suggested_actions: list[str] = Field(default_factory=list)
    metrics: EvaluationMetrics | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Document ingestion
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Ingest raw text into the knowledge base."""
    content: str = Field(..., min_length=10, description="Raw document text")
    source_name: str = Field(..., description="Human-readable document identifier")
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str
    source_name: str
    chunks_added: int
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────────────────────────────────────

class SessionInfo(BaseModel):
    session_id: str
    turn_count: int
    created_at: datetime
    last_active: datetime
    intents_seen: list[str] = Field(default_factory=list)


class SessionHistoryItem(BaseModel):
    role: str        # "human" | "ai"
    content: str
    timestamp: datetime


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: list[SessionHistoryItem]


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_loaded: bool
    active_sessions: int

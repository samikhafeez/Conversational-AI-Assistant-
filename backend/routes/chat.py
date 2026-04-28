"""
routes/chat.py — /api/v1/chat endpoints.

POST /api/v1/chat/message   — send a message, get a structured response
GET  /api/v1/chat/history   — retrieve logged turns for a session (from DB)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.database import get_session_conversations
from backend.memory import SessionMemoryManager
from backend.models import ChatRequest, ChatResponse
from backend.orchestrator import ChatOrchestrator
from backend.rag_pipeline import RAGPipeline

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Dependency helpers ────────────────────────────────────────────────────────

def _get_orchestrator(request: Request) -> ChatOrchestrator:
    """
    Build (or reuse) a ChatOrchestrator from the shared singletons stored on
    app.state.  This avoids re-creating LLM clients on every request.
    """
    if not hasattr(request.app.state, "_orchestrator"):
        mem: SessionMemoryManager = request.app.state.memory_manager
        rag: RAGPipeline          = request.app.state.rag_pipeline
        request.app.state._orchestrator = ChatOrchestrator(
            memory_manager=mem,
            rag_pipeline=rag,
        )
    return request.app.state._orchestrator


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/message",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a chat message",
    description=(
        "Submit a user message within a session. "
        "Include the same `session_id` on every turn to maintain conversation context."
    ),
)
async def send_message(
    body: ChatRequest,
    orchestrator: ChatOrchestrator = Depends(_get_orchestrator),
) -> ChatResponse:
    logger.info("POST /chat/message  session=%s  msg='%.60s…'", body.session_id[:8], body.message)
    try:
        return orchestrator.handle(body)
    except Exception as exc:
        logger.exception("Unhandled error in /chat/message: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        )


@router.get(
    "/history/{session_id}",
    summary="Retrieve conversation history for a session",
    description="Returns all logged turns for the given session from the persistent database.",
)
async def get_history(session_id: str) -> dict:
    rows = get_session_conversations(session_id)
    if not rows:
        return {"session_id": session_id, "turns": [], "message": "No history found for this session."}
    # Serialise datetime objects
    turns = []
    for row in rows:
        turns.append({
            "message_id":   row["message_id"],
            "user_message": row["user_message"],
            "ai_response":  row["ai_response"],
            "intent":       row["intent"],
            "latency_ms":   row["latency_ms"],
            "created_at":   row["created_at"].isoformat() if row["created_at"] else None,
        })
    return {"session_id": session_id, "turns": turns}

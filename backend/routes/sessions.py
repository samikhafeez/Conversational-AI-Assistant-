"""
routes/sessions.py — /api/v1/sessions endpoints.

GET    /api/v1/sessions/{session_id}          — session metadata
GET    /api/v1/sessions/{session_id}/history  — in-memory turn history
DELETE /api/v1/sessions/{session_id}          — clear session memory
GET    /api/v1/sessions                       — active session count
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, Response, status

from backend.memory import SessionMemoryManager
from backend.models import SessionHistoryItem, SessionHistoryResponse, SessionInfo

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_memory(request: Request) -> SessionMemoryManager:
    return request.app.state.memory_manager


@router.get(
    "/{session_id}",
    response_model=SessionInfo,
    summary="Get session metadata",
)
async def get_session(session_id: str, request: Request) -> SessionInfo:
    mem = _get_memory(request)
    with mem._lock:
        state = mem._sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return SessionInfo(
        session_id=state.session_id,
        turn_count=state.turn_count,
        created_at=datetime.utcfromtimestamp(state.created_at),
        last_active=datetime.utcfromtimestamp(state.last_active),
        intents_seen=state.intents_seen,
    )


@router.get(
    "/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get in-memory conversation history",
)
async def get_session_history(session_id: str, request: Request) -> SessionHistoryResponse:
    mem = _get_memory(request)
    raw = mem.get_history(session_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    items = [
        SessionHistoryItem(
            role=h["role"],
            content=h["content"],
            timestamp=datetime.fromisoformat(h["timestamp"]),
        )
        for h in raw
    ]
    return SessionHistoryResponse(session_id=session_id, history=items)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Clear session memory",
)
async def delete_session(session_id: str, request: Request) -> Response:
    mem = _get_memory(request)
    deleted = mem.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "",
    summary="List active sessions",
)
async def list_sessions(request: Request) -> dict:
    mem = _get_memory(request)
    with mem._lock:
        sessions = [
            {
                "session_id": sid,
                "turn_count": state.turn_count,
                "last_active": datetime.utcfromtimestamp(state.last_active).isoformat(),
            }
            for sid, state in mem._sessions.items()
        ]
    return {"active_sessions": len(sessions), "sessions": sessions}
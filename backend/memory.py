"""
memory.py — Session-scoped conversation memory.

Each session stores an ordered list of (role, content, timestamp) tuples.
The SessionMemoryManager maintains a dict of sessions and handles TTL expiry.

LangChain's ChatMessageHistory is used internally so the history is natively
compatible with LangChain's ConversationChain / LCEL pipelines.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from dataclasses import dataclass, field

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    session_id: str
    memory: ChatMessageHistory = field(default_factory=ChatMessageHistory)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    intents_seen: list[str] = field(default_factory=list)

    # ── helpers ──────────────────────────────────────────────────────────────

    def add_human(self, content: str) -> None:
        self.memory.add_user_message(content)
        self.last_active = time.time()

    def add_ai(self, content: str) -> None:
        self.memory.add_ai_message(content)
        self.last_active = time.time()

    def record_intent(self, intent: str) -> None:
        if not self.intents_seen or self.intents_seen[-1] != intent:
            self.intents_seen.append(intent)

    @property
    def turn_count(self) -> int:
        """Number of complete human+AI exchange pairs."""
        return sum(1 for m in self.memory.messages if isinstance(m, HumanMessage))

    @property
    def messages(self) -> list[BaseMessage]:
        return self.memory.messages

    def trimmed_messages(self, max_turns: int | None = None) -> list[BaseMessage]:
        """Return the last *max_turns* pairs to keep the context window bounded."""
        limit = (max_turns or settings.max_history_turns) * 2   # pairs → messages
        return self.memory.messages[-limit:] if limit else self.memory.messages

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > settings.session_ttl_seconds


class SessionMemoryManager:
    """
    Thread-safe store of SessionState objects.

    A background daemon thread cleans up expired sessions every 5 minutes.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._start_cleanup_thread()
        logger.info("SessionMemoryManager started (TTL=%ds)", settings.session_ttl_seconds)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)
                logger.debug("New session created: %s", session_id)
            return self._sessions[session_id]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.debug("Session deleted: %s", session_id)
                return True
            return False

    def get_history(self, session_id: str) -> list[dict]:
        """Return serialisable history for the sessions API."""
        with self._lock:
            state = self._sessions.get(session_id)
            if not state:
                return []
            result = []
            for msg in state.messages:
                role = "human" if isinstance(msg, HumanMessage) else "ai"
                result.append({
                    "role": role,
                    "content": msg.content,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            return result

    @property
    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _cleanup_expired(self) -> None:
        with self._lock:
            expired = [sid for sid, state in self._sessions.items() if state.is_expired()]
            for sid in expired:
                del self._sessions[sid]
                logger.info("Expired session removed: %s", sid)

    def _start_cleanup_thread(self) -> None:
        def _loop():
            while True:
                time.sleep(300)   # every 5 minutes
                self._cleanup_expired()

        t = threading.Thread(target=_loop, daemon=True, name="session-cleanup")
        t.start()

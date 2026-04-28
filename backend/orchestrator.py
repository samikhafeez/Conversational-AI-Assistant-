"""
orchestrator.py — Central chat orchestrator.

This is the single entry point for every user message.  It:

  1. Loads / creates the session state (memory).
  2. Runs intent detection (keyword → LLM two-stage).
  3. Checks whether a clarifying question is needed.
  4. Retrieves relevant knowledge-base chunks (RAG).
  5. Builds the full prompt (system + history + RAG context).
  6. Calls the LLM and captures latency.
  7. Saves the exchange to session memory AND the conversation log (DB).
  8. Returns a fully structured ChatResponse.

All dependencies are injected via the constructor so the class is fully
testable without touching FastAPI or a real OpenAI key.
"""

from __future__ import annotations

import json
import logging
import time
from uuid import uuid4

from backend.config import settings
from backend.database import log_conversation
from backend.intent_detector import IntentDetector
from backend.llm_service import LLMService
from backend.memory import SessionMemoryManager, SessionState
from backend.models import (
    ChatRequest,
    ChatResponse,
    EvaluationMetrics,
    IntentType,
    ResponseStatus,
    SourceDocument,
)
from backend.prompt_templates import (
    CLARIFICATION_DETECTION_PROMPT,
    FALLBACK_RESPONSE,
    SUGGESTED_ACTIONS,
    build_rag_context_string,
    build_system_message,
)
from backend.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """
    Stateless orchestrator — session state lives in SessionMemoryManager.
    Instantiate once at app startup and share across requests.
    """

    def __init__(
        self,
        llm: LLMService | None = None,
        memory_manager: SessionMemoryManager | None = None,
        rag_pipeline: RAGPipeline | None = None,
    ) -> None:
        self._llm = llm or LLMService()
        self._memory = memory_manager or SessionMemoryManager()
        self._rag = rag_pipeline or RAGPipeline(llm=self._llm)
        self._intent_detector = IntentDetector(llm=self._llm)
        logger.info("ChatOrchestrator ready")

    # ── Main entry point ──────────────────────────────────────────────────────

    def handle(self, request: ChatRequest) -> ChatResponse:
        """Process one user turn and return a structured response."""
        t_start = time.perf_counter()
        message_id = str(uuid4())

        # 1. Session memory
        session: SessionState = self._memory.get_or_create(request.session_id)

        # 2. Intent detection
        intent_result = self._intent_detector.detect(
            request.message,
            session_intents=session.intents_seen,
        )
        session.record_intent(intent_result.intent.value)
        logger.info(
            "[%s] intent=%s conf=%.2f method=%s",
            request.session_id[:8],
            intent_result.intent.value,
            intent_result.confidence,
            intent_result.method,
        )

        # 3. Clarification check (skip for trivial intents)
        clarification_q: str | None = None
        status = ResponseStatus.SUCCESS

        if intent_result.intent in (IntentType.UNKNOWN, IntentType.SUPPORT, IntentType.FAQ):
            clarification_q = self._maybe_clarify(request.message)
            if clarification_q:
                status = ResponseStatus.CLARIFICATION

        # 4. RAG retrieval
        retrieved_docs: list[dict] = []
        if intent_result.intent not in (IntentType.GREETING, IntentType.FAREWELL, IntentType.SMALLTALK):
            retrieved_docs = self._rag.retrieve(request.message)

        # 5. Build prompt
        rag_context_str = build_rag_context_string(retrieved_docs)
        system_msg = build_system_message(intent_result.intent, rag_context_str or None)

        history_msgs = [
            {"role": "user" if m.type == "human" else "assistant", "content": m.content}
            for m in session.trimmed_messages()
        ]

        messages = [
            {"role": "system", "content": system_msg},
            *history_msgs,
            {"role": "user", "content": request.message},
        ]

        # 6. LLM call
        ai_response = self._call_llm_safe(messages)

        # 7. Persist to session memory
        session.add_human(request.message)
        session.add_ai(ai_response)

        # 8. Latency & metrics
        latency_ms = (time.perf_counter() - t_start) * 1000
        if latency_ms > settings.latency_warn_threshold_ms:
            logger.warning("Slow response: %.0f ms for session %s", latency_ms, request.session_id[:8])

        metrics = EvaluationMetrics(
            latency_ms=round(latency_ms, 2),
            retrieved_docs=len(retrieved_docs),
            intent_confidence=round(intent_result.confidence, 3),
            has_rag_context=bool(retrieved_docs),
        )

        # 9. Persist to conversation log DB
        try:
            log_conversation(
                session_id=request.session_id,
                message_id=message_id,
                user_message=request.message,
                ai_response=ai_response,
                intent=intent_result.intent.value,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            logger.warning("DB logging failed (non-fatal): %s", exc)

        # 10. Build structured response
        sources = [
            SourceDocument(
                content=d["content"][:300],   # trim for wire transfer
                source=d["source"],
                relevance_score=d["relevance_score"],
            )
            for d in retrieved_docs
        ]

        return ChatResponse(
            session_id=request.session_id,
            message_id=message_id,
            response=ai_response,
            intent=intent_result.intent,
            status=status,
            sources=sources,
            clarification_question=clarification_q if status == ResponseStatus.CLARIFICATION else None,
            suggested_actions=SUGGESTED_ACTIONS.get(intent_result.intent, []),
            metrics=metrics,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _call_llm_safe(self, messages: list[dict]) -> str:
        """Call the LLM; return fallback text on any exception."""
        try:
            return self._llm.chat(messages)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return FALLBACK_RESPONSE

    def _maybe_clarify(self, message: str) -> str | None:
        """
        Ask the LLM whether a clarifying question is needed.
        Returns the question string or None.
        """
        prompt = CLARIFICATION_DETECTION_PROMPT.format(message=message)
        try:
            raw = self._llm.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant that analyses ambiguity."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=128,
                response_format={"type": "json_object"},
            )
            data = json.loads(raw)
            if data.get("needs_clarification") and data.get("clarification_question"):
                return data["clarification_question"]
        except Exception as exc:
            logger.debug("Clarification check failed (non-fatal): %s", exc)
        return None

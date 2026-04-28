"""
tests/test_chat.py — Unit & integration tests for the chat pipeline.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from backend.config import settings
from backend.intent_detector import IntentDetector, IntentResult
from backend.memory import SessionMemoryManager
from backend.models import ChatRequest, IntentType, ResponseStatus
from backend.orchestrator import ChatOrchestrator
from backend.prompt_templates import build_rag_context_string, build_system_message


# ─────────────────────────────────────────────────────────────────────────────
# Intent detector tests (keyword stage — no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentDetectorKeywords:
    def setup_method(self):
        mock_llm = MagicMock()
        self.detector = IntentDetector(llm=mock_llm)

    def test_greeting_detected(self):
        result = self.detector._keyword_scan("Hello, can you help me?")
        assert result.intent == IntentType.GREETING
        assert result.confidence > 0

    def test_farewell_detected(self):
        result = self.detector._keyword_scan("Thanks, goodbye!")
        assert result.intent == IntentType.FAREWELL

    def test_complaint_detected(self):
        result = self.detector._keyword_scan("This is broken and I'm very frustrated!")
        assert result.intent == IntentType.COMPLAINT

    def test_support_detected(self):
        result = self.detector._keyword_scan("I can't reset my password, please help")
        assert result.intent == IntentType.SUPPORT

    def test_unknown_for_gibberish(self):
        result = self.detector._keyword_scan("xyzzy plugh frobozz")
        assert result.intent == IntentType.UNKNOWN
        assert result.confidence == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Memory manager tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionMemoryManager:
    def setup_method(self):
        self.manager = SessionMemoryManager()

    def test_creates_new_session(self):
        state = self.manager.get_or_create("sess-001")
        assert state.session_id == "sess-001"
        assert state.turn_count == 0

    def test_retrieves_existing_session(self):
        s1 = self.manager.get_or_create("sess-002")
        s1.add_human("hello")
        s2 = self.manager.get_or_create("sess-002")
        assert s2.turn_count == 1

    def test_delete_session(self):
        self.manager.get_or_create("sess-003")
        deleted = self.manager.delete("sess-003")
        assert deleted is True
        assert self.manager.active_session_count == 0

    def test_delete_nonexistent_returns_false(self):
        assert self.manager.delete("does-not-exist") is False

    def test_trimmed_messages_respects_limit(self):
        state = self.manager.get_or_create("sess-004")
        for i in range(15):
            state.add_human(f"msg {i}")
            state.add_ai(f"reply {i}")
        trimmed = state.trimmed_messages(max_turns=5)
        assert len(trimmed) <= 10   # 5 pairs × 2 messages


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptTemplates:
    def test_build_system_message_no_rag(self):
        msg = build_system_message(IntentType.GREETING)
        assert "customer-support" in msg.lower()
        assert "greeting" in msg.lower() or "warmly" in msg.lower()

    def test_build_system_message_with_rag(self):
        msg = build_system_message(IntentType.FAQ, rag_context="Some KB content here.")
        assert "Some KB content here." in msg
        assert "knowledge base" in msg.lower()

    def test_build_rag_context_string_empty(self):
        result = build_rag_context_string([])
        assert result == ""

    def test_build_rag_context_string_with_docs(self):
        docs = [
            {"content": "Our refund policy is 30 days.", "source": "faq.txt", "relevance_score": 0.92}
        ]
        result = build_rag_context_string(docs)
        assert "refund policy" in result
        assert "faq.txt" in result
        assert "92%" in result


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator tests (LLM mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestChatOrchestrator:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.chat.return_value = "Hello! How can I help you today?"
        self.mock_llm.embed.return_value = [0.1] * 1536
        self.mock_llm.embed_batch.return_value = [[0.1] * 1536]

        self.mock_rag = MagicMock()
        self.mock_rag.retrieve.return_value = []
        self.mock_rag.is_loaded = False

        self.memory = SessionMemoryManager()

        self.orchestrator = ChatOrchestrator(
            llm=self.mock_llm,
            memory_manager=self.memory,
            rag_pipeline=self.mock_rag,
        )

    def test_greeting_returns_success(self):
        req = ChatRequest(message="Hello!", session_id="test-001")
        resp = self.orchestrator.handle(req)
        assert resp.session_id == "test-001"
        assert resp.status == ResponseStatus.SUCCESS
        assert resp.intent == IntentType.GREETING
        assert len(resp.response) > 0

    def test_multi_turn_memory_persists(self):
        req1 = ChatRequest(message="Hello!", session_id="test-002")
        self.orchestrator.handle(req1)

        req2 = ChatRequest(message="I need help with my account.", session_id="test-002")
        self.orchestrator.handle(req2)

        state = self.memory.get_or_create("test-002")
        assert state.turn_count == 2

    def test_different_sessions_are_isolated(self):
        req_a = ChatRequest(message="Hello from session A", session_id="sess-A")
        req_b = ChatRequest(message="Hello from session B", session_id="sess-B")
        self.orchestrator.handle(req_a)
        self.orchestrator.handle(req_b)

        state_a = self.memory.get_or_create("sess-A")
        state_b = self.memory.get_or_create("sess-B")
        assert state_a.turn_count == 1
        assert state_b.turn_count == 1

    def test_fallback_on_llm_failure(self):
        self.mock_llm.chat.side_effect = Exception("API down")
        req = ChatRequest(message="What is your refund policy?", session_id="test-003")
        resp = self.orchestrator.handle(req)
        assert "sorry" in resp.response.lower() or "trouble" in resp.response.lower()

    def test_rag_sources_attached_when_docs_returned(self):
        self.mock_rag.retrieve.return_value = [
            {"content": "30-day refund policy.", "source": "faq.txt", "relevance_score": 0.9}
        ]
        self.mock_rag.is_loaded = True
        req = ChatRequest(message="What is your refund policy?", session_id="test-004")
        resp = self.orchestrator.handle(req)
        assert len(resp.sources) == 1
        assert resp.sources[0].source == "faq.txt"

    def test_metrics_always_present(self):
        req = ChatRequest(message="Hi", session_id="test-005")
        resp = self.orchestrator.handle(req)
        assert resp.metrics is not None
        assert resp.metrics.latency_ms >= 0

"""
tests/test_rag.py — Unit tests for the RAG pipeline.

All OpenAI calls are mocked so tests run offline.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from backend.rag_pipeline import RAGPipeline


def _make_rag(dim: int = 4) -> tuple[RAGPipeline, MagicMock]:
    """Return a RAGPipeline with a mocked LLMService that returns dim-dimensional vectors."""
    mock_llm = MagicMock()
    # Single embed → unit vector
    mock_llm.embed.return_value = [1.0 / dim**0.5] * dim
    # Batch embed → list of unit vectors
    mock_llm.embed_batch.side_effect = lambda texts: [[1.0 / dim**0.5] * dim for _ in texts]
    rag = RAGPipeline.__new__(RAGPipeline)
    rag._llm = mock_llm
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from backend.config import settings
    rag._splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    rag._chunks = []
    rag._index = None
    rag._dim = 0
    return rag, mock_llm


class TestRAGPipelineIngest:
    def test_ingest_produces_chunks(self):
        rag, _ = _make_rag()
        n = rag.ingest("This is a short document for testing purposes.", "test_doc")
        assert n >= 1
        assert rag.chunk_count == n
        assert rag.is_loaded

    def test_ingest_empty_returns_zero(self):
        rag, mock_llm = _make_rag()
        mock_llm.embed_batch.return_value = []
        n = rag.ingest("", "empty_source")
        assert n == 0

    def test_multiple_ingests_accumulate(self):
        rag, _ = _make_rag()
        n1 = rag.ingest("First document with enough text to produce at least one chunk.", "doc1")
        n2 = rag.ingest("Second document with sufficient content for testing.", "doc2")
        assert rag.chunk_count == n1 + n2

    def test_source_name_preserved(self):
        rag, _ = _make_rag()
        rag.ingest("Some content for the knowledge base.", "my_source.txt")
        assert rag._chunks[0].source == "my_source.txt"


class TestRAGPipelineRetrieval:
    def test_retrieve_returns_top_k(self):
        rag, _ = _make_rag()
        # Ingest two distinct chunks
        rag.ingest(
            "Refund policy: you can get a full refund within 30 days. " * 5,
            "faq.txt"
        )
        results = rag.retrieve("What is the refund policy?", top_k=1)
        assert len(results) >= 1

    def test_retrieve_empty_index_returns_empty(self):
        rag, _ = _make_rag()
        results = rag.retrieve("any query")
        assert results == []

    def test_retrieve_result_structure(self):
        rag, _ = _make_rag()
        rag.ingest("Customer support is available 24/7 via live chat.", "support.txt")
        results = rag.retrieve("how do I contact support?")
        assert len(results) >= 1
        r = results[0]
        assert "content" in r
        assert "source" in r
        assert "relevance_score" in r
        assert 0.0 <= r["relevance_score"] <= 1.0

    def test_relevance_scores_are_normalised(self):
        rag, _ = _make_rag()
        rag.ingest("Plan pricing: Starter $29, Professional $79, Enterprise custom.", "pricing.txt")
        results = rag.retrieve("how much does it cost?")
        for r in results:
            assert 0.0 <= r["relevance_score"] <= 1.0


class TestRAGNormalisation:
    def test_normalise_unit_vector(self):
        vec = np.array([[3.0, 4.0]], dtype=np.float32)   # norm = 5
        normed = RAGPipeline._normalise(vec)
        np.testing.assert_allclose(np.linalg.norm(normed, axis=1), [1.0], atol=1e-6)

    def test_normalise_zero_vector_no_crash(self):
        vec = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        normed = RAGPipeline._normalise(vec)
        assert not np.any(np.isnan(normed))

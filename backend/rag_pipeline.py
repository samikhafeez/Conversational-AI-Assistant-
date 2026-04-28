"""
rag_pipeline.py — Retrieval-Augmented Generation (RAG) pipeline.

Flow:
  1. Ingest raw text  →  RecursiveCharacterTextSplitter chunks
  2. Embed each chunk  →  OpenAI text-embedding-3-small
  3. Store in FAISS vector index (persisted to disk)
  4. At query time: embed query → cosine similarity → top-k chunks

The pipeline intentionally owns its own LLMService reference so it can be
used independently (e.g. from a CLI ingestion script).
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.config import settings
from backend.llm_service import LLMService

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Internal chunk record ─────────────────────────────────────────────────────

@dataclass
class Chunk:
    content: str
    source: str
    chunk_index: int
    metadata: dict


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Manages document ingestion and semantic retrieval.

    Persistence layout (settings.vector_store_path):
        faiss.index   — the FAISS flat L2 index
        chunks.pkl    — list[Chunk] in insertion order
    """

    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm: LLMService = llm or LLMService()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._chunks: list[Chunk] = []
        self._index: faiss.IndexFlatIP | None = None   # inner-product ≈ cosine on unit vecs
        self._dim: int = 0

        # Try to load an existing index from disk
        self._load_if_exists()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def ingest(self, content: str, source_name: str, metadata: dict | None = None) -> int:
        """
        Split *content* into chunks, embed them, and add to the FAISS index.

        Returns the number of chunks added.
        """
        metadata = metadata or {}
        raw_chunks = self._splitter.split_text(content)
        if not raw_chunks:
            logger.warning("ingest(): no chunks produced for source='%s'", source_name)
            return 0

        logger.info("Ingesting '%s': %d raw chunks …", source_name, len(raw_chunks))

        # Embed in one batched API call
        vectors = self._llm.embed_batch(raw_chunks)

        new_chunks: list[Chunk] = []
        new_vectors: list[list[float]] = []

        for i, (text, vec) in enumerate(zip(raw_chunks, vectors)):
            new_chunks.append(Chunk(
                content=text,
                source=source_name,
                chunk_index=i,
                metadata=metadata,
            ))
            new_vectors.append(vec)

        self._extend_index(new_vectors)
        self._chunks.extend(new_chunks)

        logger.info("Ingested %d chunks from '%s'  (total=%d)", len(new_chunks), source_name, len(self._chunks))
        return len(new_chunks)

    def ingest_file(self, file_path: str | Path) -> int:
        """Convenience wrapper: read a text file and ingest it."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        return self.ingest(content, source_name=path.name)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Return the top-k most relevant chunks for *query*.

        Each result dict has keys: content, source, relevance_score, chunk_index.
        Returns an empty list if no index has been loaded.
        """
        if not self.is_loaded:
            logger.debug("retrieve(): index is empty — skipping")
            return []

        k = top_k or settings.top_k_retrieval
        query_vec = self._llm.embed(query)
        q = self._normalise(np.array([query_vec], dtype=np.float32))

        scores, indices = self._index.search(q, min(k, len(self._chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:   # FAISS returns -1 for padding
                continue
            chunk = self._chunks[idx]
            results.append({
                "content": chunk.content,
                "source": chunk.source,
                "relevance_score": float(np.clip(score, 0.0, 1.0)),
                "chunk_index": chunk.chunk_index,
            })

        logger.debug("retrieve(): returned %d chunks for query='%.60s…'", len(results), query)
        return results

    def persist(self) -> None:
        """Save the FAISS index and chunk list to disk."""
        if not self.is_loaded:
            return
        store_path = Path(settings.vector_store_path)
        store_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(store_path / "faiss.index"))
        with open(store_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info("Vector store persisted (%d chunks) → %s", len(self._chunks), store_path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extend_index(self, vectors: list[list[float]]) -> None:
        """Add normalised vectors to the FAISS index, creating it if needed."""
        mat = np.array(vectors, dtype=np.float32)
        mat = self._normalise(mat)

        if self._index is None:
            self._dim = mat.shape[1]
            self._index = faiss.IndexFlatIP(self._dim)   # inner product on unit vecs = cosine
            logger.debug("Created new FAISS IndexFlatIP (dim=%d)", self._dim)

        self._index.add(mat)

    @staticmethod
    def _normalise(mat: np.ndarray) -> np.ndarray:
        """L2-normalise each row so inner-product equals cosine similarity."""
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        return mat / norms

    def _load_if_exists(self) -> None:
        store_path = Path(settings.vector_store_path)
        index_file = store_path / "faiss.index"
        chunks_file = store_path / "chunks.pkl"

        if index_file.exists() and chunks_file.exists():
            try:
                self._index = faiss.read_index(str(index_file))
                with open(chunks_file, "rb") as f:
                    self._chunks = pickle.load(f)
                self._dim = self._index.d
                logger.info("Loaded existing vector store: %d chunks (dim=%d)", len(self._chunks), self._dim)
            except Exception as exc:
                logger.warning("Could not load vector store: %s — starting fresh", exc)
                self._index = None
                self._chunks = []
        else:
            logger.info("No existing vector store found — will build on first ingest")

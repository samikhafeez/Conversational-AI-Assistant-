"""
evaluator.py — Response quality and latency evaluation.

Two evaluation modes:

  1. Real-time (per-response) — EvaluationMetrics already attached by the
     orchestrator; this module aggregates stats across a session or corpus.

  2. Offline (batch) — evaluate a list of (question, answer, ground_truth)
     triples using an LLM-as-judge approach for accuracy scoring.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.llm_service import LLMService

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TurnMetrics:
    message_id: str
    latency_ms: float
    intent: str
    has_rag_context: bool
    retrieved_docs: int
    intent_confidence: float


@dataclass
class SessionStats:
    session_id: str
    total_turns: int
    avg_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    slow_turns: int              # turns > latency_warn_threshold
    intent_distribution: dict[str, int] = field(default_factory=dict)
    rag_hit_rate: float = 0.0   # % turns with RAG context


@dataclass
class AccuracyResult:
    question: str
    answer: str
    ground_truth: str
    score: float          # 0.0 – 1.0
    reasoning: str


# ── Aggregation ───────────────────────────────────────────────────────────────

def compute_session_stats(session_id: str, turns: list[TurnMetrics]) -> SessionStats:
    """
    Aggregate per-turn metrics into session-level statistics.
    """
    if not turns:
        return SessionStats(
            session_id=session_id,
            total_turns=0,
            avg_latency_ms=0,
            p95_latency_ms=0,
            max_latency_ms=0,
            slow_turns=0,
        )

    from backend.config import settings

    latencies = [t.latency_ms for t in turns]
    sorted_lat = sorted(latencies)
    p95_idx = int(len(sorted_lat) * 0.95)

    intent_dist: dict[str, int] = {}
    for t in turns:
        intent_dist[t.intent] = intent_dist.get(t.intent, 0) + 1

    rag_hits = sum(1 for t in turns if t.has_rag_context)

    return SessionStats(
        session_id=session_id,
        total_turns=len(turns),
        avg_latency_ms=round(statistics.mean(latencies), 2),
        p95_latency_ms=round(sorted_lat[p95_idx], 2),
        max_latency_ms=round(max(latencies), 2),
        slow_turns=sum(1 for l in latencies if l > settings.latency_warn_threshold_ms),
        intent_distribution=intent_dist,
        rag_hit_rate=round(rag_hits / len(turns), 3),
    )


# ── LLM-as-judge accuracy evaluation ─────────────────────────────────────────

_ACCURACY_SYSTEM = """\
You are an expert evaluator for a customer-support AI assistant.
Score how well the AI answer addresses the question given the ground truth.

Scoring rubric:
  1.0 — Correct, complete, and concise.
  0.75 — Mostly correct with minor omissions or slight inaccuracies.
  0.5  — Partially correct; key information missing or somewhat misleading.
  0.25 — Mostly incorrect but contains some relevant information.
  0.0  — Completely wrong or irrelevant.

Respond with JSON only:
{{"score": <0.0-1.0>, "reasoning": "<one sentence>"}}
"""

_ACCURACY_USER = """\
Question: {question}
Ground truth: {ground_truth}
AI answer: {answer}
"""


def evaluate_accuracy(
    question: str,
    answer: str,
    ground_truth: str,
    llm: "LLMService",
) -> AccuracyResult:
    """
    Use an LLM judge to score a single (question, answer, ground_truth) triple.
    """
    prompt = _ACCURACY_USER.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
    )
    try:
        raw = llm.chat(
            [
                {"role": "system", "content": _ACCURACY_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        reasoning = data.get("reasoning", "")
    except Exception as exc:
        logger.warning("Accuracy evaluation failed: %s", exc)
        score = 0.0
        reasoning = f"Evaluation error: {exc}"

    return AccuracyResult(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        score=score,
        reasoning=reasoning,
    )


def evaluate_batch(
    samples: list[dict],   # each: {"question": ..., "answer": ..., "ground_truth": ...}
    llm: "LLMService",
) -> dict:
    """
    Run LLM-as-judge on a batch of samples.

    Returns a summary dict with individual results and aggregate stats.
    """
    results = [
        evaluate_accuracy(s["question"], s["answer"], s["ground_truth"], llm)
        for s in samples
    ]
    scores = [r.score for r in results]
    summary = {
        "total_samples": len(results),
        "average_score": round(statistics.mean(scores), 3) if scores else 0.0,
        "min_score":     round(min(scores), 3) if scores else 0.0,
        "max_score":     round(max(scores), 3) if scores else 0.0,
        "results": [
            {
                "question":    r.question,
                "score":       r.score,
                "reasoning":   r.reasoning,
            }
            for r in results
        ],
    }
    logger.info(
        "Batch evaluation complete: %d samples, avg_score=%.3f",
        len(results), summary["average_score"],
    )
    return summary

"""
intent_detector.py — Two-stage intent classification.

Stage 1: Fast keyword scan → returns a candidate + confidence.
Stage 2: LLM-based classifier if Stage 1 confidence is below threshold
         or the intent is UNKNOWN.

The LLM call uses the few-shot examples from intents.py and returns a
JSON object so it can be parsed reliably.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from backend.config import settings
from backend.intents import INTENT_DEFINITIONS, INTENT_MAP
from backend.llm_service import LLMService
from backend.models import IntentType

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    intent: IntentType
    confidence: float          # 0.0 – 1.0
    method: str                # "keyword" | "llm"


# ── Prompt template for LLM classification ────────────────────────────────────

_INTENT_SYSTEM_PROMPT = """\
You are an intent classification assistant for a customer-support chatbot.

Classify the user message into EXACTLY ONE of the following intent labels:
{intent_list}

Respond with a JSON object in this exact format:
{{"intent": "<label>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}

Do NOT include any other text outside the JSON object.
"""

_INTENT_FEW_SHOT: list[dict] = []

def _build_few_shot() -> list[dict]:
    """Build few-shot examples from INTENT_DEFINITIONS once."""
    examples: list[dict] = []
    for defn in INTENT_DEFINITIONS:
        for ex in defn.examples[:2]:   # max 2 examples per intent
            examples.append({"role": "user",      "content": ex})
            examples.append({"role": "assistant", "content": json.dumps({
                "intent": defn.intent.value,
                "confidence": 0.95,
                "reasoning": f"Clear {defn.intent.value} signal.",
            })})
    return examples


class IntentDetector:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm
        self._intent_labels = [d.intent.value for d in INTENT_DEFINITIONS]
        self._few_shot = _build_few_shot()
        logger.info("IntentDetector ready  (%d intents)", len(self._intent_labels))

    # ── Public entry point ────────────────────────────────────────────────────

    def detect(self, message: str, session_intents: list[str] | None = None) -> IntentResult:
        """
        Detect intent for *message*.

        session_intents: list of previous intent values for context-aware
                         disambiguation (e.g. CLARIFICATION follows SUPPORT).
        """
        # Stage 1: keyword scan
        keyword_result = self._keyword_scan(message)

        if keyword_result.confidence >= settings.intent_confidence_threshold:
            logger.debug("Intent '%s' via keyword (conf=%.2f)", keyword_result.intent.value, keyword_result.confidence)
            return keyword_result

        # Stage 2: LLM
        llm_result = self._llm_classify(message)
        logger.debug("Intent '%s' via LLM (conf=%.2f)", llm_result.intent.value, llm_result.confidence)
        return llm_result

    # ── Stage 1 – keyword scan ────────────────────────────────────────────────

    def _keyword_scan(self, message: str) -> IntentResult:
        lower = message.lower()
        best_intent = IntentType.UNKNOWN
        best_score: float = 0.0

        for defn in INTENT_DEFINITIONS:
            hits = sum(1 for kw in defn.keywords if kw in lower)
            if hits == 0:
                continue
            # Normalise: hits / total_keywords, capped at 1.0
            score = min(hits / len(defn.keywords) * 3.0, 1.0)
            if score > best_score:
                best_score = score
                best_intent = defn.intent

        return IntentResult(intent=best_intent, confidence=best_score, method="keyword")

    # ── Stage 2 – LLM classification ─────────────────────────────────────────

    def _llm_classify(self, message: str) -> IntentResult:
        system = _INTENT_SYSTEM_PROMPT.format(
            intent_list=", ".join(self._intent_labels)
        )
        messages = [
            {"role": "system", "content": system},
            *self._few_shot,
            {"role": "user", "content": message},
        ]

        try:
            raw = self._llm.chat(
                messages,
                temperature=0.0,
                max_tokens=128,
                response_format={"type": "json_object"},
            )
            data = json.loads(raw)
            intent_val = data.get("intent", IntentType.UNKNOWN.value)
            # Validate the returned label
            if intent_val not in {i.value for i in IntentType}:
                intent_val = IntentType.UNKNOWN.value
            confidence = float(data.get("confidence", 0.5))
            return IntentResult(
                intent=IntentType(intent_val),
                confidence=confidence,
                method="llm",
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("LLM intent parse error: %s", exc)
            return IntentResult(intent=IntentType.UNKNOWN, confidence=0.0, method="llm")

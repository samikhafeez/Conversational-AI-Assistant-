"""
llm_service.py — Thin wrapper around the OpenAI ChatCompletion API.

Provides:
  - LLMService.chat()         → single completion
  - LLMService.embed()        → text → embedding vector
  - Exponential-backoff retry on rate-limit / server errors
"""

from __future__ import annotations

import logging
import time
from typing import Any

import openai
from openai import OpenAI

from backend.config import settings

logger = logging.getLogger(__name__)

# Maximum retry attempts for transient API errors
_MAX_RETRIES = 3
_RETRY_DELAY_BASE = 1.0   # seconds; doubles on each attempt


class LLMService:
    """Singleton-style wrapper; instantiate once and reuse."""

    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.chat_model
        self._embed_model = settings.embedding_model
        logger.info("LLMService initialised  model=%s", self._model)

    # ── Chat completion ───────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """
        Call the ChatCompletion API with retry logic.

        Args:
            messages:        List of {"role": ..., "content": ...} dicts.
            temperature:     Override default temperature.
            max_tokens:      Override default max_tokens.
            response_format: e.g. {"type": "json_object"} for structured output.

        Returns:
            The assistant's reply as a plain string.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.temperature,
            "max_tokens": max_tokens if max_tokens is not None else settings.max_tokens,
            "top_p": settings.top_p,
        }
        if response_format:
            kwargs["response_format"] = response_format

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                logger.debug(
                    "chat() ok  tokens=%s  attempt=%d",
                    response.usage.total_tokens if response.usage else "?",
                    attempt,
                )
                return content
            except openai.RateLimitError as exc:
                wait = _RETRY_DELAY_BASE * (2 ** (attempt - 1))
                logger.warning("Rate-limit hit; retrying in %.1fs  (%d/%d)", wait, attempt, _MAX_RETRIES)
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(wait)
            except openai.APIStatusError as exc:
                logger.error("OpenAI API error: %s", exc)
                raise

    # ── Embeddings ───────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """
        Return the embedding vector for *text* using the configured model.
        Strips excessive whitespace before sending.
        """
        clean = " ".join(text.split())
        response = self._client.embeddings.create(
            model=self._embed_model,
            input=clean,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in a single API call (more efficient)."""
        cleaned = [" ".join(t.split()) for t in texts]
        response = self._client.embeddings.create(
            model=self._embed_model,
            input=cleaned,
        )
        # API returns items in order
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

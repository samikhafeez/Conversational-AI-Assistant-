"""
prompt_templates.py — All system and user prompt templates in one place.

Design principles:
  - Each template is a plain Python string with {placeholder} variables.
  - Templates are composed by the Orchestrator; no LLM calls happen here.
  - Keeping prompts centralised makes A/B testing and iteration fast.
"""

from __future__ import annotations

from backend.models import IntentType

# ─────────────────────────────────────────────────────────────────────────────
# Core system prompt (injected on every turn)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful, friendly, and professional customer-support assistant.

## Your capabilities
- Answer questions accurately using the knowledge base context provided.
- Remember what the user has said earlier in this conversation.
- Detect when you don't know something and say so clearly.
- Ask clarifying questions when the user's request is ambiguous.
- Escalate gracefully when the user wants to speak to a human.

## Rules
1. NEVER fabricate facts. If the answer isn't in the provided context, say you don't know.
2. Keep responses concise — 2-4 sentences for simple questions, longer only when needed.
3. Always be polite, even when the user is frustrated.
4. If you suggest an action the user should take, number the steps.
5. End every response on a helpful note — offer to assist further.

## Response format
Respond in plain, friendly language. Do NOT use markdown unless the user explicitly asks for it.
"""

# ─────────────────────────────────────────────────────────────────────────────
# RAG context injection
# ─────────────────────────────────────────────────────────────────────────────

RAG_CONTEXT_TEMPLATE = """\
## Relevant knowledge base excerpts
Use the following passages to answer the user's question.
If none of them are relevant, rely on your general knowledge and say so.

{context}

---
"""

# ─────────────────────────────────────────────────────────────────────────────
# Intent-specific prompt overlays
# ─────────────────────────────────────────────────────────────────────────────

INTENT_OVERLAYS: dict[IntentType, str] = {
    IntentType.GREETING: (
        "The user is greeting you. Respond warmly and briefly. "
        "Ask how you can help them today."
    ),
    IntentType.FAREWELL: (
        "The user is saying goodbye. Wish them well and let them know "
        "they can return any time they need help."
    ),
    IntentType.COMPLAINT: (
        "The user is expressing frustration or a complaint. "
        "Acknowledge their feelings first ('I understand this is frustrating…'), "
        "then work to resolve the issue. Do NOT be defensive."
    ),
    IntentType.ESCALATION: (
        "The user wants to speak to a human agent. "
        "Acknowledge their request, apologise for any inconvenience, "
        "and inform them that a human agent will follow up shortly. "
        "Collect their contact preference if not already known."
    ),
    IntentType.SUPPORT: (
        "The user needs technical or account assistance. "
        "Ask one focused clarifying question if the issue isn't clear, "
        "then provide step-by-step guidance."
    ),
    IntentType.FAQ: (
        "The user has a general question. "
        "Provide a clear, direct answer backed by the knowledge base context."
    ),
    IntentType.PRODUCT_INQUIRY: (
        "The user is asking about products, pricing, or features. "
        "Be specific and accurate. If you don't have pricing info, "
        "direct them to the website or a human agent."
    ),
    IntentType.SMALLTALK: (
        "The user is making small talk. Engage briefly and warmly, "
        "then gently redirect to how you can assist them."
    ),
    IntentType.UNKNOWN: (
        "The user's intent is unclear. "
        "Ask a single, focused clarifying question to understand what they need."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Fallback response (used when LLM call fails entirely)
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_RESPONSE = (
    "I'm sorry, I'm having trouble processing your request right now. "
    "Could you please try again in a moment? If the issue persists, "
    "I can connect you with a human agent."
)

# ─────────────────────────────────────────────────────────────────────────────
# Clarification detection prompt
# ─────────────────────────────────────────────────────────────────────────────

CLARIFICATION_DETECTION_PROMPT = """\
Given this user message, determine if it is ambiguous enough to require a
clarifying question before answering.

User message: "{message}"

Respond with a JSON object:
{{"needs_clarification": true/false, "clarification_question": "<question or null>"}}

Only set needs_clarification=true if the message has multiple plausible interpretations
and answering incorrectly would waste the user's time.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Suggested actions per intent
# ─────────────────────────────────────────────────────────────────────────────

SUGGESTED_ACTIONS: dict[IntentType, list[str]] = {
    IntentType.COMPLAINT:       ["Speak to a human agent", "View our support articles", "Submit a ticket"],
    IntentType.ESCALATION:      ["Connect to human agent", "Request a callback"],
    IntentType.SUPPORT:         ["Reset password", "View troubleshooting guide", "Contact support"],
    IntentType.PRODUCT_INQUIRY: ["View pricing page", "Start free trial", "Compare plans"],
    IntentType.FAQ:             ["Browse help center", "Watch tutorial videos"],
    IntentType.GREETING:        ["View FAQs", "Check your account", "Browse products"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Template builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_system_message(intent: IntentType, rag_context: str | None = None) -> str:
    """
    Compose the full system prompt for a given intent and optional RAG context.
    """
    parts = [SYSTEM_PROMPT]

    if rag_context:
        parts.append(RAG_CONTEXT_TEMPLATE.format(context=rag_context))

    overlay = INTENT_OVERLAYS.get(intent)
    if overlay:
        parts.append(f"## Current focus\n{overlay}")

    return "\n".join(parts)


def build_rag_context_string(retrieved_docs: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    if not retrieved_docs:
        return ""
    lines = []
    for i, doc in enumerate(retrieved_docs, 1):
        score_pct = int(doc["relevance_score"] * 100)
        lines.append(f"[{i}] Source: {doc['source']} (relevance: {score_pct}%)\n{doc['content']}")
    return "\n\n".join(lines)

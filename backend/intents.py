"""
intents.py — Intent definitions: keyword signals and example utterances
used by the IntentDetector for rule-assisted classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.models import IntentType


@dataclass
class IntentDefinition:
    intent: IntentType
    keywords: list[str]               # fast keyword scan (lower-case)
    examples: list[str]               # few-shot examples for LLM classifier


INTENT_DEFINITIONS: list[IntentDefinition] = [
    IntentDefinition(
        intent=IntentType.GREETING,
        keywords=["hello", "hi", "hey", "howdy", "good morning", "good afternoon",
                  "good evening", "greetings", "what's up", "sup"],
        examples=[
            "Hello there!",
            "Hi, can you help me?",
            "Good morning, I need assistance.",
        ],
    ),
    IntentDefinition(
        intent=IntentType.FAREWELL,
        keywords=["bye", "goodbye", "see you", "talk later", "take care",
                  "have a good day", "farewell", "later"],
        examples=[
            "Goodbye, thanks for your help!",
            "That's all, see you later.",
            "Thanks, bye!",
        ],
    ),
    IntentDefinition(
        intent=IntentType.FAQ,
        keywords=["what is", "how does", "explain", "tell me about", "what are",
                  "define", "faq", "question", "mean", "difference between"],
        examples=[
            "What is your refund policy?",
            "How does the subscription work?",
            "Can you explain the cancellation process?",
        ],
    ),
    IntentDefinition(
        intent=IntentType.PRODUCT_INQUIRY,
        keywords=["price", "cost", "buy", "purchase", "plan", "feature", "upgrade",
                  "downgrade", "available", "offer", "discount", "promo"],
        examples=[
            "What plans do you offer?",
            "How much does the premium subscription cost?",
            "Do you have any discounts for students?",
        ],
    ),
    IntentDefinition(
        intent=IntentType.COMPLAINT,
        keywords=["problem", "issue", "broken", "not working", "error", "bug",
                  "frustrated", "unhappy", "disappointed", "terrible", "worst",
                  "failed", "crash", "wrong"],
        examples=[
            "Your app keeps crashing and I'm really frustrated.",
            "I've been charged incorrectly, this is unacceptable.",
            "The feature is broken and nobody is helping me.",
        ],
    ),
    IntentDefinition(
        intent=IntentType.SUPPORT,
        keywords=["help", "support", "assist", "fix", "trouble", "can't", "cannot",
                  "unable", "stuck", "reset", "recover", "login", "password"],
        examples=[
            "I can't log into my account.",
            "Please help me reset my password.",
            "I need assistance setting up the integration.",
        ],
    ),
    IntentDefinition(
        intent=IntentType.ESCALATION,
        keywords=["speak to human", "talk to agent", "real person", "manager",
                  "supervisor", "escalate", "not satisfied", "transfer"],
        examples=[
            "I'd like to speak to a human agent please.",
            "Can you transfer me to a manager?",
            "I want to escalate this issue.",
        ],
    ),
    IntentDefinition(
        intent=IntentType.SMALLTALK,
        keywords=["how are you", "what's your name", "who are you", "tell me a joke",
                  "weather", "your favourite", "do you like"],
        examples=[
            "How are you doing today?",
            "What's your name?",
            "Can you tell me a joke?",
        ],
    ),
]

# Quick look-up by IntentType
INTENT_MAP: dict[str, IntentDefinition] = {
    defn.intent.value: defn for defn in INTENT_DEFINITIONS
}

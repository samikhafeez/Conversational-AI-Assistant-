"""
frontend/app.py — Streamlit chat interface.

Run with:
    streamlit run frontend/app.py

Environment variable: BACKEND_URL (default: http://localhost:8000)
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CHAT_ENDPOINT  = f"{BACKEND_URL}/api/v1/chat/message"
INGEST_ENDPOINT = f"{BACKEND_URL}/api/v1/ingest/text"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"

st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="centered",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": ..., "content": ..., "meta": ...}

if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

# ── Helper functions ──────────────────────────────────────────────────────────

def send_message(user_text: str) -> dict | None:
    try:
        resp = requests.post(
            CHAT_ENDPOINT,
            json={"message": user_text, "session_id": st.session_state.session_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot reach the backend. Make sure the FastAPI server is running.")
    except requests.exceptions.Timeout:
        st.error("⚠️ The backend took too long to respond.")
    except Exception as exc:
        st.error(f"⚠️ Unexpected error: {exc}")
    return None


def get_health() -> dict | None:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def ingest_text(content: str, source_name: str) -> dict | None:
    try:
        resp = requests.post(
            INGEST_ENDPOINT,
            json={"content": content, "source_name": source_name},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Ingestion error: {exc}")
    return None


def intent_badge(intent: str) -> str:
    colours = {
        "greeting": "🟢", "farewell": "🔵", "faq": "🟡",
        "product_inquiry": "🟠", "complaint": "🔴", "support": "🟣",
        "escalation": "⛔", "smalltalk": "🩵", "unknown": "⚪",
    }
    return colours.get(intent, "⚪") + f" `{intent}`"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 AI Support Assistant")
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    # Health check
    health = get_health()
    if health:
        st.success(f"✅ Backend online  v{health.get('version', '?')}")
        st.metric("Active sessions", health.get("active_sessions", 0))
        st.metric("KB loaded", "Yes" if health.get("vector_store_loaded") else "No")
    else:
        st.error("❌ Backend offline")

    st.divider()

    # Knowledge base ingestion
    st.subheader("📚 Add to Knowledge Base")
    kb_source = st.text_input("Source name", placeholder="product_faq_v2")
    kb_content = st.text_area("Paste document text", height=150,
                              placeholder="Enter text to add to the knowledge base…")
    if st.button("Ingest Document", use_container_width=True):
        if kb_content and kb_source:
            with st.spinner("Ingesting…"):
                result = ingest_text(kb_content, kb_source)
            if result:
                st.success(f"✅ {result['chunks_added']} chunks added from '{kb_source}'")
        else:
            st.warning("Please fill in both source name and content.")

    st.divider()

    # Reset session
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_metrics = None
        st.rerun()

    # Metrics panel
    if st.session_state.last_metrics:
        m = st.session_state.last_metrics
        st.subheader("📊 Last Response Metrics")
        st.metric("Latency", f"{m.get('latency_ms', 0):.0f} ms")
        st.metric("KB docs used", m.get("retrieved_docs", 0))
        st.metric("Intent confidence", f"{m.get('intent_confidence', 0):.0%}")

# ── Main chat area ────────────────────────────────────────────────────────────

st.header("💬 Customer Support Chat")

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Show metadata for assistant turns
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            cols = st.columns(3)
            cols[0].caption(intent_badge(meta.get("intent", "unknown")))
            cols[1].caption(f"⏱ {meta.get('latency_ms', 0):.0f} ms")
            cols[2].caption(f"📄 {meta.get('retrieved_docs', 0)} KB docs")

            # Sources
            if meta.get("sources"):
                with st.expander("📖 Knowledge base sources"):
                    for src in meta["sources"]:
                        st.markdown(f"**{src['source']}** (relevance: {src['relevance_score']:.0%})")
                        st.caption(src["content"][:250] + "…" if len(src["content"]) > 250 else src["content"])

            # Clarification question
            if meta.get("clarification_question"):
                st.info(f"💬 {meta['clarification_question']}")

            # Suggested actions
            if meta.get("suggested_actions"):
                st.caption("Quick actions: " + " · ".join(
                    f"`{a}`" for a in meta["suggested_actions"][:3]
                ))

# Chat input
if prompt := st.chat_input("Type your message…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            data = send_message(prompt)

        if data:
            ai_text = data.get("response", "Sorry, I could not process that.")
            st.write(ai_text)

            metrics = data.get("metrics", {})
            st.session_state.last_metrics = metrics

            meta = {
                "intent":              data.get("intent", "unknown"),
                "latency_ms":          metrics.get("latency_ms", 0),
                "retrieved_docs":      metrics.get("retrieved_docs", 0),
                "intent_confidence":   metrics.get("intent_confidence", 0),
                "sources":             data.get("sources", []),
                "clarification_question": data.get("clarification_question"),
                "suggested_actions":   data.get("suggested_actions", []),
            }

            cols = st.columns(3)
            cols[0].caption(intent_badge(meta["intent"]))
            cols[1].caption(f"⏱ {meta['latency_ms']:.0f} ms")
            cols[2].caption(f"📄 {meta['retrieved_docs']} KB docs")

            if meta["sources"]:
                with st.expander("📖 Knowledge base sources"):
                    for src in meta["sources"]:
                        st.markdown(f"**{src['source']}** (relevance: {src['relevance_score']:.0%})")
                        st.caption(src["content"][:250] + "…" if len(src["content"]) > 250 else src["content"])

            if meta["clarification_question"]:
                st.info(f"💬 {meta['clarification_question']}")

            if meta["suggested_actions"]:
                st.caption("Quick actions: " + " · ".join(
                    f"`{a}`" for a in meta["suggested_actions"][:3]
                ))

            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_text,
                "meta": meta,
            })
        else:
            fallback = "I'm sorry, something went wrong. Please try again."
            st.write(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})

"""Chat page: streamed, cited answers with conversation memory per browser session."""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from drive_agent.core.config import ConfigError
from drive_agent.graph.rag import Citation, Source
from drive_agent.interfaces.streamlit_app.resources import get_resources
from drive_agent.llm import LLMError

PREVIEW_CHARS = 700


def _md(text: str) -> str:
    """Streamlit renders $...$ as LaTeX; salaries and prices must stay literal."""
    return text.replace("$", "\\$")


def _location(citation: Citation | dict[str, Any]) -> str:
    if "page" in citation:
        return f"page {citation['page']}"
    if "slide" in citation:
        return f"slide {citation['slide']}"
    return str(citation.get("section", ""))


def render_citations(citations: list[Citation]) -> None:
    if not citations:
        return
    st.caption("Sources")
    for citation in sorted(citations, key=lambda c: c["number"]):
        with st.container(border=True):
            text, button = st.columns([6, 1], vertical_alignment="center")
            where = _location(citation)
            text.markdown(
                f"**[{citation['number']}] {_md(citation.get('file_name', '?'))}**"
                + (f" · {where}" if where else "")
            )
            if citation.get("path"):
                text.caption(citation["path"])
            if citation.get("web_url"):
                button.link_button("Open", citation["web_url"])


def render_sources(sources: list[Source]) -> None:
    if not sources:
        return
    with st.expander(f"Retrieved passages ({len(sources)})"):
        for n, source in enumerate(sources, 1):
            header, _, body = source["text"].partition("\n\n")
            st.markdown(f"**{n}. {_md(header)}**")
            st.text(body[:PREVIEW_CHARS] + ("…" if len(body) > PREVIEW_CHARS else ""))


def _render_turn(turn: dict[str, Any]) -> None:
    with st.chat_message(turn["role"]):
        st.markdown(_md(turn["content"]))
        if turn["role"] == "assistant":
            render_citations(turn.get("citations", []))
            render_sources(turn.get("sources", []))


def render_chat() -> None:
    st.title("Ask your Drive")
    resources = get_resources()
    if resources is None:
        return
    if resources.index.count() == 0:
        st.info("The index is empty. Open the **Index** page and run a sync first.")
        return

    state = st.session_state
    state.setdefault("thread_id", uuid.uuid4().hex[:12])
    state.setdefault("turns", [])
    with st.sidebar:
        st.caption(f"Conversation `{state.thread_id}`")
        if st.button("New conversation", use_container_width=True):
            state.thread_id = uuid.uuid4().hex[:12]
            state.turns = []
            st.rerun()

    for turn in state.turns:
        _render_turn(turn)

    question = st.chat_input("Ask about your Drive files")
    if not question:
        return
    user_turn = {"role": "user", "content": question}
    state.turns.append(user_turn)
    _render_turn(user_turn)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed: list[str] = []

        def on_text(text: str) -> None:
            streamed.append(text)
            placeholder.markdown(_md("".join(streamed)) + " ▌")

        try:
            with st.spinner("Searching your Drive…"):
                reply = resources.agent().ask(question, thread_id=state.thread_id, on_text=on_text)
        except (LLMError, ConfigError) as exc:
            placeholder.empty()
            st.error(str(exc))
            state.turns.pop()
            return
        placeholder.markdown(_md(reply.answer))
        render_citations(reply.citations)
        render_sources(reply.sources)

    state.turns.append(
        {
            "role": "assistant",
            "content": reply.answer,
            "citations": reply.citations,
            "sources": reply.sources,
        }
    )

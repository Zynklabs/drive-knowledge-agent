"""``DriveAgent``: the one entry point every interface (CLI, Streamlit, Slack) calls."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from drive_agent.core.config import ConfigError, Settings
from drive_agent.graph.rag import Citation, Retriever, Source, build_rag_graph
from drive_agent.llm import LLM

CONVERSATIONS_DB = "conversations.db"


def make_retriever(store: VectorStore) -> Retriever:
    """The graph's retriever over a LangChain vector store (hybrid when the store is)."""

    def retrieve(query: str, k: int) -> list[Document]:
        return store.similarity_search(query, k=k)

    return retrieve


@dataclass
class AgentReply:
    answer: str
    status: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    """Passages the answer was based on (after grading)."""
    query: str = ""
    thread_id: str = ""


def create_checkpointer(settings: Settings) -> BaseCheckpointSaver[Any]:
    """Long-lived conversation memory, safe to share across threads (e.g. Streamlit sessions).

    ``sqlite`` keeps threads across restarts in ``<data_dir>/conversations.db``.
    """
    kind = settings.agent.checkpointer
    if kind == "memory":
        return InMemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        settings.data_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(settings.data_dir / CONVERSATIONS_DB, check_same_thread=False)
        return SqliteSaver(conn)
    raise ConfigError(f"agent.checkpointer={kind} is not supported yet; use sqlite or memory")


@contextmanager
def open_checkpointer(settings: Settings) -> Iterator[BaseCheckpointSaver[Any]]:
    """``create_checkpointer`` for one-shot use (CLI); closes the database afterwards."""
    saver = create_checkpointer(settings)
    try:
        yield saver
    finally:
        conn = getattr(saver, "conn", None)
        if conn is not None:
            conn.close()


class DriveAgent:
    def __init__(
        self,
        settings: Settings,
        llm: LLM,
        retriever: Retriever,
        checkpointer: BaseCheckpointSaver[Any],
    ) -> None:
        if settings.agent.mode != "rag":
            raise ConfigError(f"agent.mode={settings.agent.mode} is planned for M7; use rag")
        self._graph = build_rag_graph(llm, retriever, settings.retrieval, checkpointer)

    def ask(
        self,
        question: str,
        *,
        thread_id: str | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> AgentReply:
        thread_id = thread_id or uuid.uuid4().hex[:12]
        config: Any = {"configurable": {"thread_id": thread_id}}
        state: dict[str, Any] = {}
        for item in self._graph.stream(
            {"question": question}, config, stream_mode=["custom", "values"]
        ):
            mode, chunk = cast(tuple[str, dict[str, Any]], item)
            if mode == "custom" and on_text is not None:
                on_text(chunk["text"])
            elif mode == "values":
                state = chunk
        return AgentReply(
            answer=state.get("answer", ""),
            status=state.get("status", ""),
            citations=state.get("citations", []),
            sources=state.get("sources", []),
            query=state.get("query", ""),
            thread_id=thread_id,
        )

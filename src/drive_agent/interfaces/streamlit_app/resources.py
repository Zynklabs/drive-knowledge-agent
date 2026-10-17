"""Long-lived objects shared by every Streamlit session: settings, embeddings, the vector index
and conversation memory. Loaded once per server process (``st.cache_resource``)."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import streamlit as st
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.base import BaseCheckpointSaver

from drive_agent import embeddings as embeddings_factory
from drive_agent import vectorstores
from drive_agent.core.agent import DriveAgent, create_checkpointer, make_retriever
from drive_agent.core.config import ConfigError, Settings, load_settings
from drive_agent.llm import claude
from drive_agent.vectorstores import VectorIndex, VectorStoreError


@dataclass
class Resources:
    settings: Settings
    embeddings: Embeddings
    index: VectorIndex
    checkpointer: BaseCheckpointSaver[Any]

    def agent(self) -> DriveAgent:
        llm = claude.ClaudeLLM(self.settings)
        return DriveAgent(self.settings, llm, make_retriever(self.index.store), self.checkpointer)

    def close(self) -> None:
        self.index.close()
        conn = getattr(self.checkpointer, "conn", None)
        if conn is not None:
            conn.close()


@st.cache_resource(show_spinner="Loading the embedding model and index…")
def _load() -> Resources:
    settings = load_settings()
    embeddings = embeddings_factory.build_embeddings(settings)
    index = vectorstores.build_vector_index(settings, embeddings)
    return Resources(settings, embeddings, index, create_checkpointer(settings))


def get_resources() -> Resources | None:
    """The shared resources, or ``None`` after showing why they could not be loaded."""
    try:
        return _load()
    except (ConfigError, VectorStoreError) as exc:
        st.error(f"**Setup problem:** {exc}")
        st.caption("Fix the configuration (see the Setup page), then reload.")
        return None


def reset_resources() -> None:
    """Release the index (and its on-disk lock in embedded mode) and reload on next use."""
    with contextlib.suppress(Exception):  # nothing to release if loading had failed
        _load().close()
    _load.clear()

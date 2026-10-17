"""The web app's KnowledgeBase, shared by every Streamlit session for the life of the server.

Loading it also starts auto-sync, so the index stays fresh while the app runs.
"""

from __future__ import annotations

import contextlib

import streamlit as st

from drive_agent.core.config import ConfigError, load_settings
from drive_agent.core.knowledge_base import KnowledgeBase
from drive_agent.vectorstores import VectorStoreError


@st.cache_resource(show_spinner="Loading the embedding model and index…")
def _load() -> KnowledgeBase:
    kb = KnowledgeBase.open(load_settings())
    kb.start_auto_sync()
    return kb


def get_resources() -> KnowledgeBase | None:
    """The shared knowledge base, or ``None`` after showing why it could not be loaded."""
    try:
        return _load()
    except (ConfigError, VectorStoreError) as exc:
        st.error(f"**Setup problem:** {exc}")
        st.caption("Fix the configuration (see the Setup page), then reload.")
        return None


def reset_resources() -> None:
    """Stop auto-sync, release the index (and its on-disk lock), reload on next use."""
    with contextlib.suppress(Exception):  # nothing to release if loading had failed
        _load().close()
    _load.clear()

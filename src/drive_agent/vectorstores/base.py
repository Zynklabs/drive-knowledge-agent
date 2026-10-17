"""What the sync engine needs from any vector database."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.vectorstores import VectorStore


class VectorStoreError(Exception):
    """The vector database is unreachable or incompatible with the configuration."""


@dataclass(frozen=True)
class VectorIndex:
    store: VectorStore
    """A LangChain vector store; used by the indexing API and, from M4, by retrieval."""
    namespace: str
    """Identifies this physical index (provider, location, collection) in the sync state."""
    description: str
    reset: Callable[[], None]
    """Drop every vector and recreate the empty collection."""
    count: Callable[[], int]
    close: Callable[[], None]
    """Release connections and, in embedded mode, the on-disk lock."""

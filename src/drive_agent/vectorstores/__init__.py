"""Build the configured vector index. Qdrant is supported now; others land in M7."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from drive_agent.core.config import ConfigError, Settings
from drive_agent.vectorstores.base import VectorIndex, VectorStoreError


def build_vector_index(settings: Settings, embeddings: Embeddings) -> VectorIndex:
    provider = settings.vector_store.provider
    if provider == "qdrant":
        from drive_agent.vectorstores.qdrant import build_qdrant

        return build_qdrant(settings, embeddings)
    raise ConfigError(f"vector_store.provider={provider} is not supported yet (planned for M7)")


__all__ = ["VectorIndex", "VectorStoreError", "build_vector_index"]

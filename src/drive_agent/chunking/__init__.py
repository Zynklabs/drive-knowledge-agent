"""Structure-aware chunking (contextual prefixes are added at index time, M3)."""

from drive_agent.chunking.splitter import approx_tokens, chunk_document

__all__ = ["approx_tokens", "chunk_document"]

"""Qdrant via ``langchain-qdrant``.

Where the data lives is the operator's choice:
- ``QDRANT_URL`` (+ ``QDRANT_API_KEY``): a self-hosted server or Qdrant Cloud (production);
- no URL: embedded mode on disk at ``<data_dir>/qdrant`` (zero setup, single process);
- ``:memory:``: in-process, for tests.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from drive_agent.core.config import Settings
from drive_agent.vectorstores.base import VectorIndex, VectorStoreError
from drive_agent.vectorstores.location import describe_location, location, namespace_for

INDEXED_FIELDS = ("metadata.file_id", "metadata.folder_root", "metadata.kind")


def _client(settings: Settings) -> QdrantClient:
    config = settings.vector_store
    if config.url == ":memory:":
        return QdrantClient(location=":memory:")
    if config.url:
        key = settings.secrets.qdrant_api_key
        return QdrantClient(
            url=config.url, api_key=key.get_secret_value() if key else None, timeout=30
        )
    path = location(settings)
    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=path)


def _vector_size(client: QdrantClient, collection: str) -> int | None:
    vectors: Any = client.get_collection(collection).config.params.vectors
    return getattr(vectors, "size", None)


def _ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    if client.collection_exists(collection):
        existing = _vector_size(client, collection)
        if existing is not None and existing != dim:
            raise VectorStoreError(
                f"Qdrant collection {collection!r} stores {existing}-dimensional vectors but the "
                f"embedding model produces {dim}. Run `drive-agent sync --full` to rebuild it, "
                "or set vector_store.collection to a new name."
            )
        return
    client.create_collection(
        collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    with warnings.catch_warnings():  # embedded mode warns that payload indexes are no-ops
        warnings.simplefilter("ignore")
        for field in INDEXED_FIELDS:
            client.create_payload_index(collection, field, field_schema=PayloadSchemaType.KEYWORD)


def build_qdrant(settings: Settings, embeddings: Embeddings) -> VectorIndex:
    collection = settings.vector_store.collection
    client = _client(settings)
    try:
        client.collection_exists(collection)
    except Exception as exc:  # httpx / grpc / qdrant errors vary by transport
        client.close()
        raise VectorStoreError(f"cannot reach Qdrant at {location(settings)}: {exc}") from exc

    dim = len(embeddings.embed_query("dimension probe"))
    try:
        _ensure_collection(client, collection, dim)
    except VectorStoreError:
        client.close()
        raise

    def reset() -> None:
        client.delete_collection(collection)
        _ensure_collection(client, collection, dim)

    def count() -> int:
        return client.count(collection, exact=True).count

    return VectorIndex(
        store=QdrantVectorStore(client=client, collection_name=collection, embedding=embeddings),
        namespace=namespace_for(settings),
        description=describe_location(settings),
        reset=reset,
        count=count,
        close=client.close,
    )

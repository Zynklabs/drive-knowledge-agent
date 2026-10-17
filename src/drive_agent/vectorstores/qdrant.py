"""Qdrant via ``langchain-qdrant``.

Where the data lives is the operator's choice:
- ``QDRANT_URL`` (+ ``QDRANT_API_KEY``): a self-hosted server or Qdrant Cloud (production);
- no URL: embedded mode on disk at ``<data_dir>/qdrant`` (zero setup, single process);
- ``:memory:``: in-process, for tests.

With ``retrieval.hybrid`` (default) every chunk also gets a BM25 sparse vector, and searches
fuse dense + keyword results with Reciprocal Rank Fusion inside Qdrant.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

from drive_agent.core.config import Settings
from drive_agent.vectorstores.base import VectorIndex, VectorStoreError
from drive_agent.vectorstores.location import describe_location, location, namespace_for

DENSE = "dense"
SPARSE = "bm25"
SPARSE_MODEL = "Qdrant/bm25"
INDEXED_FIELDS = ("metadata.file_id", "metadata.folder_root", "metadata.kind")
REBUILD_HINT = (
    "Run `drive-agent sync --full` to rebuild it, or set vector_store.collection to a new name."
)


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


def _check_schema(client: QdrantClient, collection: str, dim: int, hybrid: bool) -> None:
    params: Any = client.get_collection(collection).config.params
    dense = params.vectors.get(DENSE) if isinstance(params.vectors, dict) else None
    if dense is None:
        raise VectorStoreError(
            f"Qdrant collection {collection!r} was created by an older version of the agent. "
            + REBUILD_HINT
        )
    if dense.size != dim:
        raise VectorStoreError(
            f"Qdrant collection {collection!r} stores {dense.size}-dimensional vectors but the "
            f"embedding model produces {dim}. " + REBUILD_HINT
        )
    if hybrid and SPARSE not in (params.sparse_vectors or {}):
        raise VectorStoreError(
            f"Qdrant collection {collection!r} has no keyword (BM25) vectors but "
            "retrieval.hybrid is on. " + REBUILD_HINT
        )


def _ensure_collection(client: QdrantClient, collection: str, dim: int, hybrid: bool) -> None:
    if client.collection_exists(collection):
        _check_schema(client, collection, dim, hybrid)
        return
    client.create_collection(
        collection,
        vectors_config={DENSE: VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config=(
            {SPARSE: SparseVectorParams(modifier=Modifier.IDF)} if hybrid else None
        ),
    )
    with warnings.catch_warnings():  # embedded mode warns that payload indexes are no-ops
        warnings.simplefilter("ignore")
        for field in INDEXED_FIELDS:
            client.create_payload_index(collection, field, field_schema=PayloadSchemaType.KEYWORD)


def _bm25(settings: Settings) -> SparseEmbeddings:
    from langchain_qdrant import FastEmbedSparse  # deferred: loads onnxruntime

    cache = settings.data_dir / "models"
    cache.mkdir(parents=True, exist_ok=True)
    return FastEmbedSparse(model_name=SPARSE_MODEL, cache_dir=str(cache))


def build_qdrant(
    settings: Settings,
    embeddings: Embeddings,
    *,
    sparse: SparseEmbeddings | None = None,
    recreate: bool = False,
) -> VectorIndex:
    collection = settings.vector_store.collection
    hybrid = settings.retrieval.hybrid
    client = _client(settings)
    try:
        client.collection_exists(collection)
    except Exception as exc:  # httpx / grpc / qdrant errors vary by transport
        client.close()
        raise VectorStoreError(f"cannot reach Qdrant at {location(settings)}: {exc}") from exc

    dim = len(embeddings.embed_query("dimension probe"))
    try:
        if recreate and client.collection_exists(collection):
            client.delete_collection(collection)
        _ensure_collection(client, collection, dim, hybrid)
    except VectorStoreError:
        client.close()
        raise

    def reset() -> None:
        client.delete_collection(collection)
        _ensure_collection(client, collection, dim, hybrid)

    def count() -> int:
        return client.count(collection, exact=True).count

    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
        vector_name=DENSE,
        sparse_embedding=(sparse or _bm25(settings)) if hybrid else None,
        sparse_vector_name=SPARSE,
        retrieval_mode=RetrievalMode.HYBRID if hybrid else RetrievalMode.DENSE,
    )
    return VectorIndex(
        store=store,
        namespace=namespace_for(settings),
        description=describe_location(settings) + (" (hybrid: dense + BM25)" if hybrid else ""),
        reset=reset,
        count=count,
        close=client.close,
    )

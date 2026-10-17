from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from drive_agent.core.config import ConfigError, Settings
from drive_agent.vectorstores import VectorStoreError, build_vector_index
from drive_agent.vectorstores.location import describe_location, location, namespace_for
from drive_agent.vectorstores.qdrant import _ensure_collection


def settings(**vector_store: object) -> Settings:
    return Settings.model_validate({"vector_store": vector_store})


def test_in_memory_qdrant_roundtrip() -> None:
    index = build_vector_index(settings(url=":memory:"), DeterministicFakeEmbedding(size=8))
    try:
        assert index.count() == 0
        index.store.add_documents(
            [Document("hello", metadata={"file_id": "F1", "page": 2})],
            ids=["00000000-0000-0000-0000-000000000001"],
        )
        assert index.count() == 1
        hit = index.store.similarity_search("hello", k=1)[0]
        assert hit.page_content == "hello"
        assert hit.metadata["file_id"] == "F1"
        assert hit.metadata["page"] == 2
        index.reset()
        assert index.count() == 0
        assert index.description == "qdrant :memory: / drive_agent (hybrid: dense + BM25)"
    finally:
        index.close()


def test_embedded_mode_persists_on_disk(tmp_path: Path) -> None:
    s = Settings.model_validate({"data_dir": str(tmp_path)})
    index = build_vector_index(s, DeterministicFakeEmbedding(size=4))
    index.store.add_texts(["x"], metadatas=[{"file_id": "F"}])
    index.close()
    reopened = build_vector_index(s, DeterministicFakeEmbedding(size=4))
    try:
        assert reopened.count() == 1
        assert (tmp_path / "qdrant").is_dir()
    finally:
        reopened.close()


def test_dimension_mismatch_is_refused() -> None:
    client = QdrantClient(location=":memory:")
    _ensure_collection(client, "c", 8, True)
    _ensure_collection(client, "c", 8, True)  # same layout: fine
    with pytest.raises(VectorStoreError, match=r"8-dimensional .* produces 16"):
        _ensure_collection(client, "c", 16, True)


def test_collection_without_bm25_needs_rebuild_for_hybrid() -> None:
    client = QdrantClient(location=":memory:")
    _ensure_collection(client, "c", 8, False)
    with pytest.raises(VectorStoreError, match="no keyword"):
        _ensure_collection(client, "c", 8, True)


def test_pre_m4_collection_layout_is_detected() -> None:
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(location=":memory:")
    client.create_collection("c", vectors_config=VectorParams(size=8, distance=Distance.COSINE))
    with pytest.raises(VectorStoreError, match="older version"):
        _ensure_collection(client, "c", 8, True)


def test_recreate_rebuilds_an_incompatible_collection(tmp_path: Path) -> None:
    s = Settings.model_validate({"data_dir": str(tmp_path), "retrieval": {"hybrid": False}})
    old = build_vector_index(s, DeterministicFakeEmbedding(size=4))
    old.store.add_texts(["x"])
    old.close()
    hybrid = Settings.model_validate({"data_dir": str(tmp_path)})
    with pytest.raises(VectorStoreError, match="sync --full"):
        build_vector_index(hybrid, DeterministicFakeEmbedding(size=4))
    rebuilt = build_vector_index(hybrid, DeterministicFakeEmbedding(size=4), recreate=True)
    try:
        assert rebuilt.count() == 0
        assert rebuilt.description.endswith("(hybrid: dense + BM25)")
    finally:
        rebuilt.close()


def test_hybrid_search_finds_exact_keywords() -> None:
    # Random dense vectors carry no meaning here, so a hit on the rare token proves BM25 works.
    index = build_vector_index(settings(url=":memory:"), DeterministicFakeEmbedding(size=8))
    try:
        texts = [f"generic resume text number {i}" for i in range(20)] + ["Certified in XYZQ-77"]
        index.store.add_texts(texts, metadatas=[{"file_id": str(i)} for i in range(21)])
        # RRF over k candidates per method: the only BM25 match is guaranteed a top-2 spot.
        hits = index.store.similarity_search("XYZQ-77", k=5)
        assert "20" in [h.metadata["file_id"] for h in hits[:2]]
    finally:
        index.close()


def test_unreachable_server() -> None:
    with pytest.raises(VectorStoreError, match=r"cannot reach Qdrant at http://127\.0\.0\.1:9"):
        build_vector_index(settings(url="http://127.0.0.1:9"), DeterministicFakeEmbedding(size=4))


def test_other_providers_not_yet_supported() -> None:
    with pytest.raises(ConfigError, match="planned for M7"):
        build_vector_index(settings(provider="pinecone"), DeterministicFakeEmbedding(size=4))


def test_location_and_namespace() -> None:
    embedded = Settings.model_validate({"data_dir": "/data"})
    assert location(embedded) == "/data/qdrant"
    remote = settings(url="https://x.cloud.qdrant.io", collection="hr")
    assert namespace_for(remote) == "qdrant:https://x.cloud.qdrant.io/hr"
    assert describe_location(remote) == "qdrant https://x.cloud.qdrant.io / hr"
    # a different server or collection must never share sync records
    assert namespace_for(remote) != namespace_for(
        settings(url="http://localhost:6333", collection="hr")
    )

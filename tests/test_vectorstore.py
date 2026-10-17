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
        assert index.description == "qdrant :memory: / drive_agent"
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
    _ensure_collection(client, "c", 8)
    _ensure_collection(client, "c", 8)  # same dimension: fine
    with pytest.raises(VectorStoreError, match=r"8-dimensional .* produces 16"):
        _ensure_collection(client, "c", 16)


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

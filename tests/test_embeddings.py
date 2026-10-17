from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings

from drive_agent.core.config import ConfigError, Settings
from drive_agent.embeddings import build_embeddings, local
from drive_agent.embeddings.local import FastEmbedEmbeddings


class FakeModel:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts: Iterable[str], batch_size: int | None = None) -> Iterator[Any]:
        for text in texts:
            self.seen.append(text)
            yield np.array([float(len(text)), 1.0], dtype=np.float32)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    model = FakeModel()
    loaded: list[tuple[str, Path]] = []

    def load(name: str, cache_dir: Path) -> FakeModel:
        loaded.append((name, cache_dir))
        return model

    monkeypatch.setattr(local, "_load_model", load)
    model.loaded = loaded  # type: ignore[attr-defined]
    return model


def test_nomic_task_prefixes(fake_model: FakeModel, tmp_path: Path) -> None:
    emb = FastEmbedEmbeddings("nomic-ai/nomic-embed-text-v1.5", cache_dir=tmp_path / "m")
    docs = emb.embed_documents(["alpha", "beta"])
    query = emb.embed_query("who knows Go?")
    assert fake_model.seen == [
        "search_document: alpha",
        "search_document: beta",
        "search_query: who knows Go?",
    ]
    assert docs == [[22.0, 1.0], [21.0, 1.0]]
    assert isinstance(query[0], float)
    assert (tmp_path / "m").is_dir()


def test_models_without_prefixes(fake_model: FakeModel, tmp_path: Path) -> None:
    emb = FastEmbedEmbeddings("ibm-granite/granite-embedding-small-english-r2", cache_dir=tmp_path)
    emb.embed_documents(["alpha"])
    emb.embed_query("q")
    assert fake_model.seen == ["alpha", "q"]


def test_factory_local_uses_data_dir_cache(fake_model: FakeModel, tmp_path: Path) -> None:
    settings = Settings.model_validate({"data_dir": str(tmp_path)})
    emb = build_embeddings(settings)
    assert isinstance(emb, FastEmbedEmbeddings)
    assert emb.model_name == "nomic-ai/nomic-embed-text-v1.5"
    assert fake_model.loaded == [("nomic-ai/nomic-embed-text-v1.5", tmp_path / "models")]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("provider", "key_var", "cls"),
    [
        ("voyage", "voyage_api_key", VoyageAIEmbeddings),
        ("openai", "openai_api_key", OpenAIEmbeddings),
    ],
)
def test_factory_hosted_providers(provider: str, key_var: str, cls: type) -> None:
    settings = Settings.model_validate({"embeddings": {"provider": provider}})
    with pytest.raises(ConfigError, match=key_var.upper()):
        build_embeddings(settings)
    settings.secrets = settings.secrets.model_copy(update={key_var: "k"})
    from pydantic import SecretStr

    settings.secrets = settings.secrets.model_copy(update={key_var: SecretStr("k")})
    assert isinstance(build_embeddings(settings), cls)


def test_factory_unsupported_provider() -> None:
    settings = Settings.model_validate({"embeddings": {"provider": "cohere"}})
    with pytest.raises(ConfigError, match="not supported yet"):
        build_embeddings(settings)

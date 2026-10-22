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


class FakeTokenizer:
    def __init__(self, max_length: int) -> None:
        self.truncation: dict[str, Any] = {"max_length": max_length, "direction": "left"}

    def enable_truncation(self, max_length: int, direction: str = "right") -> None:
        self.truncation = {"max_length": max_length, "direction": direction}


@pytest.mark.parametrize(("model_limit", "expected"), [(8192, 1024), (512, 512)])
def test_input_length_is_capped_never_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_limit: int, expected: int
) -> None:
    tokenizer = FakeTokenizer(model_limit)
    model = FakeModel()
    model.model = type("Inner", (), {"tokenizer": tokenizer})()  # type: ignore[attr-defined]
    monkeypatch.setattr(local, "_load_model", lambda name, cache_dir: model)
    FastEmbedEmbeddings("nomic-ai/nomic-embed-text-v1.5", cache_dir=tmp_path, max_tokens=1024)
    assert tokenizer.truncation == {"max_length": expected, "direction": "left"}


class FakeSession:
    def __init__(self) -> None:
        self.run_options: list[Any] = []

    def run(self, output_names: Any, input_feed: Any, run_options: Any = None) -> list[str]:
        self.run_options.append(run_options)
        return ["out"]


def test_every_run_shrinks_the_memory_arena(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = FakeSession()
    model = FakeModel()
    model.model = type("Inner", (), {"model": session})()  # type: ignore[attr-defined]
    monkeypatch.setattr(local, "_load_model", lambda name, cache_dir: model)
    FastEmbedEmbeddings("nomic-ai/nomic-embed-text-v1.5", cache_dir=tmp_path)

    assert session.run(["x"], {"input_ids": 1}) == ["out"]
    key, value = local.ARENA_SHRINKAGE
    assert session.run_options[0].get_run_config_entry(key) == value

    explicit = object()
    session.run(["x"], {}, explicit)
    assert session.run_options[1] is explicit


def test_documents_embed_longest_first_in_small_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    batch_sizes: list[int | None] = []

    class RecordingModel(FakeModel):
        def embed(self, texts: Iterable[str], batch_size: int | None = None) -> Iterator[Any]:
            batch_sizes.append(batch_size)
            return super().embed(texts, batch_size)

    model = RecordingModel()
    monkeypatch.setattr(local, "_load_model", lambda name, cache_dir: model)
    emb = FastEmbedEmbeddings("ibm-granite/granite-embedding-small-english-r2", cache_dir=tmp_path)

    vectors = emb.embed_documents(["bb", "dddd", "a", "ccc"])

    assert model.seen == ["dddd", "ccc", "bb", "a"]
    assert [v[0] for v in vectors] == [2.0, 4.0, 1.0, 3.0]  # back in input order
    assert batch_sizes == [local.BATCH_SIZE]

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from drive_agent.core.config import Settings
from drive_agent.drive.fetcher import DriveFetcher
from drive_agent.indexing.state import SQLiteRecordManager
from drive_agent.indexing.sync import SyncError, run_sync
from drive_agent.vectorstores import VectorIndex, build_vector_index

from .doc_factory import make_pdf
from .fake_drive import FakeDrive, fake_download, file, folder, http_error

DOC = "application/vnd.google-apps.document"
PDF = "application/pdf"


@pytest.fixture(autouse=True)
def _fake_media(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DriveFetcher, "_download", fake_download)


class Env:
    def __init__(self, tmp_path: Path, folders: list[str] | None = None) -> None:
        self.settings = Settings.model_validate(
            {
                "data_dir": str(tmp_path),
                "drive": {"folder_ids": folders or ["root"]},
                "vector_store": {"url": ":memory:"},
            }
        )
        self.index: VectorIndex = build_vector_index(
            self.settings, DeterministicFakeEmbedding(size=16)
        )
        self.records = SQLiteRecordManager(self.index.namespace, tmp_path / "state.db")
        self.drive = FakeDrive(
            [
                folder("root", "Hiring"),
                file("doc", "Process", DOC, "root", md5Checksum="v1"),
                file("cv", "Jane.pdf", PDF, "root", size=10, md5Checksum="p1"),
            ],
            contents={
                "doc": b"# Stages\n\nPhone screen, then onsite.\n\n# Offer\n\nWithin a week.",
                "cv": make_pdf(["Jane Doe", "Python and Go"]),
            },
        )

    def sync(self, **kw: Any) -> Any:
        return run_sync(self.drive, self.settings, self.index, self.records, **kw)


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    e = Env(tmp_path)
    yield e
    e.records.close()
    e.index.close()


def test_first_sync_indexes_everything(env: Env) -> None:
    report = env.sync()
    assert (report.added, report.updated, report.unchanged, report.deleted) == (4, 0, 0, 0)
    assert env.index.count() == 4
    assert env.records.count() == 4
    assert env.records.list_groups() == ["cv", "doc"]
    hit = env.index.store.similarity_search("Phone screen", k=4)
    assert {h.metadata["file_id"] for h in hit} == {"doc", "cv"}
    assert {"chunk_id", "path", "web_url"} - {"web_url"} <= hit[0].metadata.keys()
    last = env.records.get_meta("last_sync")
    assert (last["files"], last["chunks"], last["failed"]) == (2, 4, 0)
    assert env.records.get_meta("embedding") == "local:nomic-ai/nomic-embed-text-v1.5"


def test_resync_is_a_no_op(env: Env) -> None:
    env.sync()
    report = env.sync()
    assert (report.added, report.unchanged, report.deleted, report.pruned_files) == (0, 4, 0, 0)
    assert env.index.count() == 4


def test_changed_file_replaces_its_chunks(env: Env) -> None:
    env.sync()
    env.drive.items["doc"]["md5Checksum"] = "v2"
    env.drive.contents["doc"] = b"# Stages\n\nTwo onsite rounds now."
    report = env.sync()
    assert report.added == 1
    assert report.deleted == 2  # both old "doc" chunks
    assert report.unchanged == 2  # the PDF
    assert env.index.count() == 3
    texts = [d.page_content for d in env.index.store.similarity_search("x", k=10)]
    assert any("Two onsite rounds" in t for t in texts)
    assert not any("Within a week" in t for t in texts)


def test_removed_file_is_pruned(env: Env) -> None:
    env.sync()
    del env.drive.items["cv"]
    report = env.sync()
    assert (report.pruned_files, report.deleted) == (1, 2)
    assert env.records.list_groups() == ["doc"]
    assert env.index.count() == 2


def test_file_that_stops_yielding_text_is_pruned(env: Env) -> None:
    env.sync()
    env.drive.contents["cv"] = make_pdf([""])  # e.g. replaced by a scan
    env.drive.items["cv"]["md5Checksum"] = "p2"
    report = env.sync()
    assert report.pruned_files == 1
    assert env.records.list_groups() == ["doc"]


def test_fetch_error_keeps_previous_chunks(env: Env) -> None:
    env.sync()
    env.drive.contents["cv"] = http_error(500, "backendError", "flaky")
    report = env.sync()
    assert report.deleted == 0
    assert env.index.count() == 4
    assert [r.error is not None for r in report.files] == [False, True]
    assert env.records.get_meta("last_sync")["failed"] == 1


def test_folder_error_skips_pruning(tmp_path: Path) -> None:
    e = Env(tmp_path, folders=["root", "missing"])
    try:
        e.sync()
        del e.drive.items["cv"]
        report = e.sync()
        assert report.prune_skipped is True
        assert report.pruned_files == 0
        assert e.index.count() == 4  # nothing removed while a folder is unreadable
    finally:
        e.records.close()
        e.index.close()


def test_embedding_change_requires_full_rebuild(env: Env) -> None:
    env.sync()
    env.records.set_meta("embedding", "voyage:voyage-3.5")
    with pytest.raises(SyncError, match="sync --full"):
        env.sync()
    report = env.sync(full=True)
    assert report.added == 4
    assert env.index.count() == 4
    assert env.records.get_meta("embedding") == "local:nomic-ai/nomic-embed-text-v1.5"


def test_full_rebuild_reembeds_everything(env: Env) -> None:
    env.sync()
    report = env.sync(full=True)
    assert (report.added, report.unchanged) == (4, 0)
    assert env.index.count() == 4


def test_externally_emptied_store_heals(env: Env) -> None:
    env.sync()
    env.index.reset()  # someone dropped the collection
    report = env.sync()
    assert report.healed is True
    assert report.added == 4
    assert env.index.count() == 4


def test_progress_callback(env: Env) -> None:
    seen: list[str] = []
    env.sync(on_file=lambda r: seen.append(r.file.id))
    assert seen == ["doc", "cv"]


def test_chunk_key_is_a_stable_uuid() -> None:
    import uuid

    from langchain_core.documents import Document

    from drive_agent.indexing.sync import chunk_key

    a = Document("text", metadata={"file_id": "F", "page": 1})
    same = Document("text", metadata={"page": 1, "file_id": "F"})  # key order irrelevant
    other = Document("text", metadata={"file_id": "F", "page": 2})
    assert chunk_key(a) == chunk_key(same) != chunk_key(other)
    assert str(uuid.UUID(chunk_key(a))) == chunk_key(a)

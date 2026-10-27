from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from drive_agent.core.config import Settings
from drive_agent.core.monitor import SYNC_PROGRESS_FILE, read_json, read_sync_history
from drive_agent.drive.fetcher import DriveFetcher
from drive_agent.indexing import sync as sync_module
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


def downloads(env: Env) -> list[str]:
    return [file_id for file_id, _ in env.drive.media_calls]


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
    assert env.records.get_meta("embedding") == "local:nomic-ai/nomic-embed-text-v1.5+bm25"


def test_stopped_sync_keeps_saved_progress(env: Env) -> None:
    # Each file is saved as soon as it is loaded, so a sync stopped part-way keeps what it
    # indexed and the next run only does the rest.

    class Stop(Exception):
        pass

    def stop_at_cv(result: Any) -> None:
        if result.file.id == "cv":
            raise Stop

    with pytest.raises(Stop):
        env.sync(on_file=stop_at_cv)
    assert env.records.list_groups() == ["doc"]
    assert env.index.count() == 2

    env.drive.media_calls.clear()
    report = env.sync()
    assert (report.added, report.unchanged) == (2, 2)
    assert env.index.count() == 4
    assert downloads(env) == ["cv"]  # "doc" was finished, so it is not downloaded again


def test_file_larger_than_a_batch_is_not_re_embedded(env: Env) -> None:
    env.drive.contents["doc"] = "\n\n".join(
        f"# Section {i}\n\n" + f"Paragraph {i} " * 400 for i in range(80)
    ).encode()
    first = env.sync()
    assert first.added > 64
    again = env.sync()
    assert (again.added, again.deleted) == (0, 0)


def test_sync_progress_is_recorded(env: Env) -> None:
    env.sync()
    progress = read_json(env.settings.data_dir / SYNC_PROGRESS_FILE)
    assert progress is not None
    assert (progress["state"], progress["files_done"], progress["chunks_added"]) == ("done", 2, 4)
    assert progress["files_total"] == 2
    last = env.records.get_meta("last_sync")
    assert (last["total_files"], last["empty"], last["skipped"]) == (2, 0, 0)


def test_each_sync_is_added_to_the_history(env: Env) -> None:
    env.sync(trigger="schedule")
    env.drive.items["doc"]["md5Checksum"] = "v2"
    env.drive.contents["doc"] = b"# Stages\n\nTwo onsite rounds now."
    env.drive.items["new"] = file("new", "Notes", DOC, "root", md5Checksum="n1")
    env.drive.contents["new"] = b"# Notes\n\nNew file."
    del env.drive.items["cv"]
    env.sync()

    first, second = read_sync_history(env.settings.data_dir)
    assert (first["trigger"], first["state"], first["new_files"], first["changed_files"]) == (
        "schedule",
        "done",
        2,
        0,
    )
    assert (second["trigger"], second["new_files"], second["changed_files"]) == ("manual", 1, 1)
    assert (second["removed_files"], second["files_total"]) == (1, 2)
    assert "pid" not in second


def test_interrupted_sync_is_recorded_as_failed(env: Env) -> None:
    def interrupt(result: Any) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        env.sync(on_file=interrupt)
    progress = read_json(env.settings.data_dir / SYNC_PROGRESS_FILE)
    assert progress is not None
    assert (progress["state"], progress["error"]) == ("failed", "stopped")


def test_huge_file_is_written_in_capped_batches(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    # One request for a whole huge file once exceeded Qdrant's 32 MB request limit.
    env.drive.contents["doc"] = "\n\n".join(
        f"# Section {i}\n\n" + f"Paragraph {i} " * 400 for i in range(200)
    ).encode()
    store = env.index.store
    writes: list[int] = []
    original = store.add_documents

    def spy(docs: list[Any], **kwargs: Any) -> list[str]:
        writes.append(len(docs))
        return original(docs, **kwargs)

    monkeypatch.setattr(store, "add_documents", spy)
    report = env.sync()
    assert report.added > 2 * sync_module.UPSERT_BATCH
    assert max(writes) <= sync_module.UPSERT_BATCH
    assert env.records.count() == report.added


def test_resync_is_a_no_op(env: Env) -> None:
    env.sync()
    report = env.sync()
    assert (report.added, report.unchanged, report.deleted, report.pruned_files) == (0, 4, 0, 0)
    assert env.index.count() == 4


def test_unchanged_files_are_not_downloaded(env: Env) -> None:
    env.sync()
    assert downloads(env) == ["doc", "cv"]
    before = env.records.get_time()
    env.drive.media_calls.clear()
    report = env.sync()
    assert downloads(env) == []
    assert [r.unchanged for r in report.files] == [True, True]
    assert (report.added, report.unchanged, report.deleted, report.pruned_files) == (0, 4, 0, 0)
    assert len(env.records.list_keys(after=before)) == 4  # every record marked as seen
    progress = read_json(env.settings.data_dir / SYNC_PROGRESS_FILE)
    assert progress is not None
    assert (progress["files_done"], progress["files_empty"], progress["chunks_added"]) == (2, 0, 0)
    last = env.records.get_meta("last_sync")
    assert (last["files"], last["chunks"], last["empty"]) == (2, 4, 0)


def test_file_without_md5_is_reloaded_when_modified_time_changes(env: Env) -> None:
    del env.drive.items["doc"]["md5Checksum"]  # Google Docs have no md5
    env.drive.items["doc"]["modifiedTime"] = "2026-10-01T09:00:00Z"
    env.sync()
    env.drive.media_calls.clear()
    env.sync()
    assert downloads(env) == []

    env.drive.items["doc"]["modifiedTime"] = "2026-10-02T09:00:00Z"
    env.drive.contents["doc"] = b"# Stages\n\nTwo onsite rounds now."
    report = env.sync()
    assert downloads(env) == ["doc"]
    assert (report.added, report.deleted) == (1, 2)
    texts = [d.page_content for d in env.index.store.similarity_search("x", k=10)]
    assert any("Two onsite rounds" in t for t in texts)


def test_file_without_any_version_is_always_reloaded(env: Env) -> None:
    del env.drive.items["doc"]["md5Checksum"]
    env.sync()
    env.drive.media_calls.clear()
    report = env.sync()
    assert downloads(env) == ["doc"]
    assert report.unchanged == 4


def test_renamed_file_is_reindexed(env: Env) -> None:
    # Same content (md5), but the name is in every chunk's text and metadata.
    env.sync()
    env.drive.items["cv"]["name"] = "Jane Doe CV.pdf"
    env.drive.media_calls.clear()
    env.sync()
    assert downloads(env) == ["cv"]
    names = {d.metadata["file_name"] for d in env.index.store.similarity_search("x", k=10)}
    assert names == {"Process", "Jane Doe CV.pdf"}


def test_full_sync_downloads_everything(env: Env) -> None:
    env.sync()
    env.drive.media_calls.clear()
    env.sync(full=True)
    assert downloads(env) == ["doc", "cv"]


def test_changed_file_replaces_its_chunks(env: Env) -> None:
    env.sync()
    env.drive.items["doc"]["md5Checksum"] = "v2"
    env.drive.contents["doc"] = b"# Stages\n\nTwo onsite rounds now."
    env.drive.media_calls.clear()
    report = env.sync()
    assert downloads(env) == ["doc"]  # the unchanged PDF is not downloaded
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
    env.drive.media_calls.clear()
    report = env.sync()
    assert downloads(env) == []  # "doc" is unchanged and kept without a download
    assert (report.pruned_files, report.deleted) == (1, 2)
    assert env.records.list_groups() == ["doc"]
    assert env.index.count() == 2
    assert env.records.file_versions().keys() == {"doc"}


def test_file_that_stops_yielding_text_is_pruned(env: Env) -> None:
    env.sync()
    env.drive.contents["cv"] = make_pdf([""])  # e.g. replaced by a scan
    env.drive.items["cv"]["md5Checksum"] = "p2"
    report = env.sync()
    assert report.pruned_files == 1
    assert env.records.list_groups() == ["doc"]
    env.drive.media_calls.clear()
    report = env.sync()
    assert downloads(env) == []  # now remembered as empty
    assert (report.pruned_files, report.deleted) == (0, 0)


def add_scan(env: Env) -> None:
    """An image-only PDF: loads fine but yields no text."""
    env.drive.items["scan"] = file("scan", "Scan.pdf", PDF, "root", size=10, md5Checksum="s1")
    env.drive.contents["scan"] = make_pdf([""])


def test_empty_file_is_not_reloaded(env: Env) -> None:
    add_scan(env)
    env.sync()
    assert downloads(env) == ["doc", "cv", "scan"]
    env.drive.media_calls.clear()
    report = env.sync()
    assert downloads(env) == []
    assert [r.unchanged for r in report.files] == [True, True, True]
    assert (report.added, report.unchanged, report.deleted) == (0, 4, 0)
    progress = read_json(env.settings.data_dir / SYNC_PROGRESS_FILE)
    assert progress is not None
    assert (progress["files_done"], progress["files_empty"]) == (3, 1)
    last = env.records.get_meta("last_sync")
    assert (last["files"], last["empty"], last["total_files"]) == (2, 1, 3)


def test_changed_empty_file_is_reloaded(env: Env) -> None:
    add_scan(env)
    env.sync()
    env.drive.items["scan"]["md5Checksum"] = "s2"  # re-scanned with OCR
    env.drive.contents["scan"] = make_pdf(["Signed offer letter"])
    env.drive.media_calls.clear()
    report = env.sync()
    assert downloads(env) == ["scan"]
    assert report.added == 1
    assert env.records.list_groups() == ["cv", "doc", "scan"]


def test_empty_file_removed_from_drive_is_forgotten(env: Env) -> None:
    add_scan(env)
    env.sync()
    assert "scan" in env.records.file_version_ids()
    del env.drive.items["scan"]
    report = env.sync()
    assert report.pruned_files == 0  # it had no chunks to prune
    assert env.records.file_version_ids() == ["cv", "doc"]


def test_empty_file_version_is_kept_while_a_folder_fails(tmp_path: Path) -> None:
    e = Env(tmp_path, folders=["root", "missing"])
    try:
        add_scan(e)
        e.sync()
        del e.drive.items["scan"]
        e.sync()
        assert "scan" in e.records.file_version_ids()  # nothing forgotten on a partial read
    finally:
        e.records.close()
        e.index.close()


def test_empty_file_that_fails_keeps_its_version(env: Env) -> None:
    add_scan(env)
    env.sync()
    env.drive.items["scan"]["md5Checksum"] = "s2"
    env.drive.contents["scan"] = http_error(500, "backendError", "flaky")
    env.sync()
    assert env.records.file_versions()["scan"].empty is True
    env.drive.contents["scan"] = make_pdf([""])
    env.drive.media_calls.clear()
    env.sync()
    assert downloads(env) == ["scan"]  # still the old version on record, so retried


def test_fetch_error_keeps_previous_chunks(env: Env) -> None:
    env.sync()
    env.drive.items["cv"]["md5Checksum"] = "p2"  # changed, so it is fetched again
    env.drive.contents["cv"] = http_error(500, "backendError", "flaky")
    report = env.sync()
    assert report.deleted == 0
    assert env.index.count() == 4
    assert [r.error is not None for r in report.files] == [False, True]
    assert env.records.get_meta("last_sync")["failed"] == 1
    # The old version is still on record, so the next sync tries the file again.
    env.drive.contents["cv"] = make_pdf(["Jane Doe", "Python and Go", "Rust"])
    env.drive.media_calls.clear()
    env.sync()
    assert downloads(env) == ["cv"]


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
    assert env.records.get_meta("embedding") == "local:nomic-ai/nomic-embed-text-v1.5+bm25"


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


def test_toggling_hybrid_requires_full_rebuild(env: Env) -> None:
    from drive_agent.indexing.sync import embedding_fingerprint

    env.sync()
    dense_only = env.settings.model_copy(
        update={"retrieval": env.settings.retrieval.model_copy(update={"hybrid": False})}
    )
    assert embedding_fingerprint(dense_only) == "local:nomic-ai/nomic-embed-text-v1.5"
    env.settings = dense_only
    with pytest.raises(SyncError, match="sync --full"):
        env.sync()


def test_summarize() -> None:
    from drive_agent.indexing.sync import SyncReport, summarize

    assert summarize(SyncReport(added=2, unchanged=5, seconds=1.5)) == (
        "+2 added · 0 updated · 5 unchanged · 0 deleted · 1.5s"
    )
    assert "(1 files removed)" in summarize(SyncReport(deleted=3, pruned_files=1))

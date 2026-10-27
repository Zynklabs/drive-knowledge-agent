"""``drive-agent sync``: load Drive files, embed new/changed chunks, prune stale ones.

A file whose Drive version matches the one it was indexed from (see ``file_version``) is
not downloaded again; its chunks are kept and counted as unchanged. Files that loaded fine
but had no text are remembered the same way, so they are not reloaded either.

Safety rules:
- a file that fails to fetch or parse keeps its previously indexed chunks;
- files that disappeared from Drive are pruned only when every folder was read successfully;
- changing the embedding model requires an explicit ``--full`` rebuild.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.indexing import index

from drive_agent.core.config import IngestSettings, Settings
from drive_agent.core.monitor import SyncProgress
from drive_agent.drive.models import DriveFile
from drive_agent.drive.walker import Progress
from drive_agent.indexing.loader import FileResult, FolderLoad, stream_folder
from drive_agent.indexing.state import STATE_DB, IndexedVersion, SQLiteRecordManager
from drive_agent.vectorstores.base import VectorIndex

UPSERT_BATCH = 64
"""Chunks per vector-store write: well under request size limits even with long chunks."""


class SyncError(Exception):
    """The sync cannot run safely with the current configuration."""


@dataclass
class SyncReport:
    folders: list[FolderLoad] = field(default_factory=list)
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    """Stale chunks removed (changed files, files that vanished or no longer yield text)."""
    pruned_files: int = 0
    new_files: int = 0
    """Files loaded for the first time (every file, after a full rebuild)."""
    changed_files: int = 0
    """Files loaded again because they changed in Drive since the last sync."""
    prune_skipped: bool = False
    """True when a folder failed, so vanished-file pruning was skipped for safety."""
    healed: bool = False
    """True when the vector store was found empty and the sync state was reset."""
    seconds: float = 0.0

    @property
    def files(self) -> list[FileResult]:
        return [result for load in self.folders for result in load.files]


def chunk_key(doc: Document) -> str:
    """Content+metadata hash formatted as a UUID: Qdrant point IDs must be UUIDs or integers."""
    payload = json.dumps(
        {"content": doc.page_content, "metadata": doc.metadata},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return str(uuid.UUID(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]))


def embedding_fingerprint(settings: Settings) -> str:
    """What the stored vectors were built with; any change requires `sync --full`."""
    suffix = "+bm25" if settings.retrieval.hybrid else ""
    return f"{settings.embeddings.provider}:{settings.embeddings.model}{suffix}"


def file_version(file: DriveFile, ingest: IngestSettings) -> str | None:
    """What a file's chunks were built from; a sync skips the file while this is unchanged.

    Content is identified by Drive's ``md5Checksum``, or ``modifiedTime`` for Google
    Docs/Slides, which have no md5. The metadata copied into every chunk and the chunking
    settings are hashed in too, so a rename, a move or a new chunk size still re-indexes.
    None (no md5 or modifiedTime) means the file is always reloaded.
    """
    content = file.md5 or file.modified_time
    if content is None:
        return None
    context = json.dumps(
        [
            file.name,
            file.path,
            file.mime_type,
            file.kind,
            file.web_url,
            file.owner,
            file.drive_id,
            file.root_folder_id,
            ingest.chunk_tokens,
            ingest.chunk_overlap,
        ],
        ensure_ascii=False,
    )
    return f"{content}:{hashlib.sha256(context.encode('utf-8')).hexdigest()[:16]}"


def _index_file(
    file: DriveFile,
    chunks: list[Document],
    records: SQLiteRecordManager,
    vector_index: VectorIndex,
    report: SyncReport,
    version: str | None,
) -> None:
    """Embed and save one file's new chunks and drop its stale ones, as soon as it is loaded.

    Saving file by file means a stopped sync keeps its progress and the next run skips what
    is already indexed. Chunks are written in batches of ``UPSERT_BATCH``, so one huge file
    stays under the vector store's request size limit (Qdrant: 32 MB). The file's stale
    chunks are removed here, once all its new ones are saved, rather than by ``index()``'s
    incremental cleanup: that runs per batch and would delete, then re-embed, the later
    chunks of any file spanning several batches. The file's version is recorded last, so a
    file stopped part-way is downloaded again.
    """
    result = index(
        chunks,
        records,
        vector_index.store,
        cleanup=None,
        source_id_key="file_id",
        key_encoder=chunk_key,
        batch_size=UPSERT_BATCH,
    )
    current = {chunk_key(chunk) for chunk in chunks}
    stale = [key for key in records.list_keys(group_ids=[file.id]) if key not in current]
    if stale:
        vector_index.store.delete(ids=stale)
        records.delete_keys(stale)
    _save_version(records, file, version, empty=False)
    report.added += result["num_added"]
    report.updated += result["num_updated"]
    report.unchanged += result["num_skipped"]
    report.deleted += len(stale)


def _save_version(
    records: SQLiteRecordManager, file: DriveFile, version: str | None, *, empty: bool
) -> None:
    if version is None:
        records.delete_file_versions([file.id])
    else:
        records.set_file_version(file.id, version, empty=empty)


def _keep_unchanged(file: DriveFile, records: SQLiteRecordManager, report: SyncReport) -> None:
    """Refresh a skipped file's records, so they read as seen by this sync, like re-indexed ones."""
    keys = records.list_keys(group_ids=[file.id])
    records.update(keys, group_ids=[file.id] * len(keys))
    report.unchanged += len(keys)


def run_sync(
    service: Any,
    settings: Settings,
    vector_index: VectorIndex,
    records: SQLiteRecordManager,
    *,
    full: bool = False,
    on_file: Callable[[FileResult], None] | None = None,
    on_scan: Progress | None = None,
    trigger: str = "manual",
) -> SyncReport:
    # Read by the admin portal, which also lists past syncs from what ``finish`` records.
    progress = SyncProgress(settings.data_dir, full=full, trigger=trigger)
    try:
        report = _run_sync(
            service, settings, vector_index, records, progress, full, on_file, on_scan
        )
    except Exception as exc:
        progress.finish(ok=False, error=str(exc) or type(exc).__name__)
        raise
    except BaseException:  # Ctrl+C or a service stop: progress is saved, so say so plainly
        progress.finish(ok=False, error="stopped")
        raise
    folder_errors = [load.error for load in report.folders if load.error]
    progress.finish(
        ok=not folder_errors, error="; ".join(folder_errors) or None, files=_file_counts(report)
    )
    return report


def _file_counts(report: SyncReport) -> dict[str, int]:
    return {
        "new_files": report.new_files,
        "changed_files": report.changed_files,
        "removed_files": report.pruned_files,
        "chunks_removed": report.deleted,
    }


def _run_sync(
    service: Any,
    settings: Settings,
    vector_index: VectorIndex,
    records: SQLiteRecordManager,
    progress: SyncProgress,
    full: bool,
    on_file: Callable[[FileResult], None] | None,
    on_scan: Progress | None,
) -> SyncReport:
    started = time.monotonic()
    report = SyncReport()
    fingerprint = embedding_fingerprint(settings)
    previous = records.get_meta("embedding")

    if full:
        vector_index.reset()
        records.clear()
    elif previous is not None and previous != fingerprint:
        raise SyncError(
            f"The index was built with embeddings {previous!r} but the config now says "
            f"{fingerprint!r}. Run `drive-agent sync --full` to rebuild it."
        )
    elif records.count() and vector_index.count() == 0:
        records.clear()  # the collection was dropped outside the agent: start over
        report.healed = True

    indexed = records.file_versions()  # empty after a --full or healing reset

    def is_current(file: DriveFile) -> bool:
        version = file_version(file, settings.ingest)
        previous = indexed.get(file.id)
        return version is not None and previous is not None and previous.version == version

    for folder_id in settings.drive.folder_ids:
        load = FolderLoad(folder_id=folder_id)
        report.folders.append(load)
        for loaded in stream_folder(
            service,
            settings,
            load,
            on_file=on_file,
            on_scan=on_scan,
            on_listed=progress.listed,
            is_current=is_current,
        ):
            added_before = report.added
            version = file_version(loaded.file, settings.ingest)
            if not loaded.unchanged and loaded.error is None:
                if loaded.file.id in indexed:
                    report.changed_files += 1
                else:
                    report.new_files += 1
            if loaded.unchanged:
                if not indexed[loaded.file.id].empty:
                    _keep_unchanged(loaded.file, records, report)
            elif loaded.chunks:
                _index_file(loaded.file, loaded.chunks, records, vector_index, report, version)
            elif loaded.error is None:  # loaded fine but has no text: skip it next time too
                _save_version(records, loaded.file, version, empty=True)
            progress.file_done(
                failed=loaded.error is not None,
                empty=_is_empty(loaded, indexed),
                chunks_added=report.added - added_before,
            )

    if any(load.error for load in report.folders):
        report.prune_skipped = True
    else:
        keep = {r.file.id for r in report.files if r.error or _has_chunks(r, indexed)}
        for group in records.list_groups():
            if group in keep:
                continue
            keys = records.list_keys(group_ids=[group])
            vector_index.store.delete(ids=keys)
            records.delete_keys(keys)
            report.deleted += len(keys)
            report.pruned_files += 1
        # Files now empty keep their (empty) version; only files gone from Drive are forgotten.
        listed = {r.file.id for r in report.files}
        records.delete_file_versions([f for f in records.file_version_ids() if f not in listed])

    report.seconds = round(time.monotonic() - started, 1)
    records.set_meta("embedding", fingerprint)
    records.set_meta(
        "last_sync",
        {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "files": len({r.file.id for r in report.files if _has_chunks(r, indexed)}),
            "chunks": records.count(),
            "failed": sum(1 for r in report.files if r.error),
            "total_files": len(report.files),
            "empty": sum(1 for r in report.files if _is_empty(r, indexed)),
            "skipped": sum(len(load.skipped) for load in report.folders),
        },
    )
    return report


def _has_chunks(result: FileResult, indexed: dict[str, IndexedVersion]) -> bool:
    """Loaded with text this sync, or skipped as unchanged with its chunks kept."""
    if result.unchanged:
        return not indexed[result.file.id].empty
    return bool(result.chunks)


def _is_empty(result: FileResult, indexed: dict[str, IndexedVersion]) -> bool:
    return result.error is None and not _has_chunks(result, indexed)


def sync_now(
    settings: Settings,
    vector_index: VectorIndex,
    *,
    full: bool = False,
    on_file: Callable[[FileResult], None] | None = None,
    on_scan: Progress | None = None,
    trigger: str = "manual",
) -> SyncReport:
    """Connect to Drive (non-interactively) and run one sync. Used by the web app and Slack.

    Raises ``DriveAuthError`` or ``SyncError``.
    """
    from drive_agent.drive.auth import connect  # deferred: Google client import is slow

    _auth, service = connect(settings.drive, settings.data_dir, interactive=False)
    records = SQLiteRecordManager(vector_index.namespace, settings.data_dir / STATE_DB)
    try:
        return run_sync(
            service,
            settings,
            vector_index,
            records,
            full=full,
            on_file=on_file,
            on_scan=on_scan,
            trigger=trigger,
        )
    finally:
        records.close()


def summarize(report: SyncReport) -> str:
    text = (
        f"+{report.added} added · {report.updated} updated · {report.unchanged} unchanged · "
        f"{report.deleted} deleted"
    )
    if report.pruned_files:
        text += f" ({report.pruned_files} files removed)"
    failed = sum(1 for r in report.files if r.error)
    if failed:
        text += f" · {failed} file(s) failed"
    return text + f" · {report.seconds}s"

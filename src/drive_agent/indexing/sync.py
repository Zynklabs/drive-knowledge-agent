"""``drive-agent sync``: load Drive files, embed new/changed chunks, prune stale ones.

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
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.indexing import index

from drive_agent.core.config import Settings
from drive_agent.indexing.loader import FileResult, FolderLoad, load_folder
from drive_agent.indexing.state import SQLiteRecordManager
from drive_agent.vectorstores.base import VectorIndex


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


def run_sync(
    service: Any,
    settings: Settings,
    vector_index: VectorIndex,
    records: SQLiteRecordManager,
    *,
    full: bool = False,
    on_file: Callable[[FileResult], None] | None = None,
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

    def documents() -> Iterator[Document]:
        for folder_id in settings.drive.folder_ids:
            load = load_folder(service, settings, folder_id, on_file=on_file)
            report.folders.append(load)
            for result in load.files:
                yield from result.chunks

    result = index(
        documents(),
        records,
        vector_index.store,
        cleanup="incremental",
        source_id_key="file_id",
        key_encoder=chunk_key,
        batch_size=64,
    )
    report.added = result["num_added"]
    report.updated = result["num_updated"]
    report.unchanged = result["num_skipped"]
    report.deleted = result["num_deleted"]

    if any(load.error for load in report.folders):
        report.prune_skipped = True
    else:
        keep = {r.file.id for r in report.files if r.chunks or r.error}
        for group in records.list_groups():
            if group in keep:
                continue
            keys = records.list_keys(group_ids=[group])
            vector_index.store.delete(ids=keys)
            records.delete_keys(keys)
            report.deleted += len(keys)
            report.pruned_files += 1

    report.seconds = round(time.monotonic() - started, 1)
    records.set_meta("embedding", fingerprint)
    records.set_meta(
        "last_sync",
        {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "files": len({r.file.id for r in report.files if r.chunks}),
            "chunks": records.count(),
            "failed": sum(1 for r in report.files if r.error),
        },
    )
    return report

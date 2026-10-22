"""Walk → fetch → parse → chunk. One bad file never stops the run."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError
from langchain_core.documents import Document

from drive_agent.chunking import approx_tokens, chunk_document
from drive_agent.core.config import IngestSettings, Settings
from drive_agent.drive.errors import explain
from drive_agent.drive.fetcher import DriveFetcher, FileTooLargeError
from drive_agent.drive.models import DriveFile, DriveFolder, SkippedFile
from drive_agent.drive.walker import DriveWalker, NotAFolderError, Progress
from drive_agent.parsing import ParseError, parse


@dataclass
class FileResult:
    file: DriveFile
    chunks: list[Document] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    unchanged: bool = False
    """Same Drive version as when last indexed, so not downloaded; ``chunks`` stays empty."""

    @property
    def tokens(self) -> int:
        return sum(approx_tokens(chunk.page_content) for chunk in self.chunks)


@dataclass
class FolderLoad:
    folder_id: str
    root: DriveFolder | None = None
    error: str | None = None
    """Set when the folder itself could not be read; per-file problems live on FileResult."""
    files: list[FileResult] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)


def load_file(fetcher: DriveFetcher, file: DriveFile, ingest: IngestSettings) -> FileResult:
    result = FileResult(file=file)
    try:
        content = fetcher.fetch(file)
        parsed = parse(content.data, content.format)
    except (FileTooLargeError, ParseError) as exc:
        result.error = str(exc)
        return result
    except (HttpError, RefreshError) as exc:
        result.error = explain(exc)
        return result

    result.warnings.extend(parsed.warnings)
    result.chunks = chunk_document(file, parsed, ingest)
    if not result.chunks and not result.warnings:
        result.warnings.append("no text found")
    return result


def stream_folder(
    service: Any,
    settings: Settings,
    load: FolderLoad,
    *,
    on_file: Callable[[FileResult], None] | None = None,
    on_scan: Progress | None = None,
    on_listed: Callable[[int], None] | None = None,
    is_current: Callable[[DriveFile], bool] | None = None,
) -> Iterator[FileResult]:
    """Yield each file as soon as it is loaded, recording results and errors on ``load``.

    The whole tree is listed before any download (listing is fast; downloads are not), so
    ``on_listed`` gets the number of files to load up front and progress can show "n of N".
    Files for which ``is_current`` returns True are yielded as ``unchanged`` without being
    downloaded or parsed.
    """
    walker = DriveWalker(service, settings.drive, on_progress=on_scan)
    fetcher = DriveFetcher(service, max_bytes=settings.drive.max_file_mb * 1024 * 1024)
    try:
        load.root = walker.resolve_root(load.folder_id)
        files: list[DriveFile] = []
        for item in walker.walk(load.root):
            if isinstance(item, SkippedFile):
                load.skipped.append(item)
            else:
                files.append(item)
        if on_listed is not None:
            on_listed(len(files))
        for item in files:
            if is_current is not None and is_current(item):
                result = FileResult(file=item, unchanged=True)
            else:
                result = load_file(fetcher, item, settings.ingest)
            load.files.append(result)
            if on_file is not None:
                on_file(result)
            yield result
    except NotAFolderError as exc:
        load.error = str(exc)
    except (HttpError, RefreshError) as exc:
        load.error = explain(exc, folder_id=load.folder_id)


def load_folder(
    service: Any,
    settings: Settings,
    folder_id: str,
    *,
    on_file: Callable[[FileResult], None] | None = None,
    on_scan: Progress | None = None,
) -> FolderLoad:
    load = FolderLoad(folder_id=folder_id)
    for _ in stream_folder(service, settings, load, on_file=on_file, on_scan=on_scan):
        pass
    return load

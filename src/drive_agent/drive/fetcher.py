"""Download file content from Drive. Google-native files are exported; others are downloaded."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Literal

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from drive_agent.core.config import FileKind
from drive_agent.drive.models import DriveFile

ContentFormat = Literal["md", "txt", "html", "json", "pdf", "docx", "pptx"]

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

# Google-native kinds: (export MIME type, resulting format).
EXPORTS: dict[FileKind, tuple[str, ContentFormat]] = {
    "docs": ("text/markdown", "md"),  # keeps headings
    "slides": (PPTX_MIME, "pptx"),  # keeps slide boundaries and speaker notes
}
DOWNLOADS: dict[FileKind, ContentFormat] = {
    "pdf": "pdf",
    "docx": "docx",
    "pptx": "pptx",
    "txt": "txt",
    "md": "md",
    "html": "html",
    "json": "json",
}
_CHUNK_BYTES = 8 * 1024 * 1024


class FileTooLargeError(Exception):
    """Content exceeded ``max_file_mb`` while downloading."""


@dataclass(frozen=True, slots=True)
class FetchedContent:
    data: bytes
    format: ContentFormat


class DriveFetcher:
    """Fetches file bytes. Read-only: uses only ``files.export_media`` and ``files.get_media``."""

    def __init__(self, service: Any, *, max_bytes: int, num_retries: int = 5) -> None:
        self._files = service.files()
        self._max_bytes = max_bytes
        self._retries = num_retries

    def fetch(self, file: DriveFile) -> FetchedContent:
        if file.kind in EXPORTS:
            mime, fmt = EXPORTS[file.kind]
            try:
                return FetchedContent(self._export(file.id, mime), fmt)
            except HttpError as exc:
                # Older Workspace tenants may not offer Markdown export: fall back to DOCX.
                if file.kind == "docs" and exc.resp.status == 400:
                    return FetchedContent(self._export(file.id, DOCX_MIME), "docx")
                raise
        request = self._files.get_media(fileId=file.id, supportsAllDrives=True)
        return FetchedContent(self._download(request), DOWNLOADS[file.kind])

    def _export(self, file_id: str, mime_type: str) -> bytes:
        return self._download(self._files.export_media(fileId=file_id, mimeType=mime_type))

    def _download(self, request: Any) -> bytes:
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request, chunksize=_CHUNK_BYTES)
        done = False
        while not done:
            _status, done = downloader.next_chunk(num_retries=self._retries)
            if buffer.tell() > self._max_bytes:
                raise FileTooLargeError(
                    f"larger than max_file_mb ({self._max_bytes // (1024 * 1024)} MB)"
                )
        return buffer.getvalue()

"""Exercise the real googleapiclient download path against canned HTTP responses."""

import json
from typing import Any

import pytest
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import HttpMockSequence

from drive_agent.drive.fetcher import DOCX_MIME, PPTX_MIME, DriveFetcher, FileTooLargeError
from drive_agent.drive.models import DriveFile


class RecordingHttp(HttpMockSequence):  # type: ignore[misc]
    def __init__(self, responses: list[tuple[dict[str, str], bytes]]) -> None:
        super().__init__(responses)
        self.uris: list[str] = []

    def request(self, uri: str, *args: Any, **kwargs: Any) -> Any:
        self.uris.append(uri)
        return super().request(uri, *args, **kwargs)


def ok(body: bytes) -> tuple[dict[str, str], bytes]:
    return {"status": "200", "content-length": str(len(body))}, body


def error(status: int, reason: str) -> tuple[dict[str, str], bytes]:
    body = {"error": {"code": status, "message": reason, "errors": [{"reason": reason}]}}
    return {"status": str(status)}, json.dumps(body).encode()


def fetcher(
    responses: list[tuple[dict[str, str], bytes]], max_bytes: int = 10_000
) -> tuple[DriveFetcher, RecordingHttp]:
    http = RecordingHttp(responses)
    service = build("drive", "v3", http=http, static_discovery=True, cache_discovery=False)
    return DriveFetcher(service, max_bytes=max_bytes, num_retries=0), http


def drive_file(kind: str, mime: str = "x") -> DriveFile:
    return DriveFile(id="F1", name="f", mime_type=mime, kind=kind, path="R/f", root_folder_id="R")  # type: ignore[arg-type]


def test_google_doc_exported_as_markdown() -> None:
    f, http = fetcher([ok(b"# Title\n\nBody")])
    content = f.fetch(drive_file("docs"))
    assert (content.data, content.format) == (b"# Title\n\nBody", "md")
    assert "files/F1/export" in http.uris[0]
    assert "mimeType=text%2Fmarkdown" in http.uris[0]


def test_google_doc_falls_back_to_docx_when_markdown_unsupported() -> None:
    f, http = fetcher([error(400, "badRequest"), ok(b"PK-docx")])
    content = f.fetch(drive_file("docs"))
    assert (content.data, content.format) == (b"PK-docx", "docx")
    assert DOCX_MIME.replace("/", "%2F") in http.uris[1]


def test_google_slides_exported_as_pptx() -> None:
    f, http = fetcher([ok(b"PK-pptx")])
    assert f.fetch(drive_file("slides")).format == "pptx"
    assert PPTX_MIME.replace("/", "%2F") in http.uris[0]


def test_binary_file_downloaded() -> None:
    f, http = fetcher([ok(b"%PDF-1.4 ...")])
    content = f.fetch(drive_file("pdf", "application/pdf"))
    assert (content.data, content.format) == (b"%PDF-1.4 ...", "pdf")
    assert "files/F1?" in http.uris[0]
    assert "alt=media" in http.uris[0]
    assert "supportsAllDrives=true" in http.uris[0]


def test_too_large_download_aborts() -> None:
    f, _ = fetcher([ok(b"x" * 2_000)], max_bytes=1_000)
    with pytest.raises(FileTooLargeError, match="max_file_mb"):
        f.fetch(drive_file("txt", "text/plain"))


def test_other_errors_propagate() -> None:
    f, _ = fetcher([error(403, "exportSizeLimitExceeded")])
    with pytest.raises(HttpError):
        f.fetch(drive_file("docs"))

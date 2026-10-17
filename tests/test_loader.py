from typing import Any

import pytest

from drive_agent.core.config import Settings
from drive_agent.drive.fetcher import DriveFetcher, FileTooLargeError
from drive_agent.indexing.loader import FileResult, load_folder

from .doc_factory import make_pdf
from .fake_drive import FakeDrive, fake_download, file, folder, http_error

DOC = "application/vnd.google-apps.document"
PDF = "application/pdf"


@pytest.fixture(autouse=True)
def _fake_media(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DriveFetcher, "_download", fake_download)


def test_load_folder_end_to_end() -> None:
    drive = FakeDrive(
        [
            folder("root", "Hiring"),
            file("doc", "Process", DOC, "root"),
            file("cv", "Jane.pdf", PDF, "root", size=100),
            file("bad", "Broken.pdf", PDF, "root", size=10),
            file("scan", "Scan.pdf", PDF, "root", size=10),
            file("deny", "Private.pdf", PDF, "root", size=10),
            file("big", "Huge export", DOC, "root"),
            file("sheet", "Budget", "application/vnd.google-apps.spreadsheet", "root"),
        ],
        contents={
            "doc": b"# Stages\n\nPhone screen, then onsite.",
            "cv": make_pdf(["Jane Doe", "Python, Go"]),
            "bad": b"not a pdf",
            "scan": make_pdf([""]),
            "deny": http_error(403, "forbidden", "no"),
            "big": FileTooLargeError("larger than max_file_mb (50 MB)"),
        },
    )
    seen: list[FileResult] = []
    load = load_folder(drive, Settings(), "root", on_file=seen.append)

    assert load.error is None
    assert load.root is not None
    assert load.root.name == "Hiring"
    assert [s.id for s in load.skipped] == ["sheet"]
    by_id = {r.file.id: r for r in load.files}
    assert [r.file.id for r in seen] == [r.file.id for r in load.files]

    doc = by_id["doc"]
    assert [c.page_content for c in doc.chunks] == [
        "Process > Stages\n\nPhone screen, then onsite."
    ]
    assert ("doc", "text/markdown") in drive.media_calls

    cv = by_id["cv"]
    assert [c.metadata["page"] for c in cv.chunks] == [1, 2]
    assert cv.tokens > 0
    assert cv.error is None

    assert by_id["bad"].error is not None
    assert by_id["bad"].chunks == []
    assert "scanned PDF" in by_id["scan"].warnings[0]
    assert "has no access" in (by_id["deny"].error or "")
    assert "max_file_mb" in (by_id["big"].error or "")


def test_empty_text_file_warns() -> None:
    drive = FakeDrive(
        [folder("root", "R"), file("t", "empty.txt", "text/plain", "root", size=0)],
        contents={"t": b"   "},
    )
    result = load_folder(drive, Settings(), "root").files[0]
    assert result.chunks == []
    assert result.warnings == ["no text found"]


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([], "not shared with"),
        ([file("f", "A doc", DOC, "x")], "is a file"),
    ],
)
def test_folder_level_errors(items: list[dict[str, Any]], expected: str) -> None:
    load = load_folder(FakeDrive(items), Settings(), "f" if items else "missing")
    assert load.root is None
    assert expected in (load.error or "")

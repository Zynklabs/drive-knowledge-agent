from typing import Any

import pytest

from drive_agent.core.config import DriveSettings
from drive_agent.drive.mime import FOLDER, file_kind, is_out_of_scope
from drive_agent.drive.models import DriveFile
from drive_agent.drive.walker import DriveWalker, NotAFolderError

from .fake_drive import FakeDrive, file, folder, http_error, shortcut

DOC = "application/vnd.google-apps.document"
SLIDES = "application/vnd.google-apps.presentation"
SHEET = "application/vnd.google-apps.spreadsheet"
PDF = "application/pdf"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
OCTET = "application/octet-stream"


def walk(items: list[dict[str, Any]], root: str = "root", **settings: Any) -> tuple[Any, FakeDrive]:
    page_size = settings.pop("page", 1000)
    drive = FakeDrive(items)
    walker = DriveWalker(drive, DriveSettings(**settings), page_size=page_size)
    return walker.scan(walker.resolve_root(root)), drive


def paths(report: Any) -> set[str]:
    return {f.path for f in report.files}


# --- mime classification ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "mime", "kind"),
    [
        ("Policy", DOC, "docs"),
        ("Deck", SLIDES, "slides"),
        ("a.pdf", PDF, "pdf"),
        ("notes.md", OCTET, "md"),
        ("README.MARKDOWN", OCTET, "md"),
        ("page.htm", OCTET, "html"),
        ("x.md", "text/markdown", "md"),
        ("photo.png", "image/png", None),
        ("Form.pdf", "application/vnd.google-apps.form", None),  # never guess Google types
    ],
)
def test_file_kind(name: str, mime: str, kind: str | None) -> None:
    assert file_kind(name, mime) == kind


@pytest.mark.parametrize(
    ("name", "mime"),
    [
        ("Budget", SHEET),
        ("q.xlsx", XLSX),
        ("old.xls", OCTET),
        ("d.csv", "text/csv"),
        ("d.tsv", OCTET),
    ],
)
def test_spreadsheets_out_of_scope(name: str, mime: str) -> None:
    assert is_out_of_scope(name, mime)


# --- walking ----------------------------------------------------------------


def test_recursive_walk_with_paths_and_metadata() -> None:
    report, _ = walk(
        [
            folder("root", "Company"),
            file("d1", "Handbook", DOC, "root", webViewLink="https://docs/d1", modifiedTime="t1"),
            folder("hr", "HR", "root"),
            folder("pol", "Policies", "hr"),
            file(
                "p1",
                "Leave.pdf",
                PDF,
                "pol",
                size=1000,
                md5Checksum="abc",
                owners=[{"emailAddress": "hr@example.com"}],
            ),
            file("gone", "Old", DOC, "root", trashed=True),
        ]
    )
    assert report.root.name == "Company"
    assert paths(report) == {"Company/Handbook", "Company/HR/Policies/Leave.pdf"}
    pdf = next(f for f in report.files if f.id == "p1")
    assert pdf == DriveFile(
        id="p1",
        name="Leave.pdf",
        mime_type=PDF,
        kind="pdf",
        path="Company/HR/Policies/Leave.pdf",
        root_folder_id="root",
        size=1000,
        md5="abc",
        owner="hr@example.com",
    )
    doc = next(f for f in report.files if f.id == "d1")
    assert doc.web_url == "https://docs/d1"
    assert doc.size is None  # Google-native files have no size
    assert report.stats.folders == 2


def test_pagination() -> None:
    items = [folder("root", "R")] + [file(f"f{i}", f"doc{i}", DOC, "root") for i in range(7)]
    report, drive = walk(items, page=3)
    assert len(report.files) == 7
    assert len(drive.list_calls) == 3
    assert drive.list_calls[1]["pageToken"] == "3"
    assert all(call["corpora"] == "allDrives" for call in drive.list_calls)


def test_skip_reasons() -> None:
    report, _ = walk(
        [
            folder("root", "R"),
            file("s", "Budget", SHEET, "root"),
            file("x", "q.xlsx", XLSX, "root"),
            file("c", "data.csv", "text/csv", "root"),
            file("img", "logo.png", "image/png", "root"),
            file("big", "huge.pdf", PDF, "root", size=3 * 1024 * 1024),
            file("md", "notes.md", OCTET, "root", size=10),
            file("dx", "x.docx", OCTET, "root"),
        ],
        max_file_mb=2,
        include_mime=["pdf", "md"],
    )
    assert {f.id for f in report.files} == {"md"}
    reasons = {s.id: s.reason for s in report.skipped}
    assert reasons == {
        "s": "out_of_scope",
        "x": "out_of_scope",
        "c": "out_of_scope",
        "img": "unsupported",
        "big": "too_large",
        "dx": "excluded_type",
    }
    big = next(s for s in report.skipped if s.id == "big")
    assert big.detail == "3.0 MB"
    assert big.path == "R/huge.pdf"


def test_shortcuts_to_files_and_folders() -> None:
    report, _ = walk(
        [
            folder("root", "R"),
            folder("elsewhere", "Elsewhere"),  # outside the tree, reached via shortcut
            file("e1", "Linked doc", DOC, "elsewhere"),
            file("t", "Target", DOC, "elsewhere"),
            shortcut("sc1", "Shortcut to Target", "root", "t", DOC),
            shortcut("sc2", "Linked folder", "root", "elsewhere", FOLDER),
        ]
    )
    by_id = {f.id: f for f in report.files}
    assert by_id["t"].via_shortcut == "sc1"
    assert by_id["t"].path == "R/Target"
    assert by_id["e1"].path == "R/Linked folder/Linked doc"
    assert report.stats.shortcuts == 2
    # "t" is reachable both via sc1 and inside the linked folder: indexed once
    assert len(report.files) == 2
    assert report.stats.revisits == 1


def test_shortcut_cycle_terminates() -> None:
    report, _ = walk(
        [
            folder("root", "R"),
            folder("a", "A", "root"),
            shortcut("loop", "Back to root", "a", "root", FOLDER),
            file("f", "Doc", DOC, "a"),
        ]
    )
    assert paths(report) == {"R/A/Doc"}
    assert report.stats.revisits == 1


def test_broken_and_inaccessible_shortcuts_are_skipped() -> None:
    report, _ = walk(
        [
            folder("root", "R"),
            {**shortcut("b", "Broken", "root", "", DOC), "shortcutDetails": {}},
            shortcut("p", "Private", "root", "secret", DOC),  # target not visible -> 404
        ]
    )
    assert report.files == []
    assert {(s.id, s.detail) for s in report.skipped} == {
        ("b", "broken shortcut"),
        ("p", "shortcut target is not accessible"),
    }


def test_inaccessible_subfolder_is_counted_not_fatal() -> None:
    drive = FakeDrive(
        [
            folder("root", "R"),
            shortcut("sf", "Not shared", "root", "private", FOLDER),
            file("ok", "Doc", DOC, "root"),
        ],
        list_errors={"private": http_error(404, "notFound")},
    )
    walker = DriveWalker(drive, DriveSettings())
    report = walker.scan(walker.resolve_root("root"))
    assert [f.id for f in report.files] == ["ok"]
    assert report.stats.inaccessible_folders == 1


def test_root_listing_errors_propagate() -> None:
    drive = FakeDrive([folder("root", "R")], list_errors={"root": http_error(403, "forbidden")})
    walker = DriveWalker(drive, DriveSettings())
    root = walker.resolve_root("root")
    with pytest.raises(Exception, match="403"):
        walker.scan(root)


def test_max_depth() -> None:
    items = [
        folder("root", "R"),
        file("f0", "top", DOC, "root"),
        folder("l1", "L1", "root"),
        file("f1", "one", DOC, "l1"),
        folder("l2", "L2", "l1"),
        file("f2", "two", DOC, "l2"),
    ]
    report, _ = walk(items, max_depth=1)
    assert paths(report) == {"R/top", "R/L1/one"}
    assert report.stats.depth_limited == 1

    report, _ = walk(items, max_depth=0)
    assert paths(report) == {"R/top"}


def test_shared_drive_root_and_edit_capability() -> None:
    report, _ = walk(
        [
            folder("sd", "Engineering", driveId="sd", capabilities={"canEdit": True}),
            file("f", "Doc", DOC, "sd", driveId="sd"),
        ],
        root="sd",
    )
    assert report.root.drive_id == "sd"
    assert report.root.can_edit is True
    assert report.files[0].drive_id == "sd"


def test_incomplete_search_flag() -> None:
    drive = FakeDrive([folder("root", "R")], incomplete=True)
    walker = DriveWalker(drive, DriveSettings())
    assert walker.scan(walker.resolve_root("root")).stats.incomplete


def test_root_must_be_a_folder() -> None:
    drive = FakeDrive([file("f", "Doc", DOC, "x")])
    with pytest.raises(NotAFolderError, match="is a file"):
        DriveWalker(drive, DriveSettings()).resolve_root("f")


def test_root_shortcut_to_folder_resolves() -> None:
    drive = FakeDrive([folder("real", "Real"), shortcut("sc", "Link", "x", "real", FOLDER)])
    assert DriveWalker(drive, DriveSettings()).resolve_root("sc").id == "real"


def test_root_shortcut_to_file_rejected() -> None:
    drive = FakeDrive([shortcut("sc", "Link", "x", "doc", DOC)])
    with pytest.raises(NotAFolderError, match="shortcut to a file"):
        DriveWalker(drive, DriveSettings()).resolve_root("sc")

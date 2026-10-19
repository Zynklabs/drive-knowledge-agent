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


# --- fast listing -------------------------------------------------------------


def _tree(subfolders: int, drive_id: str | None = None) -> list[dict[str, Any]]:
    extra = {"driveId": drive_id} if drive_id else {}
    root_id = drive_id or "root"
    items = [folder(root_id, "Root", **extra)]
    for i in range(subfolders):
        items.append(folder(f"sub{i}", f"Sub{i}", root_id, **extra))
        items.append(file(f"doc{i}", f"Doc{i}", DOC, f"sub{i}", **extra))
        items.append(folder(f"deep{i}", f"Deep{i}", f"sub{i}", **extra))
        items.append(file(f"deepdoc{i}", f"DeepDoc{i}", DOC, f"deep{i}", **extra))
    return items


def test_parallel_listing_matches_sequential_and_never_combines_folders() -> None:
    sequential, parallel = FakeDrive(_tree(30)), FakeDrive(_tree(30))
    w1 = DriveWalker(sequential, DriveSettings(), workers=1)
    w8 = DriveWalker(parallel, DriveSettings(), workers=8)
    r1 = w1.scan(w1.resolve_root("root"))
    r8 = w8.scan(w8.resolve_root("root"))
    assert [(f.id, f.path) for f in r8.files] == [(f.id, f.path) for f in r1.files]  # same order
    assert len(r8.files) == 60
    # One query per folder (root + 60): Drive drops results from "'a' in parents or ..." queries.
    assert len(parallel.list_calls) == 61
    assert not any(" or " in call["q"] for call in parallel.list_calls)


def test_whole_shared_drive_is_listed_flat() -> None:
    drive = FakeDrive(_tree(30, drive_id="0Adrive"))
    walker = DriveWalker(drive, DriveSettings(), page_size=50)
    report = walker.scan(walker.resolve_root("0Adrive"))
    assert len(report.files) == 60
    assert {c["corpora"] for c in drive.list_calls} == {"drive"}
    assert all(c["driveId"] == "0Adrive" for c in drive.list_calls)
    assert len(drive.list_calls) == 3  # 120 items in pages of 50, nothing per folder
    assert report.stats.folders == 60
    assert "Root/Sub7/Deep7/DeepDoc7" in {f.path for f in report.files}


def test_flat_listing_respects_depth_and_follows_shortcuts_out_of_the_drive() -> None:
    items = _tree(2, drive_id="0Adrive")
    items += [
        folder("outside", "Outside"),  # in someone's My Drive
        file("o1", "OutsideDoc", DOC, "outside"),
        shortcut("sc", "Link out", "0Adrive", "outside", FOLDER),
    ]
    items[-1]["driveId"] = "0Adrive"
    drive = FakeDrive(items)
    walker = DriveWalker(drive, DriveSettings(max_depth=1))
    report = walker.scan(walker.resolve_root("0Adrive"))
    assert {f.path for f in report.files} == {
        "Root/Sub0/Doc0",
        "Root/Sub1/Doc1",
        "Root/Link out/OutsideDoc",
    }
    assert report.stats.depth_limited == 2  # Deep0, Deep1
    outside_calls = [c for c in drive.list_calls if c["corpora"] == "allDrives"]
    assert len(outside_calls) == 1  # only the shortcut target needed its own query


def test_one_unreadable_folder_does_not_hide_others() -> None:
    drive = FakeDrive(
        [
            folder("root", "R"),
            folder("a", "A", "root"),
            file("fa", "DocA", DOC, "a"),
            shortcut("sc", "Private", "root", "private", FOLDER),
            folder("b", "B", "root"),
            file("fb", "DocB", DOC, "b"),
        ],
        list_errors={"private": http_error(404, "notFound")},
    )
    walker = DriveWalker(drive, DriveSettings())
    report = walker.scan(walker.resolve_root("root"))
    assert {f.id for f in report.files} == {"fa", "fb"}
    assert report.stats.inaccessible_folders == 1


def test_progress_is_reported() -> None:
    seen: list[tuple[int, int]] = []
    walker = DriveWalker(
        FakeDrive(_tree(30)),
        DriveSettings(),
        on_progress=lambda f, i: seen.append((f, i)),
    )
    walker.scan(walker.resolve_root("root"))
    assert [folders for folders, _ in seen] == list(range(1, 62))  # root + 30 + 30 folders
    assert seen[-1][1] == 120  # every listed item counted


def test_shared_drive_root_uses_the_drive_name() -> None:
    items = _tree(1, drive_id="0Adrive")
    named = FakeDrive(items, drive_names={"0Adrive": "Compliance"})
    walker = DriveWalker(named, DriveSettings())
    report = walker.scan(walker.resolve_root("0Adrive"))
    assert report.root.name == "Compliance"
    assert "Compliance/Sub0/Doc0" in {f.path for f in report.files}
    unnamed = DriveWalker(FakeDrive(items), DriveSettings())  # drives.get fails: keep folder name
    assert unnamed.resolve_root("0Adrive").name == "Root"

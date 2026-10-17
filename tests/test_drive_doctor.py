from pathlib import Path
from typing import Any

import pytest
from google.auth.exceptions import RefreshError

from drive_agent.core.config import Settings
from drive_agent.drive.doctor import run_check

from .conftest import SA_EMAIL
from .fake_drive import FakeDrive, file, folder, http_error

DOC = "application/vnd.google-apps.document"


def settings(key: Path | None, folders: list[str], **drive: Any) -> Settings:
    return Settings.model_validate(
        {"drive": {"credentials_file": key, "folder_ids": folders, **drive}}
    )


def check(s: Settings, drive: FakeDrive) -> Any:
    return run_check(s, service_factory=lambda creds: drive, interactive=False)


def test_all_good(sa_key_file: Path) -> None:
    drive = FakeDrive(
        [folder("root", "Company"), file("d", "Doc", DOC, "root")], user_email=SA_EMAIL
    )
    result = check(settings(sa_key_file, ["root"]), drive)
    assert result.ok
    assert result.identity == SA_EMAIL
    assert result.folders[0].report is not None
    assert result.folders[0].warnings == []


def test_no_folders_configured(sa_key_file: Path) -> None:
    result = check(settings(sa_key_file, []), FakeDrive([]))
    assert not result.ok
    assert "No Drive folders configured" in (result.error or "")


def test_auth_failure_is_reported(tmp_path: Path) -> None:
    result = check(settings(tmp_path / "missing.json", ["root"]), FakeDrive([]))
    assert not result.ok
    assert result.description is None
    assert "not found" in (result.error or "")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (http_error(403, "accessNotConfigured"), "not enabled"),
        (RefreshError("unauthorized_client"), "Domain-wide delegation"),
    ],
)
def test_first_api_call_failure(sa_key_file: Path, error: Exception, expected: str) -> None:
    result = check(settings(sa_key_file, ["root"]), FakeDrive([], about_error=error))
    assert not result.ok
    assert result.description is not None
    assert expected in (result.error or "")
    assert result.folders == []


def test_one_bad_folder_does_not_hide_others(sa_key_file: Path) -> None:
    drive = FakeDrive(
        [folder("good", "Good"), file("d", "Doc", DOC, "good"), file("f", "AFile", DOC, "x")]
    )
    result = check(settings(sa_key_file, ["good", "missing", "f"]), drive)
    assert not result.ok
    good, missing, not_folder = result.folders
    assert good.ok
    assert "not shared with" in (missing.error or "")
    assert "is a file" in (not_folder.error or "")


def test_warnings(sa_key_file: Path) -> None:
    drive = FakeDrive(
        [folder("root", "R", capabilities={"canEdit": True})],
        incomplete=True,
    )
    result = check(settings(sa_key_file, ["root"]), drive)
    assert result.ok  # warnings do not fail the check
    warnings = " ".join(result.folders[0].warnings)
    assert "Viewer access is enough" in warnings
    assert "incomplete listing" in warnings
    assert "No indexable files" in warnings

"""``drive-agent check``: verify Drive access end to end before indexing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from google.auth.credentials import Credentials
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

from drive_agent.core.config import Settings
from drive_agent.drive import auth
from drive_agent.drive.errors import explain
from drive_agent.drive.models import WalkReport
from drive_agent.drive.walker import DriveWalker, NotAFolderError


@dataclass
class FolderCheck:
    folder_id: str
    report: WalkReport | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class CheckResult:
    mode: str
    identity: str | None = None
    description: str | None = None
    error: str | None = None
    """Set when authentication or the first API call fails; folders are then not checked."""
    folders: list[FolderCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.folders) and all(f.ok for f in self.folders)


ServiceFactory = Callable[[Credentials], Any]


def run_check(
    settings: Settings,
    *,
    service_factory: ServiceFactory | None = None,
    interactive: bool = True,
) -> CheckResult:
    result = CheckResult(mode=settings.drive.auth_mode)
    if not settings.drive.folder_ids:
        result.error = (
            "No Drive folders configured. Set DRIVE_FOLDER_IDS or drive.folder_ids "
            "(run `drive-agent init`)."
        )
        return result

    try:
        drive_auth = auth.authenticate(settings.drive, settings.data_dir, interactive=interactive)
    except auth.DriveAuthError as exc:
        result.error = str(exc)
        return result
    result.identity = drive_auth.email
    result.description = drive_auth.description

    service = (service_factory or auth.build_drive_service)(drive_auth.credentials)
    try:
        about = service.about().get(fields="user(emailAddress)").execute(num_retries=3)
    except (HttpError, RefreshError) as exc:
        result.error = explain(exc, identity=result.identity)
        return result
    result.identity = (about.get("user") or {}).get("emailAddress") or result.identity

    for folder_id in settings.drive.folder_ids:
        result.folders.append(_check_folder(service, settings, folder_id, result.identity))
    return result


def _check_folder(
    service: Any, settings: Settings, folder_id: str, identity: str | None
) -> FolderCheck:
    check = FolderCheck(folder_id=folder_id)
    walker = DriveWalker(service, settings.drive)
    try:
        root = walker.resolve_root(folder_id)
        check.report = walker.scan(root)
    except NotAFolderError as exc:
        check.error = str(exc)
        return check
    except (HttpError, RefreshError) as exc:
        check.error = explain(exc, identity=identity, folder_id=folder_id)
        return check

    report, stats = check.report, check.report.stats
    if root.can_edit:
        check.warnings.append(
            f"{identity or 'This identity'} can edit this folder. The agent never writes, "
            "but Viewer access is enough: consider downgrading it (least privilege)."
        )
    if stats.incomplete:
        check.warnings.append("Drive returned an incomplete listing; some files may be missing.")
    if stats.inaccessible_folders:
        check.warnings.append(
            f"{stats.inaccessible_folders} sub-folder(s) could not be opened (not shared with "
            "this identity)."
        )
    if not report.files:
        check.warnings.append("No indexable files found in this folder.")
    return check

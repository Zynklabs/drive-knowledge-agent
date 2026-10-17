"""Data returned by the Drive walker."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from drive_agent.core.config import FileKind

SkipReason = Literal["out_of_scope", "unsupported", "excluded_type", "too_large"]

SKIP_LABELS: dict[SkipReason, str] = {
    "out_of_scope": "out of scope (spreadsheet / tabular)",
    "unsupported": "unsupported type",
    "excluded_type": "type not in include_mime",
    "too_large": "larger than max_file_mb",
}


@dataclass(frozen=True, slots=True)
class DriveFolder:
    id: str
    name: str
    drive_id: str | None
    """Shared Drive ID, or None for My Drive."""
    can_edit: bool


@dataclass(frozen=True, slots=True)
class DriveFile:
    id: str
    name: str
    mime_type: str
    kind: FileKind
    path: str
    """Slash-joined path from the root folder, e.g. ``HR/Policies/Leave Policy``."""
    root_folder_id: str
    web_url: str | None = None
    modified_time: str | None = None
    md5: str | None = None
    size: int | None = None
    owner: str | None = None
    drive_id: str | None = None
    via_shortcut: str | None = None
    """ID of the shortcut this file was reached through, if any."""


@dataclass(frozen=True, slots=True)
class SkippedFile:
    id: str
    name: str
    mime_type: str
    path: str
    reason: SkipReason
    detail: str | None = None


@dataclass
class WalkStats:
    folders: int = 0
    shortcuts: int = 0
    revisits: int = 0
    """Folders or files reached more than once (shortcut cycles, multiple parents)."""
    depth_limited: int = 0
    """Sub-folders not entered because of ``max_depth``."""
    inaccessible_folders: int = 0
    """Sub-folders that returned 403/404 when listed (e.g. shortcut targets not shared)."""
    incomplete: bool = False
    """Drive reported ``incompleteSearch`` for at least one listing."""


@dataclass
class WalkReport:
    root: DriveFolder
    files: list[DriveFile] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)
    stats: WalkStats = field(default_factory=WalkStats)

    def kinds(self) -> Counter[str]:
        return Counter(f.kind for f in self.files)

    def skip_reasons(self) -> Counter[SkipReason]:
        return Counter(s.reason for s in self.skipped)

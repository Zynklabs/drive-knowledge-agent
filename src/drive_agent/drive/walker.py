"""Breadth-first walk of a Drive folder tree, including Shared Drives and shortcuts."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Any

from googleapiclient.errors import HttpError

from drive_agent.core.config import DriveSettings
from drive_agent.drive.mime import FOLDER, SHORTCUT, file_kind, is_out_of_scope
from drive_agent.drive.models import (
    DriveFile,
    DriveFolder,
    SkippedFile,
    SkipReason,
    WalkReport,
    WalkStats,
)

FILE_FIELDS = (
    "id,name,mimeType,modifiedTime,md5Checksum,size,webViewLink,driveId,"
    "owners(emailAddress),shortcutDetails(targetId,targetMimeType)"
)
LIST_FIELDS = f"nextPageToken,incompleteSearch,files({FILE_FIELDS})"
ROOT_FIELDS = (
    "id,name,mimeType,driveId,capabilities(canEdit),shortcutDetails(targetId,targetMimeType)"
)


class NotAFolderError(Exception):
    """The configured ID points at a file, not a folder."""


class DriveWalker:
    """Lists every file under a root folder. Read-only: only ``files.get`` / ``files.list``."""

    def __init__(
        self,
        service: Any,
        settings: DriveSettings,
        *,
        page_size: int = 1000,
        num_retries: int = 5,
    ) -> None:
        self._files = service.files()
        self._settings = settings
        self._page_size = page_size
        self._retries = num_retries  # googleapiclient backs off on 429, 5xx and rate-limit 403s
        self.stats = WalkStats()

    def resolve_root(self, folder_id: str) -> DriveFolder:
        meta = self._get(folder_id, ROOT_FIELDS)
        if meta["mimeType"] == SHORTCUT:
            details = meta.get("shortcutDetails") or {}
            if details.get("targetMimeType") != FOLDER:
                raise NotAFolderError(f"{meta['name']!r} is a shortcut to a file, not a folder")
            meta = self._get(details["targetId"], ROOT_FIELDS)
        if meta["mimeType"] != FOLDER:
            raise NotAFolderError(f"{meta['name']!r} ({folder_id}) is a file, not a folder")
        return DriveFolder(
            id=meta["id"],
            name=meta["name"],
            drive_id=meta.get("driveId"),
            can_edit=bool((meta.get("capabilities") or {}).get("canEdit")),
        )

    def scan(self, root: DriveFolder) -> WalkReport:
        """Walk the whole tree and collect the results."""
        report = WalkReport(root=root, stats=self.stats)
        for item in self.walk(root):
            if isinstance(item, DriveFile):
                report.files.append(item)
            else:
                report.skipped.append(item)
        return report

    def walk(self, root: DriveFolder) -> Iterator[DriveFile | SkippedFile]:
        """Yield every file under ``root``, either indexable or skipped with a reason."""
        queue: deque[tuple[str, str, int]] = deque([(root.id, root.name, 0)])
        seen: set[str] = {root.id}
        while queue:
            folder_id, path, depth = queue.popleft()
            try:
                for item in self._list_children(folder_id):
                    yield from self._visit(item, root, path, depth, queue, seen)
            except HttpError as exc:
                # A sub-folder we cannot open (e.g. a shortcut into a folder that is not
                # shared with us) is counted, not fatal. The root must always be readable.
                if folder_id == root.id or exc.resp.status not in (403, 404):
                    raise
                self.stats.inaccessible_folders += 1

    def _visit(
        self,
        item: dict[str, Any],
        root: DriveFolder,
        parent_path: str,
        depth: int,
        queue: deque[tuple[str, str, int]],
        seen: set[str],
    ) -> Iterator[DriveFile | SkippedFile]:
        via_shortcut: str | None = None
        if item["mimeType"] == SHORTCUT:
            self.stats.shortcuts += 1
            resolved = self._resolve_shortcut(item, parent_path)
            if isinstance(resolved, SkippedFile):
                yield resolved
                return
            via_shortcut, item = item["id"], resolved

        path = f"{parent_path}/{item['name']}"
        if item["id"] in seen:
            self.stats.revisits += 1
            return
        seen.add(item["id"])

        if item["mimeType"] == FOLDER:
            max_depth = self._settings.max_depth
            if max_depth is not None and depth + 1 > max_depth:
                self.stats.depth_limited += 1
            else:
                self.stats.folders += 1
                queue.append((item["id"], path, depth + 1))
            return

        yield self._classify(item, root, path, via_shortcut)

    def _resolve_shortcut(
        self, shortcut: dict[str, Any], parent_path: str
    ) -> dict[str, Any] | SkippedFile:
        details = shortcut.get("shortcutDetails") or {}
        target_id = details.get("targetId")
        if not target_id:
            return self._skip(shortcut, parent_path, "unsupported", "broken shortcut")
        if details.get("targetMimeType") == FOLDER:
            return {"id": target_id, "name": shortcut["name"], "mimeType": FOLDER}
        try:
            return self._get(target_id, FILE_FIELDS)
        except HttpError as exc:
            if exc.resp.status in (403, 404):
                return self._skip(
                    shortcut, parent_path, "unsupported", "shortcut target is not accessible"
                )
            raise

    def _classify(
        self, item: dict[str, Any], root: DriveFolder, path: str, via_shortcut: str | None
    ) -> DriveFile | SkippedFile:
        name, mime = item["name"], item["mimeType"]
        parent_path = path.rsplit("/", 1)[0]
        if is_out_of_scope(name, mime):
            return self._skip(item, parent_path, "out_of_scope")
        kind = file_kind(name, mime)
        if kind is None:
            return self._skip(item, parent_path, "unsupported")
        if kind not in self._settings.include_mime:
            return self._skip(item, parent_path, "excluded_type", kind)
        size = int(item["size"]) if item.get("size") is not None else None
        if size is not None and size > self._settings.max_file_mb * 1024 * 1024:
            return self._skip(item, parent_path, "too_large", f"{size / 1024 / 1024:.1f} MB")
        owners = item.get("owners") or []
        return DriveFile(
            id=item["id"],
            name=name,
            mime_type=mime,
            kind=kind,
            path=path,
            root_folder_id=root.id,
            web_url=item.get("webViewLink"),
            modified_time=item.get("modifiedTime"),
            md5=item.get("md5Checksum"),
            size=size,
            owner=owners[0].get("emailAddress") if owners else None,
            drive_id=item.get("driveId"),
            via_shortcut=via_shortcut,
        )

    @staticmethod
    def _skip(
        item: dict[str, Any], parent_path: str, reason: SkipReason, detail: str | None = None
    ) -> SkippedFile:
        return SkippedFile(
            id=item["id"],
            name=item["name"],
            mime_type=item["mimeType"],
            path=f"{parent_path}/{item['name']}",
            reason=reason,
            detail=detail,
        )

    def _get(self, file_id: str, fields: str) -> dict[str, Any]:
        result: dict[str, Any] = self._files.get(
            fileId=file_id, fields=fields, supportsAllDrives=True
        ).execute(num_retries=self._retries)
        return result

    def _list_children(self, folder_id: str) -> Iterator[dict[str, Any]]:
        # corpora=allDrives so folders inside Shared Drives (incl. ones reached via
        # shortcuts) are listed; the parents filter keeps the query narrow.
        query = f"'{_escape(folder_id)}' in parents and trashed = false"
        page_token: str | None = None
        while True:
            response: dict[str, Any] = self._files.list(
                q=query,
                fields=LIST_FIELDS,
                pageSize=self._page_size,
                pageToken=page_token,
                corpora="allDrives",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute(num_retries=self._retries)
            if response.get("incompleteSearch"):
                self.stats.incomplete = True
            yield from response.get("files", [])
            page_token = response.get("nextPageToken")
            if not page_token:
                return


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")

"""In-memory stand-in for the subset of the Drive v3 client the agent uses."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import httplib2
from googleapiclient.errors import HttpError

from drive_agent.drive.mime import FOLDER, SHORTCUT

PARENT_QUERY = re.compile(r"'([^']+)' in parents and trashed = false")


def http_error(status: int, reason: str = "", message: str = "error") -> HttpError:
    body = {"error": {"code": status, "message": message, "errors": [{"reason": reason}]}}
    return HttpError(httplib2.Response({"status": status}), json.dumps(body).encode())


def folder(id: str, name: str, parent: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "id": id,
        "name": name,
        "mimeType": FOLDER,
        "parents": [parent] if parent else [],
        **extra,
    }


def file(
    id: str, name: str, mime: str, parent: str, size: int | None = None, **extra: Any
) -> dict[str, Any]:
    item = {"id": id, "name": name, "mimeType": mime, "parents": [parent], **extra}
    if size is not None:
        item["size"] = str(size)  # the API returns sizes as strings
    return item


def shortcut(id: str, name: str, parent: str, target: str, target_mime: str) -> dict[str, Any]:
    return {
        "id": id,
        "name": name,
        "mimeType": SHORTCUT,
        "parents": [parent],
        "shortcutDetails": {"targetId": target, "targetMimeType": target_mime},
    }


class _Request:
    def __init__(self, fn: Callable[[], Any]) -> None:
        self._fn = fn

    def execute(self, num_retries: int = 0) -> Any:
        return self._fn()


class Media:
    """Stands in for a media request; ``fake_download`` turns it into bytes."""

    def __init__(self, payload: bytes | Exception) -> None:
        self.payload = payload


def fake_download(self: Any, request: Media) -> bytes:
    """Replacement for ``DriveFetcher._download`` used with FakeDrive."""
    if isinstance(request.payload, Exception):
        raise request.payload
    return request.payload


class FakeDrive:
    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        user_email: str = "agent@example.iam.gserviceaccount.com",
        get_errors: dict[str, HttpError] | None = None,
        list_errors: dict[str, HttpError] | None = None,
        about_error: Exception | None = None,
        incomplete: bool = False,
        contents: dict[str, bytes | Exception] | None = None,
    ) -> None:
        self.items = {item["id"]: item for item in items}
        self.user_email = user_email
        self.get_errors = get_errors or {}
        self.list_errors = list_errors or {}
        self.about_error = about_error
        self.incomplete = incomplete
        self.list_calls: list[dict[str, Any]] = []
        self.contents = contents or {}
        self.media_calls: list[tuple[str, str | None]] = []

    # client surface ---------------------------------------------------------
    def files(self) -> FakeDrive:
        return self

    def about(self) -> FakeDrive:
        return self

    def get(self, fileId: str | None = None, fields: str = "", **kw: Any) -> _Request:
        if fileId is None:  # about().get()
            return _Request(self._about)
        assert kw.get("supportsAllDrives") is True
        return _Request(lambda: self._get(fileId))

    def list(self, **kw: Any) -> _Request:
        assert kw["supportsAllDrives"] is True
        assert kw["includeItemsFromAllDrives"] is True
        self.list_calls.append(kw)
        return _Request(lambda: self._list(kw))

    def export_media(self, fileId: str, mimeType: str) -> Media:
        self.media_calls.append((fileId, mimeType))
        return Media(self.contents.get(fileId, b""))

    def get_media(self, fileId: str, **kw: Any) -> Media:
        assert kw.get("supportsAllDrives") is True
        self.media_calls.append((fileId, None))
        return Media(self.contents.get(fileId, b""))

    # behaviour --------------------------------------------------------------
    def _about(self) -> dict[str, Any]:
        if self.about_error:
            raise self.about_error
        return {"user": {"emailAddress": self.user_email}}

    def _get(self, file_id: str) -> dict[str, Any]:
        if file_id in self.get_errors:
            raise self.get_errors[file_id]
        if file_id not in self.items:
            raise http_error(404, "notFound", f"File not found: {file_id}.")
        return {k: v for k, v in self.items[file_id].items() if k not in ("parents", "trashed")}

    def _list(self, kw: dict[str, Any]) -> dict[str, Any]:
        match = PARENT_QUERY.fullmatch(kw["q"])
        assert match, kw["q"]
        parent = match.group(1)
        if parent in self.list_errors:
            raise self.list_errors[parent]
        children = [
            {k: v for k, v in item.items() if k not in ("parents", "trashed")}
            for item in self.items.values()
            if parent in item.get("parents", []) and not item.get("trashed")
        ]
        start = int(kw.get("pageToken") or 0)
        end = start + kw["pageSize"]
        response: dict[str, Any] = {"files": children[start:end]}
        if end < len(children):
            response["nextPageToken"] = str(end)
        if self.incomplete:
            response["incompleteSearch"] = True
        return response

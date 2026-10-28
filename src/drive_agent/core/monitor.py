"""Small files a running bot writes into its data dir so the admin portal can watch it.

- ``heartbeat.json``: rewritten every few seconds while a long-running interface is up;
- ``activity.jsonl``: one line per question answered (metadata; question text is opt-in);
- ``sync_progress.json``: the current or last sync, updated as each file is indexed;
- ``sync_history.jsonl``: one line per finished sync (new, updated and removed files);
- ``sync_request.json``: left by the admin portal's "Sync now"; the running bot takes it and
  syncs, so the portal never runs a sync (or loads a model) itself.

Writers never raise: monitoring must not break answering or syncing.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HEARTBEAT_FILE = "heartbeat.json"
ACTIVITY_FILE = "activity.jsonl"
SYNC_PROGRESS_FILE = "sync_progress.json"
SYNC_HISTORY_FILE = "sync_history.jsonl"
SYNC_REQUEST_FILE = "sync_request.json"

HEARTBEAT_SECONDS = 30
STALE_AFTER_SECONDS = 3 * HEARTBEAT_SECONDS
"""A heartbeat older than this means the process is gone (or hung)."""

_append_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Replace ``path`` atomically, so readers never see a half-written file."""
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class Heartbeat:
    """Rewrites ``heartbeat.json`` from a daemon thread while the process runs."""

    def __init__(
        self,
        data_dir: Path,
        interface: str,
        *,
        details: dict[str, Any] | None = None,
        probe: Callable[[], dict[str, Any]] | None = None,
        every: float = HEARTBEAT_SECONDS,
    ) -> None:
        self._path = data_dir / HEARTBEAT_FILE
        self._probe = probe
        """Live status merged into every beat (e.g. the Slack connection); errors are ignored."""
        self._every = every
        self._stop = threading.Event()
        self._info = {
            "interface": interface,
            "pid": os.getpid(),
            "started_at": now_iso(),
            **(details or {}),  # e.g. the settings this process actually runs with
        }
        self._thread = threading.Thread(target=self._run, name="heartbeat", daemon=True)

    def start(self) -> Heartbeat:
        self.beat()
        self._thread.start()
        return self

    def beat(self) -> None:
        live: dict[str, Any] = {}
        if self._probe is not None:
            with contextlib.suppress(Exception):
                live = self._probe()
        write_json(self._path, {**self._info, **live, "updated_at": now_iso()})

    def stop(self) -> None:
        self._stop.set()
        write_json(self._path, {**self._info, "updated_at": now_iso(), "stopped_at": now_iso()})

    def _run(self) -> None:
        while not self._stop.wait(self._every):
            self.beat()


def _append(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with _append_lock, contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _read_lines(path: Path, limit: int) -> list[dict[str, Any]]:
    """The newest ``limit`` records, oldest first."""
    try:
        with path.open(encoding="utf-8") as fh:
            lines = deque(fh, maxlen=limit)
    except OSError:
        return []
    events = []
    for line in lines:
        with contextlib.suppress(ValueError):
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
    return events


def record_activity(data_dir: Path, event: dict[str, Any]) -> None:
    _append(data_dir / ACTIVITY_FILE, {"at": now_iso(), **event})


def read_activity(data_dir: Path, limit: int = 1000) -> list[dict[str, Any]]:
    """The newest ``limit`` events, oldest first."""
    return _read_lines(data_dir / ACTIVITY_FILE, limit)


def read_sync_history(data_dir: Path, limit: int = 500) -> list[dict[str, Any]]:
    """The newest ``limit`` finished syncs, oldest first."""
    return _read_lines(data_dir / SYNC_HISTORY_FILE, limit)


def request_sync(data_dir: Path, *, by: str = "admin") -> None:
    write_json(data_dir / SYNC_REQUEST_FILE, {"requested_at": now_iso(), "by": by})


def pending_sync_request(data_dir: Path) -> dict[str, Any] | None:
    return read_json(data_dir / SYNC_REQUEST_FILE)


def take_sync_request(data_dir: Path) -> dict[str, Any] | None:
    """Claim a pending request (once: renaming it away is atomic), or None if there is none."""
    path = data_dir / SYNC_REQUEST_FILE
    taken = path.with_name(f".{path.name}.{os.getpid()}.taken")
    try:
        path.replace(taken)
    except OSError:
        return None
    request = read_json(taken) or {}
    taken.unlink(missing_ok=True)
    return request


class SyncProgress:
    """Keeps ``sync_progress.json`` current while a sync runs."""

    def __init__(self, data_dir: Path, *, full: bool = False, trigger: str = "manual") -> None:
        self._path = data_dir / SYNC_PROGRESS_FILE
        self._history = data_dir / SYNC_HISTORY_FILE
        self.state: dict[str, Any] = {
            "state": "running",
            "pid": os.getpid(),
            "full": full,
            "trigger": trigger,
            "started_at": now_iso(),
            "files_total": None,
            "files_done": 0,
            "files_failed": 0,
            "files_empty": 0,
            "chunks_added": 0,
        }
        self._save()

    def listed(self, files: int) -> None:
        """A folder was listed: ``files`` more to load."""
        self.state["files_total"] = (self.state["files_total"] or 0) + files
        self._save()

    def file_done(self, *, failed: bool, chunks_added: int, empty: bool = False) -> None:
        self.state["files_done"] += 1
        self.state["files_failed"] += int(failed)
        self.state["files_empty"] += int(empty)
        self.state["chunks_added"] += chunks_added
        self._save()

    def finish(
        self, *, ok: bool, error: str | None = None, files: dict[str, int] | None = None
    ) -> None:
        """Record the outcome, plus ``files`` (new, updated, removed...) from the sync report."""
        self.state.update(state="done" if ok else "failed", finished_at=now_iso(), **(files or {}))
        if error:
            self.state["error"] = error
        self._save()
        _append(self._history, {k: v for k, v in self.state.items() if k != "pid"})

    def _save(self) -> None:
        write_json(self._path, {**self.state, "updated_at": now_iso()})

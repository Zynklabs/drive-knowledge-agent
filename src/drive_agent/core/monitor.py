"""Small files a running bot writes into its data dir so the admin portal can watch it.

- ``heartbeat.json``: rewritten every few seconds while a long-running interface is up;
- ``activity.jsonl``: one line per question answered (metadata; question text is opt-in);
- ``sync_progress.json``: the current or last sync, updated as each file is indexed.

Writers never raise: monitoring must not break answering or syncing.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HEARTBEAT_FILE = "heartbeat.json"
ACTIVITY_FILE = "activity.jsonl"
SYNC_PROGRESS_FILE = "sync_progress.json"

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
        every: float = HEARTBEAT_SECONDS,
    ) -> None:
        self._path = data_dir / HEARTBEAT_FILE
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
        write_json(self._path, {**self._info, "updated_at": now_iso()})

    def stop(self) -> None:
        self._stop.set()
        write_json(self._path, {**self._info, "updated_at": now_iso(), "stopped_at": now_iso()})

    def _run(self) -> None:
        while not self._stop.wait(self._every):
            self.beat()


def record_activity(data_dir: Path, event: dict[str, Any]) -> None:
    line = json.dumps({"at": now_iso(), **event}, ensure_ascii=False)
    with _append_lock, contextlib.suppress(OSError):
        data_dir.mkdir(parents=True, exist_ok=True)
        with (data_dir / ACTIVITY_FILE).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def read_activity(data_dir: Path, limit: int = 1000) -> list[dict[str, Any]]:
    """The newest ``limit`` events, oldest first."""
    try:
        with (data_dir / ACTIVITY_FILE).open(encoding="utf-8") as fh:
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


class SyncProgress:
    """Keeps ``sync_progress.json`` current while a sync runs."""

    def __init__(self, data_dir: Path, *, full: bool = False) -> None:
        self._path = data_dir / SYNC_PROGRESS_FILE
        self.state: dict[str, Any] = {
            "state": "running",
            "pid": os.getpid(),
            "full": full,
            "started_at": now_iso(),
            "files_done": 0,
            "files_failed": 0,
            "chunks_added": 0,
        }
        self._save()

    def file_done(self, *, failed: bool, chunks_added: int) -> None:
        self.state["files_done"] += 1
        self.state["files_failed"] += int(failed)
        self.state["chunks_added"] += chunks_added
        self._save()

    def finish(self, *, ok: bool, error: str | None = None) -> None:
        self.state.update(state="done" if ok else "failed", finished_at=now_iso())
        if error:
            self.state["error"] = error
        self._save()

    def _save(self) -> None:
        write_json(self._path, {**self.state, "updated_at": now_iso()})

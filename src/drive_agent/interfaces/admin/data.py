"""What the admin portal shows, read from files only.

Nothing here loads a model, opens the vector store or starts a sync: the portal must stay
cheap and must never interfere with the bots it watches. Sync state is read from each bot's
``state.db`` in read-only mode; liveness, activity and sync progress come from the files in
``drive_agent.core.monitor``.
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from drive_agent.core.config import BOTS_DIR, ConfigError, Settings, load_settings
from drive_agent.core.knowledge_base import next_due
from drive_agent.core.monitor import (
    HEARTBEAT_FILE,
    STALE_AFTER_SECONDS,
    SYNC_PROGRESS_FILE,
    read_activity,
    read_json,
)
from drive_agent.indexing.state import STATE_DB
from drive_agent.vectorstores.location import namespace_for

DEFAULT_BOT = "(default)"
"""The knowledge base configured by .env alone, when no bots/<name>.env files exist."""

Health = Literal["running", "not responding", "stopped", "never started"]
SlackHealth = Literal["connected", "unstable", "disconnected", "unknown"]

UNSTABLE_FAILURES = 3
"""Connection drops or errors within the bot's 10-minute window that mark Slack as unstable.
One drop is normal (Slack rotates connections); a reconnect loop fails every few seconds, and
after a network blip it may report only errors, never a close."""


@dataclass
class IndexState:
    chunks: int = 0
    synced_files: int = 0
    """Files with at least one chunk in the index right now (grows while a sync runs)."""
    total_files: int | None = None
    """Files found in the folder at the last finished sync (None: not recorded yet)."""
    failed: int = 0
    empty: int = 0
    """Files with no readable text (e.g. scanned PDFs) at the last finished sync."""
    skipped: int = 0
    """Files left out by type or size (spreadsheets, images, too large)."""
    last_sync: datetime | None = None


@dataclass
class ActivityStats:
    questions: int = 0
    answered: int = 0
    not_found: int = 0
    errors: int = 0
    avg_seconds: float | None = None
    users: int = 0


@dataclass
class BotOverview:
    name: str
    settings: Settings | None = None
    config_error: str | None = None
    health: Health = "never started"
    heartbeat: dict[str, Any] | None = None
    index: IndexState = field(default_factory=IndexState)
    sync: dict[str, Any] | None = None
    """The current or last sync's progress file, if any."""
    schedule: str | None = None
    """Auto-sync cron: what the running bot reports, else its configuration."""
    next_sync: datetime | None = None
    last_24h: ActivityStats = field(default_factory=ActivityStats)
    slack_restarts: list[dict[str, Any]] = field(default_factory=list)
    """The Slack watchdog's reconnects and restarts in the last 24 h, newest first."""

    @property
    def slack_health(self) -> SlackHealth:
        return slack_health_of(self.heartbeat, self.health)

    @property
    def syncing(self) -> bool:
        return bool(self.sync and self.sync.get("state") == "running")

    @property
    def total_files(self) -> int | None:
        """Files in the folder: the running sync's listing, else the last finished sync's."""
        if self.syncing and self.sync and isinstance(self.sync.get("files_total"), int):
            return int(self.sync["files_total"])
        return self.index.total_files


def bot_names(bots_dir: Path = BOTS_DIR) -> list[str]:
    names = sorted(path.stem for path in bots_dir.glob("*.env"))
    return names or [DEFAULT_BOT]


def bot_settings(name: str, bots_dir: Path = BOTS_DIR) -> Settings:
    extra = None if name == DEFAULT_BOT else bots_dir / f"{name}.env"
    return load_settings(extra_env_file=extra)


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def health_of(heartbeat: dict[str, Any] | None, now: datetime | None = None) -> Health:
    if heartbeat is None:
        return "never started"
    if heartbeat.get("stopped_at"):
        return "stopped"
    updated = parse_time(heartbeat.get("updated_at"))
    now = now or datetime.now(UTC)
    if updated is None or now - updated > timedelta(seconds=STALE_AFTER_SECONDS):
        return "not responding"
    return "running"


def slack_health_of(heartbeat: dict[str, Any] | None, health: Health) -> SlackHealth:
    """The Slack connection as the running bot last reported it ("unknown" if it doesn't)."""
    slack = (heartbeat or {}).get("slack")
    if health != "running" or not isinstance(slack, dict):
        return "unknown"
    if not slack.get("connected"):
        return "disconnected"
    failures = max(int(slack.get("drops_10m") or 0), int(slack.get("errors_10m") or 0))
    if failures >= UNSTABLE_FAILURES:
        return "unstable"
    return "connected"


def read_index_state(settings: Settings) -> IndexState:
    db = settings.data_dir / STATE_DB
    if not db.is_file():
        return IndexState()
    namespace = namespace_for(settings)
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
    except sqlite3.Error:
        return IndexState()
    try:
        chunks, synced = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT group_id) FROM upsertion_record WHERE namespace = ?",
            [namespace],
        ).fetchone()
        row = conn.execute(
            "SELECT value FROM meta WHERE namespace = ? AND name = 'last_sync'", [namespace]
        ).fetchone()
    except sqlite3.Error:
        return IndexState()
    finally:
        conn.close()
    meta = json.loads(row[0]) if row else {}
    total = meta.get("total_files")
    return IndexState(
        chunks=int(chunks),
        synced_files=int(synced),
        total_files=int(total) if isinstance(total, int) else None,
        failed=int(meta.get("failed", 0)),
        empty=int(meta.get("empty", 0)),
        skipped=int(meta.get("skipped", 0)),
        last_sync=parse_time(meta.get("at")),
    )


def activity_stats(events: list[dict[str, Any]], since: datetime) -> ActivityStats:
    recent = [
        e
        for e in events
        if e.get("kind") == "question" and (parse_time(e.get("at")) or since) > since
    ]
    seconds = [float(e["seconds"]) for e in recent if isinstance(e.get("seconds"), int | float)]
    return ActivityStats(
        questions=len(recent),
        answered=sum(1 for e in recent if e.get("outcome") == "answered"),
        not_found=sum(1 for e in recent if e.get("outcome") == "not_found"),
        errors=sum(1 for e in recent if e.get("outcome") == "error"),
        avg_seconds=round(sum(seconds) / len(seconds), 1) if seconds else None,
        users=len({e.get("user") for e in recent if e.get("user")}),
    )


def slack_restarts(events: list[dict[str, Any]], since: datetime) -> list[dict[str, Any]]:
    return [
        e
        for e in reversed(events)
        if e.get("kind") == "slack_restart" and (parse_time(e.get("at")) or since) > since
    ]


def overview(name: str, bots_dir: Path = BOTS_DIR, now: datetime | None = None) -> BotOverview:
    result = BotOverview(name=name)
    try:
        settings = bot_settings(name, bots_dir)
    except ConfigError as exc:
        result.config_error = str(exc)
        return result
    now = now or datetime.now(UTC)
    result.settings = settings
    result.heartbeat = read_json(settings.data_dir / HEARTBEAT_FILE)
    result.health = health_of(result.heartbeat, now)
    result.index = read_index_state(settings)
    result.sync = read_json(settings.data_dir / SYNC_PROGRESS_FILE)
    hb = result.heartbeat or {}
    live = result.health == "running" and "schedule" in hb
    schedule = hb["schedule"] if live else settings.ingest.schedule_cron
    result.schedule = schedule if isinstance(schedule, str) and schedule else None
    if result.schedule:
        result.next_sync = next_due(result.schedule, result.index.last_sync)
    events, day_ago = read_activity(settings.data_dir), now - timedelta(hours=24)
    result.last_24h = activity_stats(events, since=day_ago)
    result.slack_restarts = slack_restarts(events, since=day_ago)
    return result


def log_files(log_dir: Path | None) -> list[Path]:
    if log_dir is None or not log_dir.is_dir():
        return []
    return sorted(
        (p for p in log_dir.iterdir() if p.is_file() and p.suffix in {".log", ".txt"}),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def tail(path: Path, lines: int = 200, contains: str = "") -> list[str]:
    """The last ``lines`` lines of ``path`` (that contain ``contains``, case-insensitive)."""
    needle = contains.lower()
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            # Progress bars rewrite one line with \r; show only each line's final state.
            matching = (
                line.rstrip("\n").rsplit("\r", 1)[-1]
                for line in fh
                if not needle or needle in line.lower()
            )
            return list(deque(matching, maxlen=lines))
    except OSError as exc:
        return [f"(cannot read {path.name}: {exc.strerror})"]

"""``KnowledgeBase``: one Drive knowledge base and everything needed to keep it useful.

It owns the vector index, conversation memory, the agent, and the index's freshness:
syncs go through one coordinator (never two at once), and ``start_auto_sync`` re-syncs on
``ingest.schedule_cron`` for as long as the hosting process runs (Slack bot, web app,
``drive-agent sync --watch``). Every interface builds on this instead of wiring pieces itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from drive_agent.core.agent import DriveAgent, create_checkpointer, make_retriever
from drive_agent.core.config import Settings
from drive_agent.indexing.loader import FileResult
from drive_agent.indexing.scheduler import AutoSync, SyncCoordinator, SyncFn, is_overdue
from drive_agent.indexing.state import STATE_DB, SQLiteRecordManager
from drive_agent.indexing.sync import SyncReport, sync_now
from drive_agent.llm import LLM
from drive_agent.vectorstores.base import VectorIndex


@dataclass(frozen=True)
class IndexStatus:
    chunks: int
    files: int
    failed: int
    last_sync: datetime | None
    schedule: str | None
    """Auto-sync cron (UTC), or None when auto-sync is off."""
    next_sync: datetime | None
    """When the next automatic sync runs (or is due, if no long-running process is up)."""
    overdue: bool
    syncing: bool


def next_due(schedule: str, last_sync: datetime | None) -> datetime | None:
    from apscheduler.triggers.cron import CronTrigger

    trigger = CronTrigger.from_crontab(schedule, timezone="UTC")
    start = last_sync or datetime.now(trigger.timezone)
    result: datetime | None = trigger.get_next_fire_time(None, start)
    return result


def _sync_now(settings: Settings, vector_index: VectorIndex, **kwargs: Any) -> SyncReport:
    return sync_now(settings, vector_index, **kwargs)  # looked up at call time (tests swap it)


class KnowledgeBase:
    def __init__(
        self,
        settings: Settings,
        index: VectorIndex,
        checkpointer: BaseCheckpointSaver[Any],
        *,
        sync_fn: SyncFn | None = None,
    ) -> None:
        self.settings = settings
        self.index = index
        self.checkpointer = checkpointer
        self.coordinator = SyncCoordinator(settings, index, sync_fn=sync_fn or _sync_now)
        schedule = settings.ingest.schedule_cron
        self.auto_sync = AutoSync(self.coordinator, schedule, self.last_sync) if schedule else None

    @classmethod
    def open(cls, settings: Settings) -> KnowledgeBase:
        """Load embeddings and the vector index for ``settings``."""
        # Module-level lookups (not `from x import y`) keep these swappable in tests.
        from drive_agent import embeddings, vectorstores

        index = vectorstores.build_vector_index(settings, embeddings.build_embeddings(settings))
        return cls(settings, index, create_checkpointer(settings))

    # --- answering -----------------------------------------------------------

    def agent(self, llm: LLM | None = None) -> DriveAgent:
        if llm is None:
            from drive_agent.llm import claude

            llm = claude.ClaudeLLM(self.settings)
        return DriveAgent(self.settings, llm, make_retriever(self.index.store), self.checkpointer)

    # --- freshness -------------------------------------------------------------

    def sync(
        self,
        trigger: str = "manual",
        *,
        full: bool = False,
        on_file: Callable[[FileResult], None] | None = None,
    ) -> SyncReport:
        """Sync now. Raises ``SyncBusy`` if a sync is already running."""
        return self.coordinator.run(trigger, full=full, on_file=on_file)

    def start_auto_sync(self) -> bool:
        """Re-sync on the schedule while this process runs. False when auto-sync is off."""
        if self.auto_sync is None:
            return False
        self.auto_sync.start()
        return True

    def _last_sync_meta(self) -> dict[str, Any]:
        db = self.settings.data_dir / STATE_DB
        if not db.is_file():
            return {}
        records = SQLiteRecordManager(self.index.namespace, db)
        try:
            meta = records.get_meta("last_sync")
        finally:
            records.close()
        return meta if isinstance(meta, dict) else {}

    def last_sync(self) -> datetime | None:
        at = self._last_sync_meta().get("at")
        return datetime.fromisoformat(at) if at else None

    def status(self) -> IndexStatus:
        meta = self._last_sync_meta()
        last = datetime.fromisoformat(meta["at"]) if meta.get("at") else None
        schedule = self.settings.ingest.schedule_cron
        running = self.auto_sync.next_run() if self.auto_sync and self.auto_sync.started else None
        return IndexStatus(
            chunks=self.index.count(),
            files=int(meta.get("files", 0)),
            failed=int(meta.get("failed", 0)),
            last_sync=last,
            schedule=schedule,
            next_sync=running or (next_due(schedule, last) if schedule else None),
            overdue=bool(schedule) and is_overdue(schedule or "", last),
            syncing=self.coordinator.running,
        )

    def close(self) -> None:
        if self.auto_sync is not None:
            self.auto_sync.stop()
        self.index.close()
        conn = getattr(self.checkpointer, "conn", None)
        if conn is not None:
            conn.close()

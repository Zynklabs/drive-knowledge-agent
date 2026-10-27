"""Keeping an index fresh: one-at-a-time syncs plus a cron schedule with catch-up.

``SyncCoordinator`` guarantees a single sync at a time, whoever starts it (the schedule, a
Slack command, the web app, the CLI). ``AutoSync`` fires it on ``ingest.schedule_cron`` (UTC)
and, on start, runs one immediately if a scheduled run was missed while nothing was running.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from drive_agent.core.config import Settings
from drive_agent.drive.auth import DriveAuthError
from drive_agent.indexing.loader import FileResult
from drive_agent.indexing.sync import SyncError, SyncReport, summarize, sync_now
from drive_agent.vectorstores.base import VectorIndex

log = logging.getLogger(__name__)


class SyncFn(Protocol):
    def __call__(
        self,
        settings: Settings,
        vector_index: VectorIndex,
        *,
        full: bool = ...,
        on_file: Callable[[FileResult], None] | None = ...,
        trigger: str = ...,
    ) -> SyncReport: ...


class SyncBusy(Exception):
    """Another sync is already running."""


class SyncCoordinator:
    def __init__(
        self, settings: Settings, vector_index: VectorIndex, *, sync_fn: SyncFn = sync_now
    ) -> None:
        self._settings = settings
        self._index = vector_index
        self._sync_fn = sync_fn
        self._lock = threading.Lock()
        self.last_report: SyncReport | None = None

    @property
    def running(self) -> bool:
        return self._lock.locked()

    def run(
        self,
        trigger: str,
        *,
        full: bool = False,
        on_file: Callable[[FileResult], None] | None = None,
    ) -> SyncReport:
        """Run one sync now. Raises ``SyncBusy``, ``DriveAuthError`` or ``SyncError``."""
        if not self._lock.acquire(blocking=False):
            raise SyncBusy
        try:
            log.info("sync started (%s)", trigger)
            report = self._sync_fn(
                self._settings, self._index, full=full, on_file=on_file, trigger=trigger
            )
            self.last_report = report
            log.info("sync finished (%s): %s", trigger, summarize(report))
            return report
        finally:
            self._lock.release()


def is_overdue(schedule: str, last_sync: datetime | None, now: datetime | None = None) -> bool:
    """True when a scheduled run should have happened since ``last_sync`` (or never synced)."""
    if last_sync is None:
        return True
    from apscheduler.triggers.cron import CronTrigger

    trigger = CronTrigger.from_crontab(schedule, timezone="UTC")
    due = trigger.get_next_fire_time(None, last_sync)
    return due is not None and due <= (now or datetime.now(UTC))


class AutoSync:
    JOB_ID = "drive-sync"
    CATCH_UP_ID = "drive-sync-catch-up"

    def __init__(
        self,
        coordinator: SyncCoordinator,
        schedule: str,
        last_sync: Callable[[], datetime | None],
    ) -> None:
        self._coordinator = coordinator
        self.schedule = schedule
        self._last_sync = last_sync
        self._scheduler: Any = None

    @property
    def started(self) -> bool:
        return self._scheduler is not None

    def start(self) -> None:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        if self.started:
            return
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._scheduler.add_job(
            self.tick,
            CronTrigger.from_crontab(self.schedule, timezone="UTC"),
            id=self.JOB_ID,
            max_instances=1,  # never stack runs
            coalesce=True,  # after downtime, run once rather than once per missed slot
            misfire_grace_time=600,
        )
        if is_overdue(self.schedule, self._last_sync()):
            self._scheduler.add_job(self.tick, id=self.CATCH_UP_ID)  # runs right away
            log.info("index is overdue for a sync: catching up now")
        self._scheduler.start()
        log.info("auto-sync on: %s (UTC), next run %s", self.schedule, self.next_run())

    def tick(self) -> None:
        try:
            self._coordinator.run("schedule")
        except SyncBusy:
            log.info("auto-sync skipped: a sync is already running")
        except (DriveAuthError, SyncError) as exc:
            log.warning("auto-sync failed: %s", exc)
        except Exception:
            log.exception("auto-sync crashed")

    def next_run(self) -> datetime | None:
        if self._scheduler is None:
            return None
        job = self._scheduler.get_job(self.JOB_ID)
        return job.next_run_time if job is not None else None

    def stop(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None

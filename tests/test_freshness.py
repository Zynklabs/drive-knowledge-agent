"""Keeping the index fresh: schedule, one-at-a-time syncs, catch-up, KnowledgeBase, --watch."""

import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from typer.testing import CliRunner

import drive_agent.core.knowledge_base as kb_module
import drive_agent.embeddings
from drive_agent.core.config import ConfigError, Settings, load_settings
from drive_agent.core.knowledge_base import KnowledgeBase, next_due
from drive_agent.drive.auth import DriveAuthError
from drive_agent.indexing.scheduler import AutoSync, SyncBusy, SyncCoordinator, is_overdue
from drive_agent.indexing.state import STATE_DB, SQLiteRecordManager
from drive_agent.indexing.sync import SyncReport
from drive_agent.interfaces import cli
from drive_agent.vectorstores import build_vector_index

from .fakes import FakeLLM

HOURLY = "0 * * * *"
NOW = datetime(2026, 10, 17, 12, 30, tzinfo=UTC)


class Recorder:
    """A sync function that records calls and can block or fail."""

    def __init__(self, error: Exception | None = None, block: threading.Event | None = None):
        self.calls: list[dict[str, Any]] = []
        self.error, self.block = error, block
        self.started = threading.Event()

    def __call__(self, settings: Any, index: Any, **kwargs: Any) -> SyncReport:
        self.calls.append(kwargs)
        self.started.set()
        if self.block is not None:
            self.block.wait(5)
        if self.error is not None:
            raise self.error
        return SyncReport(added=1)


# --- config -------------------------------------------------------------------


@pytest.mark.parametrize("value", ["off", "OFF", "none", "never", "disabled"])
def test_schedule_can_be_turned_off(value: str) -> None:
    assert load_settings(env={"SYNC_SCHEDULE": value}).ingest.schedule_cron is None


def test_schedule_is_validated() -> None:
    assert load_settings(env={"SYNC_SCHEDULE": " 0 3 * * 1 "}).ingest.schedule_cron == "0 3 * * 1"
    with pytest.raises(ConfigError, match="invalid cron expression"):
        load_settings(env={"SYNC_SCHEDULE": "every hour"})


def test_default_schedule_is_every_six_hours() -> None:
    assert Settings().ingest.schedule_cron == "0 */6 * * *"


# --- overdue / next due ---------------------------------------------------------


@pytest.mark.parametrize(
    ("last", "overdue"),
    [
        (None, True),  # never synced
        (NOW - timedelta(minutes=10), False),  # synced after the 12:00 slot
        (NOW - timedelta(minutes=45), True),  # 11:45: missed the 12:00 run
    ],
)
def test_is_overdue(last: datetime | None, overdue: bool) -> None:
    assert is_overdue(HOURLY, last, now=NOW) is overdue


def test_next_due() -> None:
    assert next_due(HOURLY, NOW - timedelta(minutes=10)) == datetime(
        2026, 10, 17, 13, 0, tzinfo=UTC
    )


# --- coordinator ------------------------------------------------------------------


def test_coordinator_runs_one_sync_at_a_time() -> None:
    release = threading.Event()
    fn = Recorder(block=release)
    coordinator = SyncCoordinator(Settings(), object(), sync_fn=fn)  # type: ignore[arg-type]
    worker = threading.Thread(target=coordinator.run, args=("schedule",))
    worker.start()
    assert fn.started.wait(5)
    assert coordinator.running
    with pytest.raises(SyncBusy):
        coordinator.run("slack")
    release.set()
    worker.join(5)
    assert not coordinator.running
    assert coordinator.run("web", full=True).added == 1
    assert fn.calls[-1] == {"full": True, "on_file": None, "trigger": "web"}
    assert coordinator.last_report is not None


def test_coordinator_releases_the_lock_after_a_failure() -> None:
    coordinator = SyncCoordinator(Settings(), object(), sync_fn=Recorder(DriveAuthError("x")))  # type: ignore[arg-type]
    with pytest.raises(DriveAuthError):
        coordinator.run("manual")
    assert not coordinator.running


# --- auto-sync ----------------------------------------------------------------------


def test_auto_sync_catches_up_when_overdue() -> None:
    fn = Recorder()
    auto = AutoSync(SyncCoordinator(Settings(), object(), sync_fn=fn), HOURLY, lambda: None)  # type: ignore[arg-type]
    auto.start()
    try:
        assert fn.started.wait(5)  # catch-up ran without waiting for the next slot
        next_run = auto.next_run()
        assert next_run is not None
        assert next_run.minute == 0
    finally:
        auto.stop()
    assert not auto.started


def test_auto_sync_waits_when_fresh() -> None:
    fn = Recorder()
    fresh = datetime.now(UTC)
    auto = AutoSync(SyncCoordinator(Settings(), object(), sync_fn=fn), HOURLY, lambda: fresh)  # type: ignore[arg-type]
    auto.start()
    auto.start()  # idempotent
    try:
        time.sleep(0.3)
        assert fn.calls == []
    finally:
        auto.stop()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (DriveAuthError("no access"), "auto-sync failed: no access"),
        (RuntimeError("bug"), "auto-sync crashed"),
    ],
)
def test_auto_sync_tick_never_raises(
    caplog: pytest.LogCaptureFixture, error: Exception, message: str
) -> None:
    auto = AutoSync(
        SyncCoordinator(Settings(), object(), sync_fn=Recorder(error)), HOURLY, lambda: None
    )  # type: ignore[arg-type]
    with caplog.at_level(logging.INFO):
        auto.tick()
    assert message in caplog.text


def test_auto_sync_tick_skips_when_busy(caplog: pytest.LogCaptureFixture) -> None:
    release = threading.Event()
    coordinator = SyncCoordinator(Settings(), object(), sync_fn=Recorder(block=release))  # type: ignore[arg-type]
    worker = threading.Thread(target=coordinator.run, args=("slack",))
    worker.start()
    time.sleep(0.1)
    with caplog.at_level(logging.INFO):
        AutoSync(coordinator, HOURLY, lambda: None).tick()
    release.set()
    worker.join(5)
    assert "auto-sync skipped" in caplog.text


# --- KnowledgeBase ----------------------------------------------------------------------


def make_kb(
    tmp_path: Path, schedule: str | None = HOURLY, fn: Recorder | None = None
) -> KnowledgeBase:
    settings = Settings.model_validate(
        {
            "data_dir": str(tmp_path),
            "vector_store": {"url": ":memory:"},
            "retrieval": {"hybrid": False},
            "ingest": {"schedule_cron": schedule},
        }
    )
    index = build_vector_index(settings, DeterministicFakeEmbedding(size=8))
    from langgraph.checkpoint.memory import InMemorySaver

    return KnowledgeBase(settings, index, InMemorySaver(), sync_fn=fn or Recorder())


def test_knowledge_base_status(tmp_path: Path) -> None:
    kb = make_kb(tmp_path)
    try:
        status = kb.status()
        assert (status.chunks, status.files, status.last_sync) == (0, 0, None)
        assert status.overdue is True
        assert status.schedule == HOURLY

        records = SQLiteRecordManager(kb.index.namespace, tmp_path / STATE_DB)
        just_now = datetime.now(UTC).replace(microsecond=0)
        records.set_meta("last_sync", {"at": just_now.isoformat(), "files": 4, "failed": 1})
        records.close()
        status = kb.status()
        assert (status.files, status.failed, status.last_sync) == (4, 1, just_now)
        assert status.overdue is False
        assert status.next_sync == next_due(HOURLY, just_now)
    finally:
        kb.close()


def test_knowledge_base_sync_and_auto_sync(tmp_path: Path) -> None:
    fn = Recorder()
    kb = make_kb(tmp_path, fn=fn)
    try:
        assert kb.sync("web", full=True).added == 1
        assert fn.calls[0]["full"] is True
        assert kb.start_auto_sync() is True
        assert fn.started.wait(5)
        assert kb.status().next_sync is not None
    finally:
        kb.close()
    assert kb.auto_sync is not None
    assert not kb.auto_sync.started  # close() stops the scheduler


def test_knowledge_base_without_schedule(tmp_path: Path) -> None:
    kb = make_kb(tmp_path, schedule=None)
    try:
        assert kb.start_auto_sync() is False
        status = kb.status()
        assert (status.schedule, status.next_sync, status.overdue) == (None, None, False)
    finally:
        kb.close()


def test_knowledge_base_agent_answers(tmp_path: Path) -> None:
    kb = make_kb(tmp_path)
    try:
        kb.index.store.add_texts(
            ["Jane.pdf > page 1\n\nJane knows Go."], metadatas=[{"file_name": "Jane.pdf"}]
        )
        reply = kb.agent(FakeLLM(answer="Jane knows Go [1].")).ask("who knows Go?", thread_id="t")
        assert reply.citations[0]["file_name"] == "Jane.pdf"
    finally:
        kb.close()


def test_knowledge_base_open_uses_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        drive_agent.embeddings, "build_embeddings", lambda s: DeterministicFakeEmbedding(size=8)
    )
    kb = KnowledgeBase.open(
        load_settings(overrides={"vector_store.url": ":memory:", "retrieval.hybrid": False})
    )
    try:
        assert kb.index.count() == 0
    finally:
        kb.close()


# --- CLI: sync --watch, status row ---------------------------------------------------------


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        drive_agent.embeddings, "build_embeddings", lambda s: DeterministicFakeEmbedding(size=8)
    )


def test_watch_runs_auto_sync_until_stopped(
    monkeypatch: pytest.MonkeyPatch, fake_embeddings: None
) -> None:
    fn = Recorder()
    monkeypatch.setattr(kb_module, "sync_now", fn)
    monkeypatch.setenv("SYNC_SCHEDULE", HOURLY)

    def wait_then_stop() -> None:
        assert fn.started.wait(5)  # overdue on start: catch-up sync ran
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_wait_forever", wait_then_stop)
    result = CliRunner().invoke(cli.app, ["sync", "--watch"])
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())  # Rich wraps long lines
    assert f"Auto-sync: `{HOURLY}` (UTC)" in out
    assert "catching up now (overdue)" in out
    assert "Stopped." in out


def test_watch_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    result = CliRunner().invoke(cli.app, ["sync", "--watch"])  # SYNC_SCHEDULE=off in tests
    assert result.exit_code == 1
    assert "Auto-sync is off" in result.output
    assert CliRunner().invoke(cli.app, ["sync", "--watch", "--full"]).exit_code == 2


def test_status_shows_auto_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "auto-sync" in (off := CliRunner().invoke(cli.app, ["status"]).output)
    assert "│ off" in off
    monkeypatch.setenv("SYNC_SCHEDULE", HOURLY)
    out = CliRunner().invoke(cli.app, ["status"]).output
    assert HOURLY in out
    assert "overdue" in out  # never synced

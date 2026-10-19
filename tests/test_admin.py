"""Admin portal: the monitoring files bots write, what the portal reads, and its pages."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from drive_agent.core.config import load_settings
from drive_agent.core.monitor import (
    ACTIVITY_FILE,
    HEARTBEAT_FILE,
    SYNC_PROGRESS_FILE,
    Heartbeat,
    SyncProgress,
    read_activity,
    read_json,
    record_activity,
)
from drive_agent.indexing.state import STATE_DB, SQLiteRecordManager
from drive_agent.interfaces import admin as admin_portal
from drive_agent.interfaces.admin.data import (
    DEFAULT_BOT,
    activity_stats,
    bot_names,
    health_of,
    log_files,
    overview,
    tail,
)
from drive_agent.interfaces.cli import app as cli_app
from drive_agent.vectorstores.location import namespace_for

APP = str(Path(admin_portal.__file__).with_name("app.py"))
NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def make_bot_env(tmp_path: Path, name: str) -> Path:
    data_dir = tmp_path / "data" / name
    (tmp_path / "bots").mkdir(exist_ok=True)
    (tmp_path / "bots" / f"{name}.env").write_text(
        f"DRIVE_AGENT__DATA_DIR={data_dir}\n"
        f"DRIVE_AGENT__VECTOR_STORE__COLLECTION=drive_agent_{name}\n"
        f"SLACK_COMMAND=/drive-{name}\n"
    )
    return data_dir


# --- files the bots write ----------------------------------------------------------


def test_heartbeat_marks_start_and_stop(tmp_path: Path) -> None:
    beat = Heartbeat(tmp_path, "slack", every=60).start()
    running = read_json(tmp_path / HEARTBEAT_FILE)
    assert running is not None
    assert running["interface"] == "slack"
    assert "stopped_at" not in running
    beat.stop()
    stopped = read_json(tmp_path / HEARTBEAT_FILE)
    assert stopped is not None
    assert stopped["stopped_at"]


def test_activity_round_trip_keeps_newest(tmp_path: Path) -> None:
    for i in range(5):
        record_activity(tmp_path, {"kind": "question", "n": i})
    (tmp_path / ACTIVITY_FILE).open("a").write("not json\n")
    events = read_activity(tmp_path, limit=3)
    assert [e["n"] for e in events] == [3, 4]  # the bad line is skipped, not fatal
    assert all(e["at"] for e in events)


def test_sync_progress_counts_files(tmp_path: Path) -> None:
    progress = SyncProgress(tmp_path)
    progress.file_done(failed=False, chunks_added=4)
    progress.file_done(failed=True, chunks_added=0)
    saved = read_json(tmp_path / SYNC_PROGRESS_FILE)
    assert saved is not None
    assert (saved["state"], saved["files_done"], saved["files_failed"]) == ("running", 2, 1)
    progress.finish(ok=False, error="stopped")
    saved = read_json(tmp_path / SYNC_PROGRESS_FILE)
    assert saved is not None
    assert (saved["state"], saved["error"]) == ("failed", "stopped")


# --- what the portal reads -----------------------------------------------------------


@pytest.mark.parametrize(
    ("heartbeat", "expected"),
    [
        (None, "never started"),
        ({"updated_at": iso(NOW - timedelta(seconds=20))}, "running"),
        ({"updated_at": iso(NOW - timedelta(minutes=10))}, "not responding"),
        ({"updated_at": iso(NOW), "stopped_at": iso(NOW)}, "stopped"),
    ],
)
def test_health(heartbeat: dict[str, str] | None, expected: str) -> None:
    assert health_of(heartbeat, NOW) == expected


def test_activity_stats_cover_the_window() -> None:
    events = [
        {"kind": "question", "at": iso(NOW - timedelta(hours=30)), "outcome": "answered"},
        {"kind": "question", "at": iso(NOW - timedelta(hours=1)), "outcome": "answered",
         "seconds": 10, "user": "U1"},
        {"kind": "question", "at": iso(NOW - timedelta(hours=2)), "outcome": "not_found",
         "seconds": 20, "user": "U1"},
        {"kind": "question", "at": iso(NOW - timedelta(hours=3)), "outcome": "error",
         "user": "U2"},
    ]  # fmt: skip
    stats = activity_stats(events, since=NOW - timedelta(hours=24))
    assert (stats.questions, stats.answered, stats.not_found, stats.errors) == (3, 1, 1, 1)
    assert (stats.avg_seconds, stats.users) == (15.0, 2)


def test_overview_reads_every_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNC_SCHEDULE", "0 */6 * * *")
    data_dir = make_bot_env(tmp_path, "hr")
    settings = load_settings(extra_env_file=Path("bots/hr.env"))
    records = SQLiteRecordManager(namespace_for(settings), data_dir / STATE_DB)
    records.update(["k1", "k2", "k3"], group_ids=["f1", "f1", "f2"])
    records.set_meta("last_sync", {"at": iso(NOW - timedelta(hours=1)), "files": 2, "failed": 1})
    records.close()
    (data_dir / HEARTBEAT_FILE).write_text(json.dumps({"updated_at": iso(NOW)}))
    SyncProgress(data_dir).file_done(failed=False, chunks_added=3)
    record_activity(data_dir, {"kind": "question", "outcome": "answered", "user": "U1"})

    bot = overview("hr", now=datetime.now(UTC))
    assert bot.config_error is None
    assert (bot.index.chunks, bot.index.files, bot.index.failed) == (3, 2, 1)
    assert bot.syncing
    assert bot.sync is not None
    assert bot.sync["chunks_added"] == 3
    assert bot.last_24h.questions == 1
    assert bot.next_sync == NOW  # last sync 11:00, every 6 h -> 12:00


def test_running_bot_reports_its_effective_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNC_SCHEDULE", "0 */6 * * *")  # configured on...
    data_dir = make_bot_env(tmp_path, "hr")
    Heartbeat(data_dir, "slack", details={"schedule": None}).beat()  # ...but running with it off
    assert overview("hr").schedule is None
    Heartbeat(data_dir, "slack", details={"schedule": None}).stop()  # not running: show config
    assert overview("hr").schedule == "0 */6 * * *"


def test_overview_of_a_bot_that_never_ran(tmp_path: Path) -> None:
    make_bot_env(tmp_path, "new")
    bot = overview("new")
    assert (bot.health, bot.index.chunks, bot.sync) == ("never started", 0, None)


def test_bot_names_fall_back_to_default(tmp_path: Path) -> None:
    assert bot_names(tmp_path / "bots") == [DEFAULT_BOT]
    make_bot_env(tmp_path, "b")
    make_bot_env(tmp_path, "a")
    assert bot_names(tmp_path / "bots") == ["a", "b"]


def test_tail_filters_and_flattens_progress_bars(tmp_path: Path) -> None:
    log = tmp_path / "bot.log"
    log.write_text("start\nINFO one\nbar 10%\rbar 100%\nERROR two\nINFO three\n")
    assert tail(log, 3) == ["bar 100%", "ERROR two", "INFO three"]
    assert tail(log, 10, "error") == ["ERROR two"]
    assert tail(tmp_path / "missing.log")[0].startswith("(cannot read")


def test_log_files_newest_first(tmp_path: Path) -> None:
    (tmp_path / "old.log").write_text("x")
    (tmp_path / "notes.md").write_text("x")
    new = tmp_path / "new.log"
    new.write_text("x")
    import os

    os.utime(tmp_path / "old.log", (1, 1))
    assert log_files(tmp_path) == [new, tmp_path / "old.log"]
    assert log_files(None) == []


# --- CLI and pages -------------------------------------------------------------------


def test_admin_refuses_public_bind_without_password() -> None:
    result = CliRunner().invoke(cli_app, ["admin", "--host", "0.0.0.0"])  # noqa: S104
    assert result.exit_code == 2
    assert "without a password" in result.output


def test_admin_is_not_per_bot(tmp_path: Path) -> None:
    make_bot_env(tmp_path, "hr")
    result = CliRunner().invoke(cli_app, ["--bot", "hr", "admin"])
    assert result.exit_code == 2


def activity() -> None:
    from drive_agent.interfaces.admin.pages import activity_page

    activity_page()


def logs() -> None:
    from drive_agent.interfaces.admin.pages import logs_page

    logs_page()


@pytest.fixture
def portal_files(tmp_path: Path) -> Path:
    data_dir = make_bot_env(tmp_path, "hr")
    record_activity(
        data_dir, {"kind": "question", "outcome": "not_found", "user": "U1", "seconds": 4.2}
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "hr.log").write_text("noise\nBolt app is running!\n")
    (tmp_path / ".env").write_text(f"LOG_DIR={logs_dir}\n")
    return data_dir


@pytest.mark.usefixtures("portal_files")
def test_bots_page() -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [s.value for s in at.subheader] == ["⚪ hr"]
    assert [m.value for m in at.metric] == ["0", "0", "1", "100%"]


@pytest.mark.usefixtures("portal_files")
def test_activity_page_hides_question_text_unless_logged() -> None:
    at = AppTest.from_function(activity).run()
    assert not at.exception
    table = at.dataframe[0].value
    assert list(table["outcome"]) == ["not_found"]
    assert "question" not in table.columns
    assert "SLACK_LOG_QUESTIONS" in at.caption[0].value


@pytest.mark.usefixtures("portal_files")
def test_logs_page_tails_and_filters() -> None:
    at = AppTest.from_function(logs).run()
    assert not at.exception
    assert "Bolt app is running!" in at.code[0].value
    at.text_input[0].input("bolt").run()
    assert at.code[0].value == "Bolt app is running!"


def test_portal_password(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("ADMIN_PASSWORD=s3cret\n")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.subheader
    assert at.text_input[0].label == "Password"
    at.text_input[0].input("nope").run()
    assert at.error[0].value == "Wrong password."
    at.text_input[0].input("s3cret").run()
    assert not at.exception
    assert at.title[0].value == "Bots"

"""Start the Slack bot on top of a KnowledgeBase (Socket Mode or HTTP mode)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from drive_agent.core.config import ConfigError, Settings
from drive_agent.core.knowledge_base import KnowledgeBase
from drive_agent.core.monitor import Heartbeat, record_activity
from drive_agent.drive.auth import DriveAuthError
from drive_agent.indexing.scheduler import SyncBusy
from drive_agent.indexing.sync import SyncError, summarize
from drive_agent.interfaces.slack_bot.bot import SlackBot, register

FEEDBACK_FILE = "feedback.jsonl"


def required_tokens(settings: Settings) -> dict[str, str]:
    """The Slack credentials for the configured mode; raises ConfigError if any is missing."""
    secrets, mode = settings.secrets, settings.interfaces.slack.mode
    needed = {"SLACK_BOT_TOKEN": secrets.slack_bot_token}
    if mode == "socket":
        needed["SLACK_APP_TOKEN"] = secrets.slack_app_token
    else:
        needed["SLACK_SIGNING_SECRET"] = secrets.slack_signing_secret
    missing = [name for name, value in needed.items() if value is None]
    if missing:
        raise ConfigError(
            f"Slack {mode} mode needs {', '.join(missing)} in .env (see docs/slack-setup.md)."
        )
    return {name: value.get_secret_value() for name, value in needed.items() if value is not None}


def _utc(moment: datetime | None) -> str:
    return moment.strftime("%Y-%m-%d %H:%M UTC") if moment else "never"


def status_text(kb: KnowledgeBase) -> str:
    status = kb.status()
    lines = [
        f"*Index:* {status.chunks} chunks from {status.files} files · "
        f"last sync {_utc(status.last_sync)}"
        + (" · :arrows_counterclockwise: syncing now" if status.syncing else "")
    ]
    if status.schedule:
        lines.append(f"*Auto-sync:* `{status.schedule}` (UTC) · next {_utc(status.next_sync)}")
    else:
        lines.append("*Auto-sync:* off")
    lines.append(f"*Store:* {kb.index.description}")
    return "\n".join(lines)


def make_bot(kb: KnowledgeBase) -> SlackBot:
    def sync() -> str:
        try:
            return ":white_check_mark: Sync done: " + summarize(kb.sync("slack"))
        except SyncBusy:
            return ":hourglass_flowing_sand: A sync is already running; try again in a minute."
        except (DriveAuthError, SyncError) as exc:
            return f":warning: Sync failed: {exc}"

    return SlackBot(
        kb.agent(),
        kb.settings.interfaces.slack,
        status=lambda: status_text(kb),
        sync=sync,
        feedback_path=kb.settings.data_dir / FEEDBACK_FILE,
        activity=lambda event: record_activity(kb.settings.data_dir, event),
    )


def run(kb: KnowledgeBase) -> None:
    """Serve Slack until interrupted. Auto-sync is the caller's choice (KnowledgeBase)."""
    from slack_bolt import App

    tokens = required_tokens(kb.settings)
    slack = kb.settings.interfaces.slack
    app: Any = App(
        token=tokens["SLACK_BOT_TOKEN"], signing_secret=tokens.get("SLACK_SIGNING_SECRET")
    )
    register(app, make_bot(kb))
    # Read by the admin portal; the schedule is the effective one (env can override the file).
    heartbeat = Heartbeat(
        kb.settings.data_dir,
        "slack",
        details={"schedule": kb.settings.ingest.schedule_cron if kb.auto_sync else None},
    ).start()
    try:
        if slack.mode == "socket":
            from slack_bolt.adapter.socket_mode import SocketModeHandler

            SocketModeHandler(app, tokens["SLACK_APP_TOKEN"]).start()
        else:
            app.start(port=slack.http_port)
    finally:
        heartbeat.stop()


__all__ = ["make_bot", "required_tokens", "run", "status_text"]

"""Build the agent and start the Slack bot in Socket Mode or HTTP mode."""

from __future__ import annotations

import logging
from typing import Any

from drive_agent.core.agent import DriveAgent
from drive_agent.core.config import ConfigError, Settings
from drive_agent.drive.auth import DriveAuthError
from drive_agent.indexing.state import STATE_DB, SQLiteRecordManager
from drive_agent.indexing.sync import SyncError, summarize, sync_now
from drive_agent.interfaces.slack_bot.bot import SlackBot, register
from drive_agent.vectorstores import VectorIndex

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


def status_text(settings: Settings, vector_index: VectorIndex) -> str:
    records = SQLiteRecordManager(vector_index.namespace, settings.data_dir / STATE_DB)
    try:
        last = records.get_meta("last_sync") or {}
    finally:
        records.close()
    when = last.get("at", "never").replace("T", " ")[:16]
    return (
        f"*Index:* {vector_index.count()} chunks from {last.get('files', 0)} files · "
        f"last sync {when} UTC\n*Store:* {vector_index.description}"
    )


def make_bot(settings: Settings, agent: DriveAgent, vector_index: VectorIndex) -> SlackBot:
    def sync() -> str:
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        try:
            return ":white_check_mark: Sync done: " + summarize(sync_now(settings, vector_index))
        except (DriveAuthError, SyncError) as exc:
            return f":warning: Sync failed: {exc}"

    return SlackBot(
        agent,
        settings.interfaces.slack,
        status=lambda: status_text(settings, vector_index),
        sync=sync,
        feedback_path=settings.data_dir / FEEDBACK_FILE,
    )


def run(settings: Settings, agent: DriveAgent, vector_index: VectorIndex) -> None:
    """Start serving; blocks until interrupted."""
    from slack_bolt import App

    tokens = required_tokens(settings)
    slack = settings.interfaces.slack
    app: Any = App(
        token=tokens["SLACK_BOT_TOKEN"],
        signing_secret=tokens.get("SLACK_SIGNING_SECRET"),
    )
    register(app, make_bot(settings, agent, vector_index))
    if slack.mode == "socket":
        from slack_bolt.adapter.socket_mode import SocketModeHandler

        SocketModeHandler(app, tokens["SLACK_APP_TOKEN"]).start()
    else:
        app.start(port=slack.http_port)


__all__ = ["make_bot", "required_tokens", "run", "status_text"]

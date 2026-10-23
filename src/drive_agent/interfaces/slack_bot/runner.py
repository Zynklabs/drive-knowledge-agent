"""Start the Slack bot on top of a KnowledgeBase (Socket Mode or HTTP mode)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

from drive_agent.core.config import ConfigError, Settings, SlackSettings
from drive_agent.core.knowledge_base import KnowledgeBase
from drive_agent.core.monitor import Heartbeat, record_activity
from drive_agent.drive.auth import DriveAuthError
from drive_agent.indexing.scheduler import SyncBusy
from drive_agent.indexing.sync import SyncError, summarize
from drive_agent.interfaces.slack_bot.bot import SlackBot, register

FEEDBACK_FILE = "feedback.jsonl"

log = logging.getLogger(__name__)


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


SLACK_WINDOW_SECONDS = 600
"""How far back the connection's drops and errors are counted for the admin portal."""


class SlackConnection:
    """Tracks a Socket Mode connection for the admin portal.

    The connection can drop and keep failing while the process itself stays healthy (e.g. a
    reconnect loop after the network changes), so the process heartbeat alone can't show it.
    Listens to the client's message, error and close hooks; ``status()`` goes in each beat.
    The hooks take slack_sdk's websocket-client signatures (the socket first) as well as the
    builtin client's, so the message or error is always the last argument.
    """

    def __init__(self, client: Any, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._errors: deque[tuple[float, str]] = deque()
        self._drops: deque[float] = deque()
        self._last_message_at: float | None = None
        self._connected_at: float | None = None
        self._client: Any = None
        self.watch(client)

    def watch(self, client: Any) -> None:
        """Follow ``client`` from now on; the drops and errors seen so far were the old one's."""
        with self._lock:
            self._client = client
            self._errors.clear()
            self._drops.clear()
            self._connected_at = None
        client.on_message_listeners.append(self._on_message)
        client.on_error_listeners.append(self._on_error)
        client.on_close_listeners.append(self._on_close)

    def _on_message(self, *args: Any) -> None:
        message = str(args[-1])
        now = self._clock()
        with self._lock:
            self._last_message_at = now
            if '"type":"hello"' in message.replace(" ", ""):  # Slack greets each new connection
                self._connected_at = now

    def _on_error(self, *args: Any) -> None:
        with self._lock:
            self._errors.append((self._clock(), type(args[-1]).__name__))

    def _on_close(self, *args: Any) -> None:
        with self._lock:
            self._drops.append(self._clock())

    def status(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            while self._errors and self._errors[0][0] < now - SLACK_WINDOW_SECONDS:
                self._errors.popleft()
            while self._drops and self._drops[0] < now - SLACK_WINDOW_SECONDS:
                self._drops.popleft()
            return {
                "slack": {
                    "connected": bool(self._client.is_connected()),
                    "connected_since": _iso(self._connected_at),
                    "last_message_at": _iso(self._last_message_at),
                    "errors_10m": len(self._errors),
                    "drops_10m": len(self._drops),
                    "last_error": self._errors[-1][1] if self._errors else None,
                }
            }

    def trouble(self, max_failures: int) -> str | None:
        """Why the connection looks stuck right now (None if it looks fine).

        Drops and errors both count: a broken connection usually brings one of each, but after
        a network blip slack_sdk can fail every attempt with an error and never report a close.
        """
        slack = self.status()["slack"]
        drops, errors = slack["drops_10m"], slack["errors_10m"]
        failing = bool(max_failures) and max(drops, errors) >= max_failures
        if slack["connected"] and not failing:
            return None
        reason = "disconnected" if not slack["connected"] else "unstable"
        reason += f": {drops} drops, {errors} errors in 10 min"
        return reason + (f", last error {slack['last_error']}" if slack["last_error"] else "")


WatchdogAction = Literal["reconnect", "restart"]

WATCHDOG_CHECK_SECONDS = 15
RESTART_EXIT_CODE = 75
"""EX_TEMPFAIL: a non-zero exit, so launchd (KeepAlive), systemd or Docker starts the bot again."""


class SlackWatchdog:
    """Decides when a stuck Socket Mode connection needs a fresh client or a process restart.

    slack_sdk reconnects by itself, but it can get stuck: on 2026-10-23 every new connection
    broke at once (BrokenPipeError) for five hours, while a plain restart fixed it at once.
    Trouble lasting ``watchdog_grace_seconds`` first gets a fresh client; if trouble comes back
    before the fresh client has held for a whole window, the process should exit. The exit
    waits while ``busy()`` (a sync is running) so it never cuts a sync short.
    """

    def __init__(
        self,
        connection: SlackConnection,
        settings: SlackSettings,
        *,
        busy: Callable[[], bool],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._connection = connection
        self._max_failures = settings.watchdog_failures
        self._grace = settings.watchdog_grace_seconds
        self._busy = busy
        self._clock = clock
        self._since: float | None = None
        self._reconnected_at: float | None = None
        self._deferred = False
        self.reason: str | None = None
        """What was wrong when ``check()`` last asked for action."""

    def check(self) -> WatchdogAction | None:
        now = self._clock()
        trouble = self._connection.trouble(self._max_failures)
        if trouble is None:
            self._since, self._deferred = None, False
            if self._reconnected_at is not None and now - self._reconnected_at >= (
                SLACK_WINDOW_SECONDS
            ):
                self._reconnected_at = None  # the fresh client held: start over next time
            return None
        if self._since is None:
            self._since = now
        if now - self._since < self._grace:
            return None
        self.reason = trouble
        if self._reconnected_at is None:
            self._reconnected_at, self._since = now, None
            return "reconnect"
        if self._busy():
            if not self._deferred:
                log.warning(
                    "Slack connection still stuck (%s); restarting once the running sync ends",
                    trouble,
                )
                self._deferred = True
            return None
        return "restart"


def serve_socket(
    handler: Any,
    new_handler: Callable[[], Any],
    connection: SlackConnection,
    watchdog: SlackWatchdog | None,
    *,
    record: Callable[[dict[str, Any]], None],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Connect ``handler`` and serve until interrupted, acting on the watchdog's checks.

    A reconnect swaps in ``new_handler()`` (a fresh client and socket); a restart raises
    ``SystemExit(RESTART_EXIT_CODE)`` for the service manager. Both go in the activity log.
    """
    handler.connect()
    log.info("Slack Socket Mode connected")
    try:
        while True:
            sleep(WATCHDOG_CHECK_SECONDS)
            action = watchdog.check() if watchdog is not None else None
            if action is None:
                continue
            reason = watchdog.reason if watchdog is not None else None
            record({"kind": "slack_restart", "action": action, "reason": reason})
            if action == "restart":
                log.error(
                    "Slack connection still stuck after a fresh client (%s); exiting with "
                    "code %d so the service manager restarts the bot",
                    reason,
                    RESTART_EXIT_CODE,
                )
                raise SystemExit(RESTART_EXIT_CODE)
            log.warning("Slack connection stuck (%s); reconnecting with a fresh client", reason)
            _close(handler)
            handler = new_handler()
            connection.watch(handler.client)
            try:
                handler.connect()
            except Exception:  # the watchdog sees it disconnected and escalates
                log.exception("Slack reconnect failed")
    finally:
        _close(handler)


def _close(handler: Any) -> None:
    try:
        handler.close()
    except Exception:  # closing a broken socket can fail too; the client is dropped anyway
        log.warning("closing the Slack client failed", exc_info=True)


def _iso(moment: float | None) -> str | None:
    if moment is None:
        return None
    return datetime.fromtimestamp(moment, UTC).isoformat(timespec="seconds")


def run(kb: KnowledgeBase) -> None:
    """Serve Slack until interrupted. Auto-sync is the caller's choice (KnowledgeBase)."""
    from slack_bolt import App

    tokens = required_tokens(kb.settings)
    slack = kb.settings.interfaces.slack
    app: Any = App(
        token=tokens["SLACK_BOT_TOKEN"], signing_secret=tokens.get("SLACK_SIGNING_SECRET")
    )
    register(app, make_bot(kb))
    handler: Any = None
    connection: SlackConnection | None = None
    if slack.mode == "socket":
        # websocket-client, not slack_sdk's builtin client: after a network blip the builtin
        # one kept failing every new connection with BrokenPipeError until a restart.
        from slack_bolt.adapter.socket_mode.websocket_client import SocketModeHandler

        def new_handler() -> Any:
            return SocketModeHandler(app, tokens["SLACK_APP_TOKEN"])

        handler = new_handler()
        connection = SlackConnection(handler.client)
    # Read by the admin portal; the schedule is the effective one (env can override the file).
    heartbeat = Heartbeat(
        kb.settings.data_dir,
        "slack",
        details={"schedule": kb.settings.ingest.schedule_cron if kb.auto_sync else None},
        probe=connection.status if connection is not None else None,
    ).start()
    try:
        if handler is not None and connection is not None:
            watchdog = (
                SlackWatchdog(connection, slack, busy=lambda: kb.coordinator.running)
                if slack.watchdog
                else None
            )
            serve_socket(
                handler,
                new_handler,
                connection,
                watchdog,
                record=lambda event: record_activity(kb.settings.data_dir, event),
            )
        else:
            app.start(port=slack.http_port)
    finally:
        heartbeat.stop()


__all__ = ["make_bot", "required_tokens", "run", "serve_socket", "status_text"]

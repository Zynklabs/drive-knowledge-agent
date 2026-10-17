"""Slack bot handlers. ``SlackBot`` holds the logic; ``register`` wires it to a Bolt ``App``.

- @mention in a channel, or a direct message: answered in the message's thread. Each Slack
  thread is one conversation (memory keyed by channel + thread).
- ``/drive-agent [ask] <question>`` (the command is configurable per bot): private
  (ephemeral) answer; ``status``; ``help``; ``sync`` for admins only.
- 👍 / 👎 buttons under answers are appended to ``<data_dir>/feedback.jsonl``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from drive_agent.core.agent import DriveAgent
from drive_agent.core.config import ConfigError, SlackSettings
from drive_agent.interfaces.slack_bot.formatting import (
    FEEDBACK_DOWN,
    FEEDBACK_UP,
    SECTION_LIMIT,
    fallback_text,
    reply_blocks,
    strip_mentions,
    to_mrkdwn,
)
from drive_agent.llm import LLMError

log = logging.getLogger(__name__)

SEARCHING = ":mag: Searching your Drive…"


def help_text(command: str = "/drive-agent") -> str:
    return (
        "*Drive Knowledge Agent*\n"
        "• Mention me in a channel or send me a DM with a question; I answer in the thread "
        "and remember the thread's conversation (mention me again for follow-ups).\n"
        f"• `{command} <question>` or `{command} ask <question>`: a private answer.\n"
        f"• `{command} status`: what is indexed.\n"
        f"• `{command} sync`: re-index the Drive folders (admins only)."
    )


HELP = help_text()
SEEN_EVENTS = 1000


class SlackBot:
    def __init__(
        self,
        agent: DriveAgent,
        settings: SlackSettings,
        *,
        status: Callable[[], str],
        sync: Callable[[], str],
        feedback_path: Path,
        update_interval: float = 1.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._agent = agent
        self._settings = settings
        self._status = status
        self._sync = sync
        self._feedback_path = feedback_path
        self._interval = update_interval
        self._clock = clock
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def command(self) -> str:
        return self._settings.command

    # --- guards -------------------------------------------------------------

    def is_duplicate(self, event_id: str | None) -> bool:
        """Slack redelivers events it thinks were missed; answer each one only once."""
        if not event_id:
            return False
        with self._lock:
            if event_id in self._seen:
                return True
            self._seen[event_id] = None
            if len(self._seen) > SEEN_EVENTS:
                self._seen.popitem(last=False)
        return False

    def _allowed_channel(self, channel: str) -> bool:
        allowed = self._settings.allowed_channels
        return not allowed or channel in allowed

    # --- events -------------------------------------------------------------

    def handle_mention(self, event: dict[str, Any], client: Any) -> None:
        if event.get("bot_id"):
            return
        channel = event["channel"]
        thread_ts = event.get("thread_ts") or event["ts"]
        if not self._allowed_channel(channel):
            client.chat_postMessage(
                channel=channel, thread_ts=thread_ts, text="I'm not enabled in this channel."
            )
            return
        question = strip_mentions(event.get("text", ""))
        if not question:
            client.chat_postMessage(
                channel=channel, thread_ts=thread_ts, text=help_text(self.command)
            )
            return
        self.answer_in_thread(client, channel, thread_ts, question)

    def handle_dm(self, event: dict[str, Any], client: Any) -> None:
        # Ignore other bots (and ourselves), edits, joins and other message subtypes.
        if event.get("channel_type") != "im" or event.get("bot_id") or event.get("subtype"):
            return
        channel = event["channel"]
        thread_ts = event.get("thread_ts") or event["ts"]
        if not self._settings.allow_dms:
            client.chat_postMessage(
                channel=channel, thread_ts=thread_ts, text="Direct messages are turned off."
            )
            return
        question = event.get("text", "").strip()
        if question:
            self.answer_in_thread(client, channel, thread_ts, question)

    def answer_in_thread(self, client: Any, channel: str, thread_ts: str, question: str) -> None:
        placeholder = client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=SEARCHING)
        ts = placeholder["ts"]
        streamed: list[str] = []
        last_update = self._clock()

        def on_text(text: str) -> None:
            nonlocal last_update
            streamed.append(text)
            now = self._clock()
            if now - last_update >= self._interval:  # stay well inside chat.update rate limits
                last_update = now
                partial = to_mrkdwn("".join(streamed))[:SECTION_LIMIT]
                client.chat_update(channel=channel, ts=ts, text=partial + " …")

        thread_id = f"slack:{channel}:{thread_ts}"
        try:
            reply = self._agent.ask(question, thread_id=thread_id, on_text=on_text)
        except (LLMError, ConfigError) as exc:
            client.chat_update(channel=channel, ts=ts, text=f":warning: {exc}")
            return
        except Exception:
            log.exception("answering %r failed", question)
            client.chat_update(
                channel=channel, ts=ts, text=":warning: Something went wrong; see the bot's logs."
            )
            return
        client.chat_update(
            channel=channel,
            ts=ts,
            text=fallback_text(reply.answer),
            blocks=reply_blocks(reply, feedback_value=thread_id),
        )

    # --- slash command ------------------------------------------------------

    def handle_command(self, command: dict[str, Any], respond: Callable[..., Any]) -> None:
        text = command.get("text", "").strip()
        sub, _, rest = text.partition(" ")
        sub = sub.lower()
        if sub in {"", "help"}:
            respond(text=help_text(self.command), response_type="ephemeral")
        elif sub == "status":
            respond(text=self._status(), response_type="ephemeral")
        elif sub == "sync":
            admins = self._settings.admin_users
            if "*" not in admins and command.get("user_id") not in admins:
                respond(
                    text="Only admins can sync (interfaces.slack.admin_users).",
                    response_type="ephemeral",
                )
                return
            respond(
                text=":arrows_counterclockwise: Syncing the Drive folders…",
                response_type="ephemeral",
            )
            respond(text=self._sync(), response_type="ephemeral")
        else:
            question = rest.strip() if sub == "ask" else text
            if not question:
                respond(text=f"Usage: `{self.command} ask <question>`", response_type="ephemeral")
                return
            self._answer_command(command, question, respond)

    def _answer_command(
        self, command: dict[str, Any], question: str, respond: Callable[..., Any]
    ) -> None:
        # Slash commands are one-off: a fresh thread per question, so no memory between them.
        thread_id = f"slack-cmd:{command.get('user_id', 'unknown')}:{uuid.uuid4().hex[:8]}"
        try:
            reply = self._agent.ask(question, thread_id=thread_id)
        except (LLMError, ConfigError) as exc:
            respond(text=f":warning: {exc}", response_type="ephemeral")
            return
        respond(
            text=fallback_text(reply.answer),
            blocks=reply_blocks(reply, feedback_value=thread_id),
            response_type="ephemeral",
        )

    # --- feedback -----------------------------------------------------------

    def handle_feedback(self, body: dict[str, Any], respond: Callable[..., Any]) -> None:
        action = body["actions"][0]
        record = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "rating": "up" if action["action_id"] == FEEDBACK_UP else "down",
            "thread_id": action.get("value"),
            "user": (body.get("user") or {}).get("id"),
            "channel": (body.get("channel") or {}).get("id"),
            "message_ts": (body.get("message") or {}).get("ts"),
        }
        self._feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._feedback_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        respond(text="Thanks for the feedback!", response_type="ephemeral", replace_original=False)


def register(app: Any, bot: SlackBot) -> None:
    """Attach the bot's handlers to a Slack Bolt ``App``."""

    @app.event("app_mention")
    def _mention(event: dict[str, Any], client: Any, body: dict[str, Any]) -> None:
        if not bot.is_duplicate(body.get("event_id")):
            bot.handle_mention(event, client)

    @app.event("message")
    def _message(event: dict[str, Any], client: Any, body: dict[str, Any]) -> None:
        if not bot.is_duplicate(body.get("event_id")):
            bot.handle_dm(event, client)

    @app.command(bot.command)
    def _command(ack: Callable[[], None], command: dict[str, Any], respond: Any) -> None:
        ack()  # Slack requires an acknowledgement within 3 seconds
        bot.handle_command(command, respond)

    @app.action(FEEDBACK_UP)
    @app.action(FEEDBACK_DOWN)
    def _feedback(ack: Callable[[], None], body: dict[str, Any], respond: Any) -> None:
        ack()
        bot.handle_feedback(body, respond)

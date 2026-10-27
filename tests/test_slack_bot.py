import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from typer.testing import CliRunner

import drive_agent.embeddings
import drive_agent.llm.claude
from drive_agent.core.agent import AgentReply
from drive_agent.core.config import ConfigError, Settings, SlackSettings, load_settings
from drive_agent.drive.auth import DriveAuthError
from drive_agent.indexing.scheduler import SyncBusy
from drive_agent.indexing.sync import SyncReport
from drive_agent.interfaces.cli import app as cli_app
from drive_agent.interfaces.slack_bot import bot as bot_module
from drive_agent.interfaces.slack_bot import formatting, runner
from drive_agent.interfaces.slack_bot.bot import (
    DMS_OFF,
    FAILED,
    HELP,
    NOT_IN_CHANNEL,
    RATE_LIMITED,
    SEARCHING,
    SlackBot,
    register,
)
from drive_agent.llm import LLMError

from .fakes import FakeLLM

REPLY = AgentReply(
    answer="**Jane** knows Go [1] & <Rust> [2].",
    status="answered",
    citations=[
        {"number": 2, "file_name": "Raj.pdf", "page": 2, "web_url": "https://d/2"},
        {"number": 1, "file_name": "Jane|CV.pdf", "page": 1, "web_url": "https://d/1"},
    ],
)


class SlackError(Exception):
    """Shaped like slack_sdk's SlackApiError: the API response is on ``.response``."""

    def __init__(self, error: str) -> None:
        super().__init__(error)
        self.response = {"ok": False, "error": error}


class FakeClient:
    def __init__(self, member_of: set[str] | None = None) -> None:
        self.posts: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.ephemeral: list[dict[str, Any]] = []
        self.member_of = {"C1"} if member_of is None else member_of

    def chat_postEphemeral(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["channel"] not in self.member_of:
            raise SlackError("not_in_channel")
        self.ephemeral.append(kwargs)
        return {"ok": True}

    def chat_postMessage(self, **kwargs: Any) -> dict[str, Any]:
        self.posts.append(kwargs)
        return {"ts": "999.000"}

    def chat_update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeAgent:
    def __init__(self, reply: AgentReply = REPLY, error: Exception | None = None) -> None:
        self.reply, self.error = reply, error
        self.calls: list[tuple[str, str | None]] = []

    def ask(
        self,
        question: str,
        *,
        thread_id: str | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> AgentReply:
        self.calls.append((question, thread_id))
        if self.error:
            raise self.error
        for piece in ("Jane ", "knows ", "Go."):
            if on_text:
                on_text(piece)
        return self.reply


class Responder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def make_bot(
    tmp_path: Path, agent: Any = None, *, clock: Callable[[], float] | None = None, **slack: Any
) -> SlackBot:
    return SlackBot(
        agent or FakeAgent(),
        SlackSettings(**slack),
        status=lambda: "32 chunks",
        sync=lambda: "synced",
        feedback_path=tmp_path / "feedback.jsonl",
        clock=clock or (lambda: 0.0),
    )


def mention(text: str = "<@UBOT> who knows Go?", **extra: Any) -> dict[str, Any]:
    return {
        "type": "app_mention",
        "channel": "C1",
        "ts": "100.1",
        "user": "U1",
        "text": text,
        **extra,
    }


# --- events -------------------------------------------------------------------


def test_mention_answers_in_thread_with_blocks(tmp_path: Path) -> None:
    agent, client = FakeAgent(), FakeClient()
    make_bot(tmp_path, agent).handle_mention(mention(), client)

    assert client.posts == [
        {
            "channel": "C1",
            "thread_ts": "100.1",
            "text": SEARCHING,
            "unfurl_links": False,
            "unfurl_media": False,
        }
    ]
    assert agent.calls == [("who knows Go?", "slack:C1:100.1")]
    final = client.updates[-1]
    assert final["ts"] == "999.000"
    assert final["text"] == REPLY.answer
    section, context, actions = final["blocks"]
    assert section["text"]["text"] == "*Jane* knows Go [1] &amp; &lt;Rust&gt; [2]."
    assert [e["text"] for e in context["elements"]] == [
        "<https://d/1|[1] Jane¦CV.pdf · p.1>",
        "<https://d/2|[2] Raj.pdf · p.2>",
    ]
    assert [e["action_id"] for e in actions["elements"]] == ["feedback_up", "feedback_down"]
    assert actions["elements"][0]["value"] == "slack:C1:100.1"


def test_follow_up_in_thread_shares_conversation(tmp_path: Path) -> None:
    agent = FakeAgent()
    make_bot(tmp_path, agent).handle_mention(
        mention("<@UBOT> and Kubernetes?", ts="200.2", thread_ts="100.1"), FakeClient()
    )
    assert agent.calls == [("and Kubernetes?", "slack:C1:100.1")]


def test_mention_guards(tmp_path: Path) -> None:
    agent, client = FakeAgent(), FakeClient()
    bot = make_bot(tmp_path, agent, allowed_channels=["C9"])
    bot.handle_mention(mention(bot_id="B1"), client)  # other bots: ignored
    assert client.posts == []
    bot.handle_mention(mention(), client)  # C1 not allowed
    assert client.posts[-1]["text"] == "I'm not enabled in this channel."
    make_bot(tmp_path, agent).handle_mention(mention("<@UBOT>"), client)  # empty question
    assert client.posts[-1]["text"] == HELP
    assert agent.calls == []


def dm(**extra: Any) -> dict[str, Any]:
    return {
        "channel": "D1",
        "channel_type": "im",
        "ts": "5.5",
        "user": "U1",
        "text": "hi?",
        **extra,
    }


def test_direct_messages_when_turned_on(tmp_path: Path) -> None:
    agent, client = FakeAgent(), FakeClient()
    bot = make_bot(tmp_path, agent, allow_dms=True)
    bot.handle_dm(dm(), client)
    bot.handle_dm(dm(channel_type="channel"), client)  # channel messages come as mentions
    bot.handle_dm(dm(subtype="message_changed"), client)
    bot.handle_dm(dm(bot_id="B1"), client)
    assert agent.calls == [("hi?", "slack:D1:5.5")]


def test_direct_messages_are_off_by_default(tmp_path: Path) -> None:
    agent, client = FakeAgent(), FakeClient()
    make_bot(tmp_path, agent).handle_dm(dm(), client)
    assert client.posts[-1]["text"] == DMS_OFF
    assert agent.calls == []


def test_streaming_updates_are_throttled(tmp_path: Path) -> None:
    ticks = iter([0.0, 0.5, 2.0, 2.5])  # start, then one tick per streamed piece
    client = FakeClient()
    make_bot(tmp_path, clock=lambda: next(ticks)).handle_mention(mention(), client)
    partial, final = client.updates
    assert partial["text"] == "Jane knows  …"  # only the update after 1.5 s of silence
    assert "blocks" in final


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LLMError("No Claude credentials found."), FAILED),
        (RuntimeError("boom"), FAILED),
    ],
)
def test_errors_replace_the_placeholder(tmp_path: Path, error: Exception, expected: str) -> None:
    client = FakeClient()
    make_bot(tmp_path, FakeAgent(error=error)).handle_mention(mention(), client)
    assert client.updates == [{"channel": "C1", "ts": "999.000", "text": expected}]


def test_failures_keep_the_question_out_of_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    make_bot(tmp_path, FakeAgent(error=RuntimeError("boom"))).handle_mention(
        mention(), FakeClient()
    )
    assert "answering a question failed" in caplog.text
    assert "who knows Go" not in caplog.text
    caplog.clear()
    make_bot(tmp_path, FakeAgent(error=RuntimeError("boom")), log_questions=True).handle_mention(
        mention(), FakeClient()
    )
    assert "who knows Go" in caplog.text


def test_streamed_partials_drop_links_and_images(tmp_path: Path) -> None:
    class LinkAgent(FakeAgent):
        def ask(self, question: str, **kwargs: Any) -> AgentReply:
            kwargs["on_text"]("See [this](https://evil.example) ![x](https://evil.example/p.png)")
            kwargs["on_text"](" now.")
            return self.reply

    ticks = iter([0.0, 2.0, 4.0])
    client = FakeClient()
    make_bot(tmp_path, LinkAgent(), clock=lambda: next(ticks)).handle_mention(mention(), client)
    assert "evil.example" not in client.updates[0]["text"]


def test_questions_per_hour_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr("drive_agent.interfaces.slack_bot.bot.time.monotonic", lambda: now[0])
    agent = FakeAgent()
    bot = make_bot(tmp_path, agent, questions_per_hour=2)
    for _ in range(3):
        bot.handle_mention(mention(), FakeClient())
    respond = Responder()
    bot.handle_command(command("ask q"), respond, FakeClient())
    assert len(agent.calls) == 2
    assert respond.calls[0]["text"] == RATE_LIMITED.format(limit=2)
    other = FakeClient()
    bot.handle_mention(mention(user="U2"), other)  # the limit is per person
    assert len(agent.calls) == 3
    now[0] += 3600
    bot.handle_mention(mention(), FakeClient())  # an hour later the window has moved on
    assert len(agent.calls) == 4


def test_no_questions_per_hour_limit(tmp_path: Path) -> None:
    agent = FakeAgent()
    bot = make_bot(tmp_path, agent, questions_per_hour=0)
    for _ in range(40):
        bot.handle_mention(mention(), FakeClient())
    assert len(agent.calls) == 40


def test_not_found_has_no_feedback_buttons(tmp_path: Path) -> None:
    reply = AgentReply(answer="I couldn't find anything.", status="not_found")
    client = FakeClient()
    make_bot(tmp_path, FakeAgent(reply)).handle_mention(mention(), client)
    assert [b["type"] for b in client.updates[-1]["blocks"]] == ["section"]


def test_duplicate_events_are_answered_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = make_bot(tmp_path)
    assert bot.is_duplicate("Ev1") is False
    assert bot.is_duplicate("Ev1") is True
    assert bot.is_duplicate(None) is False
    monkeypatch.setattr(bot_module, "SEEN_EVENTS", 2)
    for event_id in ("Ev2", "Ev3"):
        bot.is_duplicate(event_id)
    assert bot.is_duplicate("Ev1") is False  # evicted from the bounded memory


# --- slash command ------------------------------------------------------------


def command(text: str, user: str = "U1", channel: str = "C1") -> dict[str, Any]:
    return {"command": "/drive-agent", "text": text, "user_id": user, "channel_id": channel}


def test_command_question_only_where_the_bot_was_added(tmp_path: Path) -> None:
    agent, respond = FakeAgent(), Responder()
    make_bot(tmp_path, agent).handle_command(command("q", channel="C2"), respond, FakeClient())
    assert respond.calls == [{"text": NOT_IN_CHANNEL, "response_type": "ephemeral"}]
    assert agent.calls == []


def test_command_question_respects_allowed_channels(tmp_path: Path) -> None:
    agent, respond = FakeAgent(), Responder()
    bot = make_bot(tmp_path, agent, allowed_channels=["C9"])
    bot.handle_command(command("q"), respond, FakeClient(member_of={"C1", "C9"}))
    assert respond.calls[0]["text"] == "I'm not enabled in this channel."
    assert agent.calls == []


def test_command_question_in_a_dm_needs_dms_on(tmp_path: Path) -> None:
    agent, respond = FakeAgent(), Responder()
    make_bot(tmp_path, agent).handle_command(command("q", channel="D1"), respond, FakeClient())
    assert respond.calls == [{"text": DMS_OFF, "response_type": "ephemeral"}]
    respond = Responder()
    make_bot(tmp_path, agent, allow_dms=True).handle_command(
        command("q", channel="D1"), respond, FakeClient()
    )
    assert agent.calls[0][0] == "q"


def test_command_question_posts_a_private_searching_notice(tmp_path: Path) -> None:
    client = FakeClient()
    make_bot(tmp_path).handle_command(command("q"), Responder(), client)
    assert client.ephemeral == [{"channel": "C1", "user": "U1", "text": SEARCHING}]


@pytest.mark.parametrize("text", ["", "help", "HELP"])
def test_command_help(tmp_path: Path, text: str) -> None:
    respond = Responder()
    make_bot(tmp_path).handle_command(command(text), respond, FakeClient())
    assert respond.calls == [{"text": HELP, "response_type": "ephemeral"}]


def test_command_status(tmp_path: Path) -> None:
    respond = Responder()
    make_bot(tmp_path).handle_command(command("status"), respond, FakeClient())
    assert respond.calls == [{"text": "32 chunks", "response_type": "ephemeral"}]


def test_command_sync_is_admin_only(tmp_path: Path) -> None:
    respond = Responder()
    make_bot(tmp_path, admin_users=["UADMIN"]).handle_command(
        command("sync"), respond, FakeClient()
    )
    assert "Only admins" in respond.calls[0]["text"]

    respond = Responder()
    make_bot(tmp_path, admin_users=["UADMIN"]).handle_command(
        command("sync", "UADMIN"), respond, FakeClient()
    )
    assert [c["text"] for c in respond.calls][-1] == "synced"


@pytest.mark.parametrize("text", ["ask who knows Go?", "who knows Go?"])
def test_command_question_is_private_and_stateless(tmp_path: Path, text: str) -> None:
    agent, respond = FakeAgent(), Responder()
    make_bot(tmp_path, agent).handle_command(command(text), respond, FakeClient())
    question, thread_id = agent.calls[0]
    assert question == "who knows Go?"
    assert thread_id is not None
    assert thread_id.startswith("slack-cmd:U1:")
    assert respond.calls[0]["response_type"] == "ephemeral"
    assert respond.calls[0]["blocks"][0]["type"] == "section"


def test_command_usage_and_errors(tmp_path: Path) -> None:
    respond = Responder()
    make_bot(tmp_path).handle_command(command("ask"), respond, FakeClient())
    assert "Usage" in respond.calls[0]["text"]
    respond = Responder()
    make_bot(tmp_path, FakeAgent(error=LLMError("rate limited"))).handle_command(
        command("ask q"), respond, FakeClient()
    )
    assert respond.calls[0]["text"] == FAILED


# --- feedback -----------------------------------------------------------------


def test_feedback_is_logged(tmp_path: Path) -> None:
    respond = Responder()
    bot = make_bot(tmp_path)
    for action_id in ("feedback_up", "feedback_down"):
        bot.handle_feedback(
            {
                "actions": [{"action_id": action_id, "value": "slack:C1:100.1"}],
                "user": {"id": "U1"},
                "channel": {"id": "C1"},
                "message": {"ts": "999.000"},
            },
            respond,
        )
    records = [json.loads(line) for line in (tmp_path / "feedback.jsonl").read_text().splitlines()]
    assert [r["rating"] for r in records] == ["up", "down"]
    assert records[0]["thread_id"] == "slack:C1:100.1"
    assert records[0]["user"] == "U1"
    assert respond.calls[0]["replace_original"] is False


def test_register_wires_bolt_listeners(tmp_path: Path) -> None:
    from slack_bolt import App

    app = App(token="xoxb-test", signing_secret="s", token_verification_enabled=False)
    register(app, make_bot(tmp_path))
    assert len(app._listeners) == 5  # mention, message, command, 2 feedback actions


# --- formatting ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("markdown", "mrkdwn"),
    [
        ("**bold** text", "*bold* text"),
        ("## Heading\nbody", "*Heading*\nbody"),
        ("see [the doc](https://x.io/a?b=1&c=2)", "see <https://x.io/a?b=1&amp;c=2|the doc>"),
        ("a < b & c > d", "a &lt; b &amp; c &gt; d"),
        ("- one\n- two\n  - nested", "• one\n• two\n    • nested"),
        ("an *italic* and **bold** word", "an _italic_ and *bold* word"),
        ("~~old~~ new", "~old~ new"),
        ("above\n---\nbelow", "above\n\nbelow"),
        ("```\n- not a list & **raw**\n```", "```\n- not a list &amp; **raw**\n```"),
    ],
)
def test_to_mrkdwn(markdown: str, mrkdwn: str) -> None:
    assert formatting.to_mrkdwn(markdown) == mrkdwn


def test_split_sections_respects_limit() -> None:
    text = "\n\n".join(["x" * 1500] * 5) + "\n\n" + "y\n" * 3000
    pieces = formatting.split_sections(text, limit=2900)
    assert all(len(p) <= 2900 for p in pieces)
    assert "".join(p.replace("\n", "") for p in pieces) == text.replace("\n", "")


def test_many_sources_are_capped() -> None:
    reply = AgentReply(
        answer="a",
        status="answered",
        citations=[{"number": n, "file_name": f"f{n}"} for n in range(1, 15)],
    )
    context = formatting.reply_blocks(reply, feedback_value="t")[1]
    assert len(context["elements"]) == 10
    assert context["elements"][-1]["text"] == "+5 more"


def test_strip_mentions_and_fallback() -> None:
    assert formatting.strip_mentions("<@U123ABC>  hello <@U9>") == "hello"
    assert len(formatting.fallback_text("x" * 1000)) == 300


# --- runner -------------------------------------------------------------------


def test_required_tokens_by_mode() -> None:
    with pytest.raises(ConfigError, match="SLACK_BOT_TOKEN, SLACK_APP_TOKEN"):
        runner.required_tokens(Settings())
    http = Settings.model_validate({"interfaces": {"slack": {"mode": "http"}}})
    with pytest.raises(ConfigError, match="SLACK_SIGNING_SECRET"):
        runner.required_tokens(http)
    from pydantic import SecretStr

    ok = Settings()
    ok.secrets = ok.secrets.model_copy(
        update={"slack_bot_token": SecretStr("xoxb-1"), "slack_app_token": SecretStr("xapp-1")}
    )
    assert runner.required_tokens(ok) == {"SLACK_BOT_TOKEN": "xoxb-1", "SLACK_APP_TOKEN": "xapp-1"}


class FakeKB:
    """Just enough KnowledgeBase for the runner: status, sync and agent."""

    def __init__(self, tmp_path: Path, sync_result: Any = None) -> None:
        from drive_agent.core.knowledge_base import IndexStatus

        self.settings = Settings.model_validate(
            {"data_dir": str(tmp_path), "interfaces": {"slack": {"admin_users": ["*"]}}}
        )
        self.index = type("Index", (), {"description": "qdrant x / y"})()
        self.sync_result = sync_result
        self.status_value = IndexStatus(
            chunks=7,
            files=3,
            failed=0,
            last_sync=datetime(2026, 10, 17, 9, 44, tzinfo=UTC),
            schedule="0 */6 * * *",
            next_sync=datetime(2026, 10, 17, 12, 0, tzinfo=UTC),
            overdue=False,
            syncing=False,
        )

    def status(self) -> Any:
        return self.status_value

    def sync(self, trigger: str = "manual") -> Any:
        if isinstance(self.sync_result, Exception):
            raise self.sync_result
        return self.sync_result

    def agent(self) -> Any:
        return FakeAgent()


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (SyncReport(added=3, unchanged=29), ":white_check_mark: Sync done: +3 added"),
        (SyncBusy(), ":hourglass_flowing_sand: A sync is already running"),
        (DriveAuthError("key not found"), ":warning: Sync failed: key not found"),
    ],
)
def test_bot_sync_goes_through_the_knowledge_base(
    tmp_path: Path, result: Any, expected: str
) -> None:
    bot = runner.make_bot(FakeKB(tmp_path, result))  # type: ignore[arg-type]
    respond = Responder()
    bot.handle_command(
        command("sync", "Uanyone"), respond, FakeClient()
    )  # admin_users ["*"]: everyone
    assert respond.calls[-1]["text"].startswith(expected)


def test_status_text(tmp_path: Path) -> None:
    kb = FakeKB(tmp_path)
    text = runner.status_text(kb)  # type: ignore[arg-type]
    assert text.splitlines() == [
        "*Index:* 7 chunks from 3 files · last sync 2026-10-17 09:44 UTC",
        "*Auto-sync:* `0 */6 * * *` (UTC) · next 2026-10-17 12:00 UTC",
        "*Store:* qdrant x / y",
    ]
    from dataclasses import replace

    kb.status_value = replace(kb.status_value, schedule=None, next_sync=None, syncing=True)
    text = runner.status_text(kb)  # type: ignore[arg-type]
    assert "syncing now" in text
    assert "*Auto-sync:* off" in text


# --- CLI ----------------------------------------------------------------------


def test_cli_slack_requires_tokens() -> None:
    result = CliRunner().invoke(cli_app, ["slack"])
    assert result.exit_code == 1
    assert "SLACK_BOT_TOKEN, SLACK_APP_TOKEN" in result.output


def test_cli_slack_starts_bot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-1")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-1")
    monkeypatch.setattr(
        drive_agent.embeddings, "build_embeddings", lambda s: DeterministicFakeEmbedding(size=8)
    )
    monkeypatch.setattr(drive_agent.llm.claude, "ClaudeLLM", lambda s: FakeLLM())
    started: list[Any] = []
    monkeypatch.setattr(runner, "run", lambda kb: started.append(kb))
    result = CliRunner().invoke(cli_app, ["slack"])
    assert result.exit_code == 0, result.output
    assert "Slack bot running in socket mode" in result.output
    assert "index is empty" in result.output
    assert "Auto-sync: off" in result.output  # tests run with SYNC_SCHEDULE=off
    assert len(started) == 1


# --- activity log (admin portal) -------------------------------------------------


def activity_bot(
    tmp_path: Path, events: list[dict[str, Any]], agent: Any = None, **slack: Any
) -> SlackBot:
    return SlackBot(
        agent or FakeAgent(),
        SlackSettings(**slack),
        status=lambda: "",
        sync=lambda: "",
        feedback_path=tmp_path / "feedback.jsonl",
        activity=events.append,
    )


def test_questions_are_recorded_without_their_text(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    activity_bot(tmp_path, events).handle_mention(mention(), FakeClient())
    (event,) = events
    assert event["outcome"] == "answered"
    assert (event["via"], event["user"], event["channel"]) == ("mention", "U1", "C1")
    assert "question" not in event
    assert event["seconds"] >= 0


def test_question_text_is_opt_in(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    activity_bot(tmp_path, events, log_questions=True).handle_mention(mention(), FakeClient())
    assert events[0]["question"] == "who knows Go?"


def test_failed_answers_are_recorded(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    bot = activity_bot(tmp_path, events, FakeAgent(error=RuntimeError("boom")))
    bot.handle_mention(mention(), FakeClient())
    assert (events[0]["outcome"], events[0]["error"]) == ("error", "RuntimeError")


class FakeSocketClient:
    def __init__(self) -> None:
        self.on_message_listeners: list[Any] = []
        self.on_error_listeners: list[Any] = []
        self.on_close_listeners: list[Any] = []
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected


def test_slack_connection_reports_drops_and_errors_in_a_window() -> None:
    now = [1_000.0]
    client = FakeSocketClient()
    watch = runner.SlackConnection(client, clock=lambda: now[0])
    (on_message,), (on_error,), (on_close,) = (
        client.on_message_listeners,
        client.on_error_listeners,
        client.on_close_listeners,
    )
    on_message('{"type": "hello", "num_connections": 1}')
    status = watch.status()["slack"]
    assert status["connected"] is True
    assert status["connected_since"] == status["last_message_at"]
    assert (status["drops_10m"], status["errors_10m"], status["last_error"]) == (0, 0, None)

    for _ in range(3):  # a reconnect loop: each new connection breaks at once
        on_error(BrokenPipeError())
        on_close(1006, None)
    status = watch.status()["slack"]
    assert (status["drops_10m"], status["errors_10m"]) == (3, 3)
    assert status["last_error"] == "BrokenPipeError"

    now[0] += runner.SLACK_WINDOW_SECONDS + 1  # old trouble ages out
    client.connected = False
    status = watch.status()["slack"]
    assert (status["connected"], status["drops_10m"], status["errors_10m"]) == (False, 0, 0)


def test_narrow_table_becomes_an_aligned_code_block() -> None:
    table = "| Bot | Files |\n|:---|---:|\n| compliance | 1,843 |\n| growth | 331 |"
    assert formatting.to_mrkdwn(table) == (
        "```\nBot        | Files\n-----------+------\ncompliance | 1,843\ngrowth     | 331\n```"
    )


def test_wide_table_becomes_one_line_per_row() -> None:
    table = (
        "Pricing:\n\n"
        "| Country | Currency | Payout type | Service Type | Projected Volume per month | Fee |\n"
        "|---|---|---|---|---|---|\n"
        "| **India** | INR | B2B | Local Transfer | 100 Mn | 5 |\n"
        "| Japan | JPY | B2B | SWIFT(SHA) | | 15 |\n"
        "\nDone."
    )
    assert formatting.to_mrkdwn(table) == (
        "Pricing:\n\n"
        "• *India* — Currency: INR · Payout type: B2B · Service Type: Local Transfer · "
        "Projected Volume per month: 100 Mn · Fee: 5\n"
        "• *Japan* — Currency: JPY · Payout type: B2B · Service Type: SWIFT(SHA) · Fee: 15\n"
        "\nDone."
    )


def test_code_block_split_across_sections_stays_closed() -> None:
    code = "```\n" + "\n".join(f"line {i} " + "x" * 40 for i in range(120)) + "\n```"
    pieces = formatting.split_sections(code)
    assert len(pieces) > 1
    assert all(piece.count("```") % 2 == 0 for piece in pieces)
    assert all(len(piece) <= formatting.SECTION_LIMIT for piece in pieces)


class FakeSocketHandler:
    def __init__(self) -> None:
        self.client = FakeSocketClient()
        self.client.connected = False
        self.closed = False

    def connect(self) -> None:
        self.client.connected = True

    def close(self) -> None:
        self.closed = True
        self.client.connected = False


class StopServing(Exception):
    """Ends a test's serve loop once the fake clock has run long enough."""


class SocketHarness:
    """``serve_socket`` with fake handlers and a fake clock; each sleep runs ``tick``."""

    def __init__(self, *, settings: SlackSettings | None = None, busy: bool = False) -> None:
        self.now = 1_000.0
        self.busy = busy
        self.handlers = [FakeSocketHandler()]
        self.events: list[dict[str, Any]] = []
        self.connection = runner.SlackConnection(self.handlers[0].client, clock=self.clock)
        self.watchdog = runner.SlackWatchdog(
            self.connection, settings or SlackSettings(), busy=lambda: self.busy, clock=self.clock
        )
        self.tick: Callable[[FakeSocketClient], None] = lambda client: None

    def clock(self) -> float:
        return self.now

    def new_handler(self) -> FakeSocketHandler:
        self.handlers.append(FakeSocketHandler())
        return self.handlers[-1]

    def serve(self, *, hours: float) -> None:
        end = self.now + hours * 3600

        def sleep(seconds: float) -> None:
            if self.now >= end:
                raise StopServing
            self.now += seconds
            self.tick(self.handlers[-1].client)

        runner.serve_socket(
            self.handlers[-1],  # a later serve() picks up where the last one stopped
            self.new_handler,
            self.connection,
            self.watchdog,
            record=self.events.append,
            sleep=sleep,
        )


WS = object()
"""The websocket-client adapter passes its WebSocketApp first to every listener."""


def flap(client: FakeSocketClient) -> None:
    """A reconnect loop: about one broken connection every 10 s (two per 15 s check)."""
    for _ in range(2):
        for on_error in client.on_error_listeners:
            on_error(WS, BrokenPipeError())
        for on_close in client.on_close_listeners:
            on_close(WS)


def fail_silently(client: FakeSocketClient) -> None:
    """After a network blip: every attempt errors, but no close is ever reported."""
    for _ in range(2):
        for on_error in client.on_error_listeners:
            on_error(WS, BrokenPipeError())


def test_flapping_connection_reconnects_then_exits() -> None:
    harness = SocketHarness()
    harness.tick = flap
    with pytest.raises(SystemExit) as exit_info:
        harness.serve(hours=1)
    assert exit_info.value.code == runner.RESTART_EXIT_CODE
    assert [e["action"] for e in harness.events] == ["reconnect", "restart"]
    assert all(e["kind"] == "slack_restart" for e in harness.events)
    assert "BrokenPipeError" in harness.events[0]["reason"]
    assert len(harness.handlers) == 2  # one fresh client before giving up
    assert all(handler.closed for handler in harness.handlers)


def test_normal_rotation_and_brief_drops_do_nothing() -> None:
    harness = SocketHarness()
    ticks = [0]

    def rotate(client: FakeSocketClient) -> None:
        ticks[0] += 1
        if ticks[0] % 120 == 0:  # Slack rotates the connection every 30 min: a drop...
            for on_close in client.on_close_listeners:
                on_close(1000, None)
            client.connected = False
        elif ticks[0] % 120 == 2:  # ...and slack_sdk is back 30 s later
            client.connected = True

    harness.tick = rotate
    with pytest.raises(StopServing):
        harness.serve(hours=6)
    assert harness.events == []
    assert len(harness.handlers) == 1


def test_fresh_client_that_holds_resets_the_watchdog() -> None:
    harness = SocketHarness()
    harness.tick = flap
    with pytest.raises(StopServing):
        harness.serve(hours=0.07)  # ~4 min: long enough to reconnect, not to restart
    assert [e["action"] for e in harness.events] == ["reconnect"]
    harness.tick = lambda client: None  # the fresh client works
    with pytest.raises(StopServing):
        harness.serve(hours=1)
    harness.tick = flap  # trouble much later starts over with a reconnect, not an exit
    with pytest.raises(StopServing):
        harness.serve(hours=0.07)
    assert [e["action"] for e in harness.events] == ["reconnect", "reconnect"]


def test_running_sync_defers_the_exit() -> None:
    harness = SocketHarness(busy=True)
    harness.tick = flap
    with pytest.raises(StopServing):
        harness.serve(hours=2)
    assert [e["action"] for e in harness.events] == ["reconnect"]  # no exit mid-sync
    harness.busy = False
    started = harness.now
    with pytest.raises(SystemExit):
        harness.serve(hours=1)
    assert harness.now - started <= runner.WATCHDOG_CHECK_SECONDS  # right after the sync
    assert [e["action"] for e in harness.events] == ["reconnect", "restart"]


def test_watchdog_can_be_turned_off_or_limited_to_disconnects() -> None:
    assert load_settings(env={"SLACK_WATCHDOG": "false"}).interfaces.slack.watchdog is False
    env = {"SLACK_WATCHDOG_FAILURES": "0", "SLACK_WATCHDOG_GRACE_SECONDS": "60"}
    disconnects_only = load_settings(env=env).interfaces.slack
    assert disconnects_only.watchdog_grace_seconds == 60
    harness = SocketHarness(settings=disconnects_only)
    harness.tick = flap  # still connected at each check, so drops alone don't count
    with pytest.raises(StopServing):
        harness.serve(hours=1)
    assert harness.events == []


def test_errors_without_drops_also_count_as_stuck() -> None:
    harness = SocketHarness()
    harness.tick = fail_silently  # stays "connected" at each check, drops_10m stays 0
    with pytest.raises(SystemExit):
        harness.serve(hours=1)
    assert [e["action"] for e in harness.events] == ["reconnect", "restart"]
    assert "0 drops" in harness.events[0]["reason"]
    assert "BrokenPipeError" in harness.events[0]["reason"]


def test_slack_connection_follows_the_websocket_client() -> None:
    from slack_sdk.socket_mode.websocket_client import SocketModeClient

    client = SocketModeClient(app_token="xapp-test")
    try:
        watch = runner.SlackConnection(client)
        for on_message in client.on_message_listeners:
            on_message(WS, '{"type":"hello"}')
        for on_error in client.on_error_listeners:
            on_error(WS, BrokenPipeError())
        for on_close in client.on_close_listeners:
            on_close(WS)
        status = watch.status()["slack"]
        assert status["connected"] is False  # never connected: no socket
        assert status["connected_since"] is not None
        assert (status["drops_10m"], status["errors_10m"]) == (1, 1)
    finally:
        client.close()

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from typer.testing import CliRunner

import drive_agent.embeddings
import drive_agent.llm.claude
from drive_agent.core.agent import AgentReply
from drive_agent.core.config import ConfigError, Settings, SlackSettings
from drive_agent.drive.auth import DriveAuthError
from drive_agent.indexing.sync import SyncReport
from drive_agent.interfaces.cli import app as cli_app
from drive_agent.interfaces.slack_bot import bot as bot_module
from drive_agent.interfaces.slack_bot import formatting, runner
from drive_agent.interfaces.slack_bot.bot import HELP, SEARCHING, SlackBot, register
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


class FakeClient:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

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

    assert client.posts == [{"channel": "C1", "thread_ts": "100.1", "text": SEARCHING}]
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


def test_direct_messages(tmp_path: Path) -> None:
    agent, client = FakeAgent(), FakeClient()
    bot = make_bot(tmp_path, agent)
    bot.handle_dm(dm(), client)
    bot.handle_dm(dm(channel_type="channel"), client)  # channel messages come as mentions
    bot.handle_dm(dm(subtype="message_changed"), client)
    bot.handle_dm(dm(bot_id="B1"), client)
    assert agent.calls == [("hi?", "slack:D1:5.5")]

    off = make_bot(tmp_path, agent, allow_dms=False)
    off.handle_dm(dm(), client)
    assert client.posts[-1]["text"] == "Direct messages are turned off."


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
        (LLMError("No Claude credentials found."), ":warning: No Claude credentials found."),
        (RuntimeError("boom"), ":warning: Something went wrong; see the bot's logs."),
    ],
)
def test_errors_replace_the_placeholder(tmp_path: Path, error: Exception, expected: str) -> None:
    client = FakeClient()
    make_bot(tmp_path, FakeAgent(error=error)).handle_mention(mention(), client)
    assert client.updates == [{"channel": "C1", "ts": "999.000", "text": expected}]


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


def command(text: str, user: str = "U1") -> dict[str, Any]:
    return {"command": "/drive-agent", "text": text, "user_id": user, "channel_id": "C1"}


@pytest.mark.parametrize("text", ["", "help", "HELP"])
def test_command_help(tmp_path: Path, text: str) -> None:
    respond = Responder()
    make_bot(tmp_path).handle_command(command(text), respond)
    assert respond.calls == [{"text": HELP, "response_type": "ephemeral"}]


def test_command_status(tmp_path: Path) -> None:
    respond = Responder()
    make_bot(tmp_path).handle_command(command("status"), respond)
    assert respond.calls == [{"text": "32 chunks", "response_type": "ephemeral"}]


def test_command_sync_is_admin_only(tmp_path: Path) -> None:
    respond = Responder()
    make_bot(tmp_path, admin_users=["UADMIN"]).handle_command(command("sync"), respond)
    assert "Only admins" in respond.calls[0]["text"]

    respond = Responder()
    make_bot(tmp_path, admin_users=["UADMIN"]).handle_command(command("sync", "UADMIN"), respond)
    assert [c["text"] for c in respond.calls][-1] == "synced"


@pytest.mark.parametrize("text", ["ask who knows Go?", "who knows Go?"])
def test_command_question_is_private_and_stateless(tmp_path: Path, text: str) -> None:
    agent, respond = FakeAgent(), Responder()
    make_bot(tmp_path, agent).handle_command(command(text), respond)
    question, thread_id = agent.calls[0]
    assert question == "who knows Go?"
    assert thread_id is not None
    assert thread_id.startswith("slack-cmd:U1:")
    assert respond.calls[0]["response_type"] == "ephemeral"
    assert respond.calls[0]["blocks"][0]["type"] == "section"


def test_command_usage_and_errors(tmp_path: Path) -> None:
    respond = Responder()
    make_bot(tmp_path).handle_command(command("ask"), respond)
    assert "Usage" in respond.calls[0]["text"]
    respond = Responder()
    make_bot(tmp_path, FakeAgent(error=LLMError("rate limited"))).handle_command(
        command("ask q"), respond
    )
    assert respond.calls[0]["text"] == ":warning: rate limited"


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


def test_bot_sync_reports_summary_or_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings.model_validate({"data_dir": str(tmp_path)})
    monkeypatch.setattr(runner, "sync_now", lambda s, i: SyncReport(added=3, unchanged=29))
    bot = runner.make_bot(settings, FakeAgent(), index := _fake_index())  # type: ignore[arg-type]
    respond = Responder()
    bot._settings = SlackSettings(admin_users=["U1"])
    bot.handle_command(command("sync"), respond)
    assert respond.calls[-1]["text"].startswith(":white_check_mark: Sync done: +3 added")

    def fail(s: Any, i: Any) -> None:
        raise DriveAuthError("key not found")

    monkeypatch.setattr(runner, "sync_now", fail)
    bot.handle_command(command("sync"), respond)
    assert respond.calls[-1]["text"] == ":warning: Sync failed: key not found"
    assert index.closed is False


def _fake_index() -> Any:
    class Index:
        namespace = "qdrant:x/y"
        description = "qdrant x / y"
        closed = False

        def count(self) -> int:
            return 7

    return Index()


def test_status_text(tmp_path: Path) -> None:
    from drive_agent.indexing.state import STATE_DB, SQLiteRecordManager

    settings = Settings.model_validate({"data_dir": str(tmp_path)})
    records = SQLiteRecordManager("qdrant:x/y", tmp_path / STATE_DB)
    records.set_meta("last_sync", {"files": 3, "at": "2026-10-17T09:44:45+00:00"})
    records.close()
    text = runner.status_text(settings, _fake_index())
    assert text.startswith("*Index:* 7 chunks from 3 files · last sync 2026-10-17 09:44 UTC")


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
    monkeypatch.setattr(runner, "run", lambda settings, agent, index: started.append(agent))
    result = CliRunner().invoke(cli_app, ["slack"])
    assert result.exit_code == 0, result.output
    assert "Slack bot running in socket mode" in result.output
    assert "index is empty" in result.output
    assert len(started) == 1

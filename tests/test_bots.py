"""One bot per Drive folder: per-bot env files, `bot add` / `bot list`, and `--bot`."""

import stat
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from drive_agent.core.config import ConfigError, bot_env_file, load_settings
from drive_agent.interfaces import cli
from drive_agent.interfaces.cli import app
from drive_agent.interfaces.slack_bot.bot import SlackBot, help_text
from drive_agent.interfaces.slack_bot.manifest import render_manifest
from drive_agent.vectorstores.location import namespace_for

from .conftest import REPO_ROOT

runner = CliRunner()
URL = "https://drive.google.com/drive/folders/1HrFolderAbc"


@pytest.fixture(autouse=True)
def wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.console, "width", 200)
    monkeypatch.setattr(cli.err_console, "width", 200)


def add(name: str, *folders: str, force: bool = False) -> Any:
    args = ["bot", "add", name]
    for folder in folders or (URL,):
        args += ["--folder", folder]
    return runner.invoke(app, [*args, "--force"] if force else args)


# --- env layering -------------------------------------------------------------


def test_bot_env_layers_over_shared_env(monkeypatch: pytest.MonkeyPatch) -> None:
    Path(".env").write_text(
        "DRIVE_FOLDER_IDS=shared\nSLACK_BOT_TOKEN=xoxb-shared\nVECTOR_STORE=qdrant\n"
    )
    Path("hr.env").write_text("DRIVE_FOLDER_IDS=hrfolder\nSLACK_BOT_TOKEN=\n")
    s = load_settings(extra_env_file=Path("hr.env"))
    assert s.drive.folder_ids == ["hrfolder"]  # bot file beats shared .env
    assert s.secrets.slack_bot_token is None  # an empty value blanks a shared token
    assert s.vector_store.provider == "qdrant"  # everything else is shared
    assert s.extra_env_file == Path("hr.env")

    monkeypatch.setenv("DRIVE_FOLDER_IDS", "fromprocess")
    assert load_settings(extra_env_file=Path("hr.env")).drive.folder_ids == ["fromprocess"]


def test_env_file_from_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    Path("x.env").write_text("DRIVE_FOLDER_IDS=xfolder\n")
    monkeypatch.setenv("DRIVE_AGENT_ENV_FILE", "x.env")  # how `ui` hands it to Streamlit
    assert load_settings().drive.folder_ids == ["xfolder"]


def test_missing_env_file() -> None:
    with pytest.raises(ConfigError, match="env file not found"):
        load_settings(extra_env_file=Path("nope.env"))


@pytest.mark.parametrize("name", ["HR", "-x", "a b", "x" * 31, "hr.env", ""])
def test_bot_names_are_validated(name: str) -> None:
    with pytest.raises(ConfigError, match="invalid bot name"):
        bot_env_file(name)


# --- bot add / list -------------------------------------------------------------


def test_bot_add_creates_private_env_and_manifest() -> None:
    result = add("hr")
    assert result.exit_code == 0, result.output
    env = Path("bots/hr.env")
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    text = env.read_text()
    assert "DRIVE_FOLDER_IDS=1HrFolderAbc\n" in text
    assert "DRIVE_AGENT__VECTOR_STORE__COLLECTION=drive_agent_hr\n" in text
    assert "DRIVE_AGENT__DATA_DIR=.drive-agent/bots/hr\n" in text
    assert "SLACK_COMMAND=/drive-hr\n" in text
    manifest = yaml.safe_load(Path("bots/hr.manifest.yaml").read_text())
    assert manifest["display_information"]["name"] == "Drive Agent (hr)"
    assert manifest["features"]["slash_commands"][0]["command"] == "/drive-hr"
    assert "drive-agent --bot hr sync" in result.output


def test_bot_add_multiple_folders_and_overwrite_rules() -> None:
    assert add("eng", "a1", URL).exit_code == 0
    assert "DRIVE_FOLDER_IDS=a1,1HrFolderAbc" in Path("bots/eng.env").read_text()
    again = add("eng")
    assert again.exit_code == 1
    assert "already exists" in again.output
    assert add("eng", force=True).exit_code == 0


@pytest.mark.parametrize(
    ("name", "folder", "error"),
    [("Bad Name", URL, "invalid bot name"), ("ok", "not a folder!", "not a Google Drive folder")],
)
def test_bot_add_rejects_bad_input(name: str, folder: str, error: str) -> None:
    result = add(name, folder)
    assert result.exit_code == 1
    assert error in result.output


def test_two_bots_are_isolated() -> None:
    add("hr", "hrfolder")
    add("eng", "engfolder")
    hr = load_settings(extra_env_file=bot_env_file("hr"))
    eng = load_settings(extra_env_file=bot_env_file("eng"))
    assert hr.drive.folder_ids != eng.drive.folder_ids
    assert hr.vector_store.collection == "drive_agent_hr"
    assert namespace_for(hr) != namespace_for(eng)
    assert hr.data_dir == Path(".drive-agent/bots/hr")
    assert hr.data_dir != eng.data_dir  # separate sync state, conversations, feedback
    assert hr.interfaces.slack.command != eng.interfaces.slack.command


def test_bot_list() -> None:
    assert "No bots yet" in runner.invoke(app, ["bot", "list"]).output
    add("hr", "hrfolder")
    add("eng", "engfolder")
    env = Path("bots/eng.env")
    env.write_text(
        env.read_text()
        .replace("SLACK_BOT_TOKEN=\n", "SLACK_BOT_TOKEN=xoxb-1\n")
        .replace("SLACK_APP_TOKEN=\n", "SLACK_APP_TOKEN=xapp-1\n")
    )
    out = runner.invoke(app, ["bot", "list"]).output
    eng_row = next(line for line in out.splitlines() if " eng " in line)
    hr_row = next(line for line in out.splitlines() if " hr " in line)
    assert "engfolder" in eng_row
    assert "ready" in eng_row
    assert "/drive-eng" in eng_row
    assert "tokens missing" in hr_row
    assert "not built yet" in hr_row


# --- --bot on commands -----------------------------------------------------------


def test_bot_flag_selects_the_bot() -> None:
    add("hr", "hrfolder")
    result = runner.invoke(app, ["--bot", "hr", "status"])
    assert result.exit_code == 0, result.output
    assert "bot env file" in result.output
    assert "bots/hr.env" in result.output
    assert "hrfolder" in result.output
    assert "drive_agent_hr" in result.output


def test_bot_flag_errors() -> None:
    unknown = runner.invoke(app, ["--bot", "ghost", "status"])
    assert unknown.exit_code == 1
    assert "drive-agent bot add ghost" in unknown.output
    both = runner.invoke(app, ["--bot", "hr", "--env-file", "x.env", "status"])
    assert both.exit_code == 2
    assert runner.invoke(app, ["--bot", "Bad Name", "status"]).exit_code == 2


def test_ui_passes_the_bot_to_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    add("hr", "hrfolder")
    seen: list[dict[str, str]] = []
    monkeypatch.setattr(subprocess, "call", lambda cmd, env: seen.append(env) or 0)
    assert runner.invoke(app, ["--bot", "hr", "ui"]).exit_code == 0
    assert seen[0]["DRIVE_AGENT_ENV_FILE"] == str(Path("bots/hr.env").resolve())


# --- per-bot Slack command and manifest ---------------------------------------------


def test_slash_command_is_per_bot(tmp_path: Path) -> None:
    from drive_agent.core.config import SlackSettings

    bot = SlackBot(
        object(),  # type: ignore[arg-type]
        SlackSettings(command="/drive-hr"),
        status=lambda: "",
        sync=lambda: "",
        feedback_path=tmp_path / "f.jsonl",
    )
    assert bot.command == "/drive-hr"
    calls: list[dict[str, Any]] = []
    bot.handle_command({"text": "help"}, lambda **kw: calls.append(kw), None)
    assert "`/drive-hr status`" in calls[0]["text"]
    bot.handle_command({"text": "ask"}, lambda **kw: calls.append(kw), None)
    assert calls[-1]["text"] == "Usage: `/drive-hr ask <question>`"

    from urllib.parse import urlencode

    from slack_bolt import App
    from slack_bolt.authorization import AuthorizeResult
    from slack_bolt.request import BoltRequest

    from drive_agent.interfaces.slack_bot.bot import register

    bolt = App(
        signing_secret="s",
        request_verification_enabled=False,
        process_before_response=True,  # run listeners synchronously for the test
        authorize=lambda **kw: AuthorizeResult(
            enterprise_id=None, team_id="T1", bot_token="xoxb-test", bot_user_id="UB", bot_id="B1"
        ),
    )
    seen: list[str] = []
    bot.handle_command = lambda command, respond, client: seen.append(command["command"])  # type: ignore[method-assign]
    register(bolt, bot)
    for name in ("/drive-hr", "/drive-agent"):
        body = urlencode(
            {"command": name, "text": "help", "user_id": "U1", "team_id": "T1",
             "channel_id": "C1", "response_url": "https://hooks.slack.com/x", "trigger_id": "t"}
        )  # fmt: skip
        bolt.dispatch(
            BoltRequest(body=body, headers={"content-type": ["application/x-www-form-urlencoded"]})
        )
    assert seen == ["/drive-hr"]  # another bot's command is not answered


@pytest.mark.parametrize("command", ["drive", "/Drive", "/a b", "/" + "x" * 32])
def test_invalid_slash_commands_rejected(command: str) -> None:
    with pytest.raises(ConfigError, match="command"):
        load_settings(env={"SLACK_COMMAND": command})


def test_repo_manifest_is_the_rendered_default() -> None:
    assert (REPO_ROOT / "slack" / "manifest.yaml").read_text() == render_manifest()
    assert "`/drive-agent status`" in help_text()

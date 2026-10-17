"""Streamlit pages driven headlessly with AppTest against embedded Qdrant + fakes."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

import drive_agent.embeddings
import drive_agent.llm.claude
from drive_agent.core.config import load_settings
from drive_agent.drive import auth
from drive_agent.drive.fetcher import DriveFetcher
from drive_agent.interfaces.cli import app as cli_app
from drive_agent.interfaces.streamlit_app import chat, resources
from drive_agent.llm import LLMError
from drive_agent.vectorstores import build_vector_index

from .conftest import SA_EMAIL
from .doc_factory import make_pdf
from .fake_drive import FakeDrive, fake_download, file, folder
from .fakes import FakeLLM

APP = str(Path(resources.__file__).with_name("app.py"))


def chat_page() -> None:
    from drive_agent.interfaces.streamlit_app.chat import render_chat

    render_chat()


def index_page() -> None:
    from drive_agent.interfaces.streamlit_app.index_page import render_index

    render_index()


def setup_page() -> None:
    from drive_agent.interfaces.streamlit_app.setup_page import render_setup

    render_setup()


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[FakeLLM]]:
    """Fake embeddings and Claude; fresh shared resources for every test."""
    monkeypatch.setattr(
        drive_agent.embeddings, "build_embeddings", lambda s: DeterministicFakeEmbedding(size=8)
    )
    llms: list[FakeLLM] = []

    def make_llm(settings: object) -> FakeLLM:
        llms.append(llm := FakeLLM(answer="Jane knows Go [1], earning $120k."))
        return llm

    monkeypatch.setattr(drive_agent.llm.claude, "ClaudeLLM", make_llm)
    resources._load.clear()
    yield llms
    resources.reset_resources()


def seed_index() -> None:
    settings = load_settings()
    index = build_vector_index(settings, DeterministicFakeEmbedding(size=8))
    index.store.add_documents(
        [
            Document(
                "Jane.pdf > page 1\n\nJane Doe knows Go.",
                metadata={
                    "file_id": "cv",
                    "file_name": "Jane.pdf",
                    "path": "Hiring/Jane.pdf",
                    "web_url": "https://drive.google.com/file/d/cv",
                    "page": 1,
                },
            )
        ]
    )
    index.close()


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += (
        [e.value for e in at.error] + [i.value for i in at.info] + [s.value for s in at.success]
    )
    return "\n".join(str(p) for p in parts)


# --- chat -------------------------------------------------------------------


def test_chat_with_empty_index() -> None:
    at = AppTest.from_function(chat_page).run(timeout=30)
    assert "index is empty" in texts(at)


def test_chat_streams_answer_with_citations_and_passages() -> None:
    seed_index()
    at = AppTest.from_function(chat_page).run(timeout=30)
    at.chat_input[0].set_value("Who knows Go?").run(timeout=30)

    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    out = texts(at)
    assert "Jane knows Go [1], earning \\$120k." in out  # $ escaped: no LaTeX
    assert "**[1] Jane.pdf** · page 1" in out
    assert "Hiring/Jane.pdf" in out
    assert at.expander[0].label == "Retrieved passages (1)"


def test_chat_remembers_conversation_within_session(fakes: list[FakeLLM]) -> None:
    seed_index()
    at = AppTest.from_function(chat_page).run(timeout=30)
    at.chat_input[0].set_value("Who knows Go?").run(timeout=30)
    at.chat_input[0].set_value("And Kubernetes?").run(timeout=30)
    assert len(at.chat_message) == 4
    roles = [m["role"] for m in fakes[-1].answer_calls[-1][1]]
    assert roles == ["user", "assistant", "user"]

    at.sidebar.button[0].click().run(timeout=30)  # New conversation
    assert len(at.chat_message) == 0


def test_chat_shows_claude_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    seed_index()

    class Broken(FakeLLM):
        def structured(self, *args: Any, **kwargs: Any) -> Any:
            raise LLMError("No Claude credentials found.")

    monkeypatch.setattr(drive_agent.llm.claude, "ClaudeLLM", lambda s: Broken())
    at = AppTest.from_function(chat_page).run(timeout=30)
    at.chat_input[0].set_value("Who knows Go?").run(timeout=30)
    assert "No Claude credentials found." in texts(at)
    assert at.chat_message[1].name == "assistant"  # the error shows in the reply bubble
    at.run(timeout=30)
    assert len(at.chat_message) == 0  # the failed exchange is not kept; a retry starts clean


def test_markdown_escaping() -> None:
    assert chat._md("$5 and $10") == "\\$5 and \\$10"


# --- index ------------------------------------------------------------------


def _fake_drive(monkeypatch: pytest.MonkeyPatch, sa_key_file: Path) -> None:
    monkeypatch.setattr(DriveFetcher, "_download", fake_download)
    drive = FakeDrive(
        [folder("root", "Hiring"), file("cv", "Jane.pdf", "application/pdf", "root", size=10)],
        contents={"cv": make_pdf(["Jane Doe knows Go"])},
        user_email=SA_EMAIL,
    )
    monkeypatch.setattr(auth, "build_drive_service", lambda creds: drive)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa_key_file))
    monkeypatch.setenv("DRIVE_FOLDER_IDS", "root")


def test_index_page_syncs_with_progress(monkeypatch: pytest.MonkeyPatch, sa_key_file: Path) -> None:
    _fake_drive(monkeypatch, sa_key_file)
    at = AppTest.from_function(index_page).run(timeout=30)
    assert at.metric[0].value == "0"
    at.button[0].click().run(timeout=60)  # Sync now

    assert not at.exception
    assert "+1 added · 0 updated · 0 unchanged · 0 deleted" in texts(at)
    table = at.dataframe[0].value
    assert list(table["file"]) == ["Hiring/Jane.pdf"]
    assert at.metric[0].value == "0"  # metrics were drawn before the sync ran

    at.run(timeout=30)
    assert at.metric[0].value == "1"
    assert at.metric[1].value == "1"


def test_full_rebuild_needs_confirmation() -> None:
    at = AppTest.from_function(index_page).run(timeout=30)
    assert at.button[1].disabled is True
    at.checkbox[0].check().run(timeout=30)
    assert at.button[1].disabled is False


def test_index_page_without_folders() -> None:
    at = AppTest.from_function(index_page).run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert "No Drive folders configured" in texts(at)


# --- setup ------------------------------------------------------------------


def test_setup_page_check_and_env_managed_folders(
    monkeypatch: pytest.MonkeyPatch, sa_key_file: Path
) -> None:
    _fake_drive(monkeypatch, sa_key_file)
    at = AppTest.from_function(setup_page).run(timeout=30)
    table = at.table[0].value
    assert "qdrant" in " ".join(str(v) for v in table["value"])
    assert at.text_area[0].disabled is True
    assert "DRIVE_FOLDER_IDS" in texts(at)

    at.button[0].click().run(timeout=30)  # Check Drive access
    out = texts(at)
    assert f"Signed in as {SA_EMAIL}" in out
    assert "**Hiring** · My Drive" in out


def test_setup_page_saves_folders_to_yaml() -> None:
    Path("drive-agent.yaml").write_text(yaml.safe_dump({"agent": {"mode": "rag"}}))
    at = AppTest.from_function(setup_page).run(timeout=30)
    at.text_area[0].set_value("https://drive.google.com/drive/folders/1AbCdEf\nsecondId").run(
        timeout=30
    )
    at.button[0].click().run(timeout=30)  # Save folders

    assert "Saved 2 folder(s)" in texts(at)
    saved = yaml.safe_load(Path("drive-agent.yaml").read_text())
    assert saved == {"agent": {"mode": "rag"}, "drive": {"folder_ids": ["1AbCdEf", "secondId"]}}


def test_setup_page_rejects_bad_folder() -> None:
    at = AppTest.from_function(setup_page).run(timeout=30)
    at.text_area[0].set_value("not a folder!").run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert "not a Google Drive folder" in texts(at)
    assert not Path("drive-agent.yaml").exists()


# --- app shell --------------------------------------------------------------


def test_password_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UI_PASSWORD", "s3cret")
    at = AppTest.from_file(APP).run(timeout=30)
    assert at.text_input[0].label == "Password"
    at.text_input[0].set_value("wrong").run(timeout=30)
    assert "Wrong password." in texts(at)
    at.text_input[0].set_value("s3cret").run(timeout=30)
    assert not at.text_input  # gate passed: the chat page renders
    assert "Ask your Drive" in [t.value for t in at.title]


def test_app_without_password_opens_chat() -> None:
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception
    assert "Ask your Drive" in [t.value for t in at.title]


def test_cli_ui_launches_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    calls: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(subprocess, "call", lambda cmd, env: calls.append((cmd, env)) or 0)
    result = CliRunner().invoke(cli_app, ["ui", "--port", "9000"])
    assert result.exit_code == 0, result.output
    cmd, _env = calls[0]
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert cmd[4] == APP
    assert cmd[cmd.index("--server.port") + 1] == "9000"
    assert cmd[cmd.index("--server.address") + 1] == "localhost"
    assert cmd[cmd.index("--browser.gatherUsageStats") + 1] == "false"
    assert cmd[cmd.index("--server.headless") + 1] == "false"


def test_cli_ui_warns_when_exposed_without_password(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    monkeypatch.setattr(subprocess, "call", lambda cmd, env: 0)
    result = CliRunner().invoke(cli_app, ["ui", "--host", "0.0.0.0"])  # noqa: S104
    assert "without UI_PASSWORD" in result.output


def test_short_time() -> None:
    from drive_agent.interfaces.streamlit_app.index_page import _short_time

    assert _short_time("2026-10-17T09:44:45+00:00") == "Oct 17, 09:44"
    assert _short_time(None) == "never"
    assert _short_time("garbage") == "garbage"

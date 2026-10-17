from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from drive_agent import __version__
from drive_agent.core.config import load_settings
from drive_agent.drive import auth
from drive_agent.interfaces import cli
from drive_agent.interfaces.cli import app

from .conftest import SA_EMAIL
from .fake_drive import FakeDrive, file, folder, http_error

FOLDER = "1AbCdEfGhIjKlMnOpQrStUvWxYz_-0123"
runner = CliRunner()


@pytest.fixture(autouse=True)
def wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop Rich from wrapping long lines at 80 columns, so assertions see whole lines."""
    monkeypatch.setattr(cli.console, "width", 200)
    monkeypatch.setattr(cli.err_console, "width", 200)


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_init_non_interactive_writes_config_and_env() -> None:
    url = f"https://drive.google.com/drive/folders/{FOLDER}"
    result = runner.invoke(app, ["init", "--folder", url, "--auth-mode", "oauth"])
    assert result.exit_code == 0, result.output

    data = yaml.safe_load(Path("drive-agent.yaml").read_text())
    assert data["drive"]["folder_ids"] == [FOLDER]
    assert data["drive"]["auth_mode"] == "oauth"
    assert "secrets" not in data
    assert "ANTHROPIC_API_KEY=" in Path(".env").read_text()

    settings = load_settings(env={})
    assert settings.drive.folder_ids == [FOLDER]


def test_init_interactive_prompts() -> None:
    result = runner.invoke(app, ["init"], input=f"service_account\n{FOLDER}\n")
    assert result.exit_code == 0, result.output
    assert load_settings(env={}).drive.folder_ids == [FOLDER]


def test_init_keeps_existing_env_file() -> None:
    Path(".env").write_text("ANTHROPIC_API_KEY=keep-me\n")
    result = runner.invoke(app, ["init", "--folder", FOLDER, "--auth-mode", "oauth"])
    assert result.exit_code == 0, result.output
    assert Path(".env").read_text() == "ANTHROPIC_API_KEY=keep-me\n"


def test_init_refuses_to_overwrite_without_force() -> None:
    Path("drive-agent.yaml").write_text("agent: {mode: agentic}\n")
    result = runner.invoke(app, ["init", "--folder", FOLDER, "--auth-mode", "oauth"])
    assert result.exit_code == 1
    assert "already exists" in result.output

    result = runner.invoke(app, ["init", "--folder", FOLDER, "--auth-mode", "oauth", "--force"])
    assert result.exit_code == 0, result.output
    assert load_settings(env={}).agent.mode == "rag"


@pytest.mark.parametrize(
    "args",
    [
        ["--folder", "not a folder!", "--auth-mode", "oauth"],
        ["--folder", FOLDER, "--auth-mode", "password"],
    ],
)
def test_init_rejects_bad_input(args: list[str]) -> None:
    result = runner.invoke(app, ["init", *args])
    assert result.exit_code == 1
    assert "Invalid input" in result.output
    assert not Path("drive-agent.yaml").exists()


def test_status_shows_config_and_secret_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("DRIVE_FOLDER_IDS", FOLDER)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert FOLDER in result.output
    assert "qdrant" in result.output
    assert "ANTHROPIC_API_KEY" in result.output
    assert "sk-ant-secret" not in result.output


def test_status_with_explicit_config(tmp_path: Path) -> None:
    cfg = tmp_path / "other.yaml"
    cfg.write_text("vector_store: {provider: pgvector}\n")
    result = runner.invoke(app, ["--config", str(cfg), "status"])
    assert result.exit_code == 0, result.output
    assert "pgvector" in result.output


def test_bad_config_exits_with_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "not-a-db")
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "Config error" in result.output


@pytest.mark.parametrize(
    ("args", "milestone"),
    [
        (["ui"], "M5"),
        (["slack"], "M6"),
    ],
)
def test_unimplemented_commands(args: list[str], milestone: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "not implemented yet" in result.output
    assert milestone in result.output


# --- check ------------------------------------------------------------------

DOC = "application/vnd.google-apps.document"


def _use_fake_drive(monkeypatch: pytest.MonkeyPatch, sa_key_file: Path, drive: FakeDrive) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa_key_file))
    monkeypatch.setattr(auth, "build_drive_service", lambda creds: drive)


def test_check_success(monkeypatch: pytest.MonkeyPatch, sa_key_file: Path) -> None:
    drive = FakeDrive(
        [
            folder("root", "Company", driveId="sd"),
            file("d", "Handbook", DOC, "root"),
            file("s", "Budget", "application/vnd.google-apps.spreadsheet", "root"),
            folder("sub", "HR", "root"),
        ],
        user_email=SA_EMAIL,
    )
    _use_fake_drive(monkeypatch, sa_key_file, drive)
    monkeypatch.setenv("DRIVE_FOLDER_IDS", "root")

    result = runner.invoke(app, ["check", "--show-skipped"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert f"✓ Auth  service account {SA_EMAIL}\n" in out
    assert "drive.readonly (read-only)" in out
    assert "Company (root) · Shared Drive" in out
    assert "files to index: 1 (docs 1)" in out
    assert "sub-folders: 1" in out
    assert "1 out of scope" in out
    assert "Company/Budget" in out  # --show-skipped table


def test_check_folder_not_shared(monkeypatch: pytest.MonkeyPatch, sa_key_file: Path) -> None:
    _use_fake_drive(monkeypatch, sa_key_file, FakeDrive([], user_email=SA_EMAIL))
    monkeypatch.setenv("DRIVE_FOLDER_IDS", "missing")
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 1
    assert "✗ Folder missing" in result.output
    assert "not shared with" in result.output


def test_check_api_disabled(monkeypatch: pytest.MonkeyPatch, sa_key_file: Path) -> None:
    drive = FakeDrive([], about_error=http_error(403, "accessNotConfigured"))
    _use_fake_drive(monkeypatch, sa_key_file, drive)
    monkeypatch.setenv("DRIVE_FOLDER_IDS", "root")
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 1
    assert "✗ Auth" in result.output
    assert "not enabled" in result.output


def test_check_without_folders() -> None:
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 1
    assert "No Drive folders configured" in result.output


# --- sync --dry-run ---------------------------------------------------------


def test_sync_dry_run(monkeypatch: pytest.MonkeyPatch, sa_key_file: Path) -> None:
    from drive_agent.drive.fetcher import DriveFetcher

    from .doc_factory import make_pdf
    from .fake_drive import fake_download

    monkeypatch.setattr(DriveFetcher, "_download", fake_download)
    drive = FakeDrive(
        [
            folder("root", "Hiring"),
            file("cv", "Jane.pdf", "application/pdf", "root", size=10),
            file("bad", "Broken.pdf", "application/pdf", "root", size=10),
            file("s", "Budget", "application/vnd.google-apps.spreadsheet", "root"),
        ],
        contents={"cv": make_pdf(["Jane Doe, engineer"]), "bad": b"junk"},
    )
    _use_fake_drive(monkeypatch, sa_key_file, drive)
    monkeypatch.setenv("DRIVE_FOLDER_IDS", "root")

    result = runner.invoke(app, ["sync", "--dry-run", "--show-chunks", "1"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Nothing is written" in out
    assert "Hiring/Jane.pdf" in out
    assert "unreadable PDF" in out
    assert "Jane.pdf > page 1" in out  # --show-chunks prints chunk text
    assert "Jane Doe, engineer" in out
    assert "Total: 2 files · 1 chunks" in out
    assert "1 failed · 1 skipped" in out


def test_sync_dry_run_missing_folder(monkeypatch: pytest.MonkeyPatch, sa_key_file: Path) -> None:
    _use_fake_drive(monkeypatch, sa_key_file, FakeDrive([]))
    monkeypatch.setenv("DRIVE_FOLDER_IDS", "nope")
    result = runner.invoke(app, ["sync", "--dry-run"])
    assert result.exit_code == 1
    assert "✗ Folder nope" in result.output


def test_sync_dry_run_auth_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "missing.json"))
    monkeypatch.setenv("DRIVE_FOLDER_IDS", "root")
    result = runner.invoke(app, ["sync", "--dry-run"])
    assert result.exit_code == 1
    assert "Auth error" in result.output


def test_sync_dry_run_without_folders() -> None:
    result = runner.invoke(app, ["sync", "--dry-run"])
    assert result.exit_code == 1
    assert "No Drive folders configured" in result.output


# --- sync (real) ------------------------------------------------------------


def test_sync_then_resync_then_status(monkeypatch: pytest.MonkeyPatch, sa_key_file: Path) -> None:
    from langchain_core.embeddings import DeterministicFakeEmbedding

    import drive_agent.embeddings
    from drive_agent.drive.fetcher import DriveFetcher

    from .doc_factory import make_pdf
    from .fake_drive import fake_download

    monkeypatch.setattr(DriveFetcher, "_download", fake_download)
    monkeypatch.setattr(
        drive_agent.embeddings, "build_embeddings", lambda s: DeterministicFakeEmbedding(size=8)
    )
    drive = FakeDrive(
        [folder("root", "Hiring"), file("cv", "Jane.pdf", "application/pdf", "root", size=10)],
        contents={"cv": make_pdf(["Jane Doe, engineer", "Python"])},
    )
    _use_fake_drive(monkeypatch, sa_key_file, drive)
    monkeypatch.setenv("DRIVE_FOLDER_IDS", "root")  # embedded Qdrant under ./.drive-agent

    first = runner.invoke(app, ["sync"])
    assert first.exit_code == 0, first.output
    assert "Syncing into qdrant .drive-agent/qdrant / drive_agent" in first.output
    assert "+2 added · 0 updated · 0 unchanged · 0 deleted" in first.output

    second = runner.invoke(app, ["sync"])
    assert second.exit_code == 0, second.output
    assert "+0 added · 0 updated · 2 unchanged · 0 deleted" in second.output

    status = runner.invoke(app, ["status"])
    assert "2 chunks from 1 files · last sync" in status.output


def test_sync_setup_errors(monkeypatch: pytest.MonkeyPatch, sa_key_file: Path) -> None:
    _use_fake_drive(monkeypatch, sa_key_file, FakeDrive([folder("root", "R")]))
    monkeypatch.setenv("DRIVE_FOLDER_IDS", "root")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "voyage")  # no VOYAGE_API_KEY set
    result = runner.invoke(app, ["sync"])
    assert result.exit_code == 1
    assert "Setup error" in result.output
    assert "VOYAGE_API_KEY" in result.output


def test_status_before_any_sync() -> None:
    result = runner.invoke(app, ["status"])
    assert "not built yet" in result.output


# --- ask --------------------------------------------------------------------


@pytest.fixture
def synced(monkeypatch: pytest.MonkeyPatch, sa_key_file: Path) -> list[object]:
    """Sync one fake CV into embedded Qdrant and swap Claude for a scripted fake."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    import drive_agent.embeddings
    import drive_agent.llm.claude
    from drive_agent.drive.fetcher import DriveFetcher

    from .doc_factory import make_pdf
    from .fake_drive import fake_download
    from .fakes import FakeLLM

    monkeypatch.setattr(DriveFetcher, "_download", fake_download)
    monkeypatch.setattr(
        drive_agent.embeddings, "build_embeddings", lambda s: DeterministicFakeEmbedding(size=8)
    )
    drive = FakeDrive(
        [folder("root", "Hiring"), file("cv", "Jane.pdf", "application/pdf", "root", size=10)],
        contents={"cv": make_pdf(["Jane Doe knows Go"])},
    )
    _use_fake_drive(monkeypatch, sa_key_file, drive)
    monkeypatch.setenv("DRIVE_FOLDER_IDS", "root")
    assert runner.invoke(app, ["sync"]).exit_code == 0

    llms: list[object] = []

    def make_llm(settings: object) -> FakeLLM:
        llms.append(fake := FakeLLM(answer="Jane Doe knows Go [1]."))
        return fake

    monkeypatch.setattr(drive_agent.llm.claude, "ClaudeLLM", make_llm)
    return llms


def test_ask_one_off_prints_answer_and_sources(synced: list[object]) -> None:
    result = runner.invoke(app, ["ask", "Who knows Go?"])
    assert result.exit_code == 0, result.output
    assert "Agent: Jane Doe knows Go [1]." in result.output
    assert "Sources" in result.output
    assert "[1] Jane.pdf, page 1" in result.output
    assert "continue with --thread" in result.output


def test_ask_thread_continues_conversation(synced: list[object]) -> None:
    from .fakes import FakeLLM

    first = runner.invoke(app, ["ask", "Who knows Go?", "--thread", "hiring"])
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, ["ask", "And Kubernetes?", "--thread", "hiring"])
    assert second.exit_code == 0, second.output
    llm = synced[-1]
    assert isinstance(llm, FakeLLM)
    roles = [m["role"] for m in llm.answer_calls[-1][1]]
    assert roles == ["user", "assistant", "user"]  # history persisted in conversations.db
    assert Path(".drive-agent/conversations.db").is_file()


def test_ask_interactive_chat(synced: list[object]) -> None:
    result = runner.invoke(app, ["ask"], input="Who knows Go?\nexit\n")
    assert result.exit_code == 0, result.output
    assert "Chatting about your Drive" in result.output
    assert result.output.count("Agent:") == 1


def test_ask_with_empty_index(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.embeddings import DeterministicFakeEmbedding

    import drive_agent.embeddings

    monkeypatch.setattr(
        drive_agent.embeddings, "build_embeddings", lambda s: DeterministicFakeEmbedding(size=8)
    )
    result = runner.invoke(app, ["ask", "anything"])
    assert result.exit_code == 1
    assert "index is empty" in result.output


def test_ask_reports_claude_errors(synced: list[object], monkeypatch: pytest.MonkeyPatch) -> None:
    import drive_agent.llm.claude
    from drive_agent.llm.claude import MISSING_CREDENTIALS, LLMError

    class NoCreds:
        def __init__(self, settings: object) -> None: ...

        def structured(self, *args: object) -> None:
            raise LLMError(MISSING_CREDENTIALS)

    monkeypatch.setattr(drive_agent.llm.claude, "ClaudeLLM", NoCreds)
    result = runner.invoke(app, ["ask", "Who knows Go?"])
    assert result.exit_code == 1
    assert "No Claude credentials found" in result.output

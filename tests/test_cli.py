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
    Path("drive-agent.yaml").write_text("agent: {mode: rag}\n")
    result = runner.invoke(app, ["init", "--folder", FOLDER, "--auth-mode", "oauth"])
    assert result.exit_code == 1
    assert "already exists" in result.output

    result = runner.invoke(app, ["init", "--folder", FOLDER, "--auth-mode", "oauth", "--force"])
    assert result.exit_code == 0, result.output
    assert load_settings(env={}).agent.mode == "auto"


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
    assert "chroma" in result.output
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
        (["sync"], "M3"),
        (["sync", "--full"], "M3"),
        (["ask", "What is our leave policy?"], "M4"),
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

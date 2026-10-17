import json
from pathlib import Path

import pytest
import rsa

from drive_agent.core.config import CONFIG_ENV_VAR, ENV_ALIASES, NESTED_ENV_PREFIX, Secrets

REPO_ROOT = Path(__file__).resolve().parent.parent
SA_EMAIL = "drive-agent@test-project.iam.gserviceaccount.com"


@pytest.fixture(scope="session")
def _private_key_pem() -> str:
    _, private = rsa.newkeys(1024)  # throwaway key, test-only
    return private.save_pkcs1().decode()


@pytest.fixture
def sa_key_file(tmp_path: Path, _private_key_pem: str) -> Path:
    """A syntactically valid service-account key that points at no real project."""
    path = tmp_path / "sa.json"
    path.write_text(
        json.dumps(
            {
                "type": "service_account",
                "project_id": "test-project",
                "private_key_id": "0" * 40,
                "private_key": _private_key_pem,
                "client_email": SA_EMAIL,
                "client_id": "1234567890",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )
    )
    return path


@pytest.fixture
def oauth_client_file(tmp_path: Path) -> Path:
    path = tmp_path / "client_secret.json"
    path.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "test.apps.googleusercontent.com",
                    "client_secret": "test-secret",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            }
        )
    )
    return path


@pytest.fixture(autouse=True)
def offline_bm25(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never download the BM25 model in tests; use a deterministic bag-of-words encoder."""
    from drive_agent.vectorstores import qdrant

    from .fakes import FakeSparse

    monkeypatch.setattr(qdrant, "_bm25", lambda settings: FakeSparse())


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run every test in an empty directory with no drive-agent env vars set."""
    import os

    managed = {CONFIG_ENV_VAR, *ENV_ALIASES, *(name.upper() for name in Secrets.model_fields)}
    for name in list(os.environ):
        if name in managed or name.startswith(NESTED_ENV_PREFIX):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    return tmp_path

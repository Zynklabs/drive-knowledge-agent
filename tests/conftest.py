from pathlib import Path

import pytest

from drive_agent.core.config import CONFIG_ENV_VAR, ENV_ALIASES, NESTED_ENV_PREFIX, Secrets

REPO_ROOT = Path(__file__).resolve().parent.parent


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

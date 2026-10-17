import json
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.oauth2.credentials import Credentials as UserCredentials

from drive_agent.core.config import DriveSettings
from drive_agent.drive import auth
from drive_agent.drive.auth import READONLY_SCOPE, DriveAuthError, authenticate

from .conftest import SA_EMAIL

FULL_SCOPE = "https://www.googleapis.com/auth/drive"


def sa(path: Path | None, **kw: Any) -> DriveSettings:
    return DriveSettings(auth_mode="service_account", credentials_file=path, **kw)


def oauth(client: Path | None = None) -> DriveSettings:
    return DriveSettings(auth_mode="oauth", oauth_client_file=client)


def write_token(path: Path, scopes: list[str], *, expired: bool = False) -> Path:
    expiry = datetime.now(UTC) + timedelta(hours=-1 if expired else 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "token": "access",
                "refresh_token": "refresh",
                "client_id": "cid",
                "client_secret": "csecret",
                "token_uri": "https://oauth2.googleapis.com/token",
                "scopes": scopes,
                "expiry": expiry.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    )
    return path


# --- service account ----------------------------------------------------------


def test_service_account_key_is_read_only(sa_key_file: Path, tmp_path: Path) -> None:
    result = authenticate(sa(sa_key_file), tmp_path)
    assert result.mode == "service_account"
    assert result.email == SA_EMAIL
    assert list(result.credentials.scopes) == [READONLY_SCOPE]  # type: ignore[attr-defined]
    assert SA_EMAIL in result.description


def test_domain_wide_delegation_impersonates(sa_key_file: Path, tmp_path: Path) -> None:
    result = authenticate(sa(sa_key_file, impersonate_user="bot@corp.com"), tmp_path)
    assert result.email == "bot@corp.com"
    assert result.credentials._subject == "bot@corp.com"  # type: ignore[attr-defined]
    assert list(result.credentials.scopes) == [READONLY_SCOPE]  # type: ignore[attr-defined]
    assert "domain-wide delegation" in result.description


def test_missing_key_file(tmp_path: Path) -> None:
    with pytest.raises(DriveAuthError, match="not found"):
        authenticate(sa(tmp_path / "missing.json"), tmp_path)


def test_oauth_client_file_given_as_key(oauth_client_file: Path, tmp_path: Path) -> None:
    with pytest.raises(DriveAuthError, match="OAuth client file, not a service account"):
        authenticate(sa(oauth_client_file), tmp_path)


@pytest.mark.parametrize("content", ["not json", "[]", '{"type": "authorized_user"}'])
def test_bad_key_files(tmp_path: Path, content: str) -> None:
    key = tmp_path / "bad.json"
    key.write_text(content)
    with pytest.raises(DriveAuthError):
        authenticate(sa(key), tmp_path)


def test_corrupt_private_key(tmp_path: Path) -> None:
    key = tmp_path / "bad.json"
    key.write_text(
        json.dumps({"type": "service_account", "client_email": "x", "private_key": "nope"})
    )
    with pytest.raises(DriveAuthError, match="invalid service account key"):
        authenticate(sa(key), tmp_path)


class _FakeAdc:
    service_account_email = "runtime@proj.iam.gserviceaccount.com"


def test_application_default_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[Any] = []

    def fake_default(scopes: list[str]) -> tuple[Any, str]:
        calls.append(scopes)
        return _FakeAdc(), "proj"

    monkeypatch.setattr(auth.google.auth, "default", fake_default)
    result = authenticate(sa(None), tmp_path)
    assert calls == [[READONLY_SCOPE]]
    assert result.email == "runtime@proj.iam.gserviceaccount.com"
    assert result.description == "application default credentials"


def test_adc_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(scopes: list[str]) -> None:
        raise DefaultCredentialsError("none")

    monkeypatch.setattr(auth.google.auth, "default", fail)
    with pytest.raises(DriveAuthError, match="GOOGLE_APPLICATION_CREDENTIALS"):
        authenticate(sa(None), tmp_path)


def test_adc_cannot_impersonate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(auth.google.auth, "default", lambda scopes: (_FakeAdc(), "proj"))
    with pytest.raises(DriveAuthError, match="needs a service account key"):
        authenticate(sa(None, impersonate_user="bot@corp.com"), tmp_path)


# --- OAuth --------------------------------------------------------------------


def test_cached_readonly_token_is_used(tmp_path: Path) -> None:
    write_token(tmp_path / "token.json", [READONLY_SCOPE])
    result = authenticate(oauth(), tmp_path, interactive=False)
    assert result.mode == "oauth"
    assert result.credentials.valid


def test_cached_token_with_broader_scope_rejected(tmp_path: Path) -> None:
    write_token(tmp_path / "token.json", [FULL_SCOPE])
    with pytest.raises(DriveAuthError, match=r"expected only the drive\.readonly scope"):
        authenticate(oauth(), tmp_path, interactive=False)


def test_corrupt_token(tmp_path: Path) -> None:
    (tmp_path / "token.json").write_text("{}")
    with pytest.raises(DriveAuthError, match="corrupt"):
        authenticate(oauth(), tmp_path, interactive=False)


def test_no_token_non_interactive(tmp_path: Path) -> None:
    with pytest.raises(DriveAuthError, match="No valid Google sign-in"):
        authenticate(oauth(), tmp_path, interactive=False)


def test_expired_token_is_refreshed_and_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_token(tmp_path / "token.json", [READONLY_SCOPE], expired=True)

    def fake_refresh(self: UserCredentials, request: Any) -> None:
        self.token = "new-access"
        self.expiry = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1)

    monkeypatch.setattr(UserCredentials, "refresh", fake_refresh)
    result = authenticate(oauth(), tmp_path, interactive=False)
    assert result.credentials.token == "new-access"
    assert json.loads((tmp_path / "token.json").read_text())["token"] == "new-access"


def test_revoked_refresh_token_requires_sign_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_token(tmp_path / "token.json", [READONLY_SCOPE], expired=True)

    def fail(self: UserCredentials, request: Any) -> None:
        raise RefreshError("invalid_grant")

    monkeypatch.setattr(UserCredentials, "refresh", fail)
    with pytest.raises(DriveAuthError, match="No valid Google sign-in"):
        authenticate(oauth(), tmp_path, interactive=False)


class _FakeFlow:
    def __init__(self, creds: UserCredentials) -> None:
        self.creds = creds

    def run_local_server(self, port: int, open_browser: bool) -> UserCredentials:
        return self.creds


def _install_flow(monkeypatch: pytest.MonkeyPatch, scopes: list[str]) -> list[Any]:
    seen: list[Any] = []
    creds = UserCredentials(
        token="t",
        refresh_token="r",
        client_id="cid",
        client_secret="cs",
        token_uri="https://oauth2.googleapis.com/token",
        scopes=scopes,
        expiry=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1),
    )

    def from_client_config(info: dict[str, Any], scopes: list[str]) -> _FakeFlow:
        seen.append(scopes)
        return _FakeFlow(creds)

    monkeypatch.setattr(auth.InstalledAppFlow, "from_client_config", from_client_config)
    return seen


def test_consent_flow_requests_readonly_and_saves_private_token(
    monkeypatch: pytest.MonkeyPatch, oauth_client_file: Path, tmp_path: Path
) -> None:
    seen = _install_flow(monkeypatch, [READONLY_SCOPE])
    data_dir = tmp_path / "state"
    result = authenticate(oauth(oauth_client_file), data_dir)
    assert seen == [[READONLY_SCOPE]]
    assert result.credentials.valid
    token = data_dir / "token.json"
    assert stat.S_IMODE(token.stat().st_mode) == 0o600
    assert json.loads(token.read_text())["scopes"] == [READONLY_SCOPE]


def test_consent_flow_rejects_broader_grant(
    monkeypatch: pytest.MonkeyPatch, oauth_client_file: Path, tmp_path: Path
) -> None:
    _install_flow(monkeypatch, [FULL_SCOPE])
    with pytest.raises(DriveAuthError, match=r"expected only the drive\.readonly"):
        authenticate(oauth(oauth_client_file), tmp_path)
    assert not (tmp_path / "token.json").exists()


def test_oauth_needs_client_file(tmp_path: Path) -> None:
    with pytest.raises(DriveAuthError, match="GOOGLE_OAUTH_CLIENT_FILE"):
        authenticate(oauth(None), tmp_path)


def test_oauth_given_service_account_key(sa_key_file: Path, tmp_path: Path) -> None:
    with pytest.raises(DriveAuthError, match="is a service account key"):
        authenticate(oauth(sa_key_file), tmp_path)


def test_build_drive_service_works_offline(sa_key_file: Path, tmp_path: Path) -> None:
    service = auth.build_drive_service(authenticate(sa(sa_key_file), tmp_path).credentials)
    assert hasattr(service, "files")

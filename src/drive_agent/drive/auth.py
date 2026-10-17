"""Google Drive authentication. Read-only by construction: only ``drive.readonly`` is requested.

Modes:

- ``service_account``: a service-account JSON key (``drive.credentials_file``), optionally
  impersonating a Workspace user via domain-wide delegation (``drive.impersonate_user``).
  With no key file, Application Default Credentials are used (Workload Identity on
  Cloud Run / GKE, or ``gcloud auth application-default login``).
- ``oauth``: a person signs in with their own Google account. The refresh token is cached in
  ``<data_dir>/token.json`` (owner-only permissions).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import google.auth
from google.auth.credentials import Credentials
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as UserCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from drive_agent.core.config import AuthMode, DriveSettings

READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
SCOPES = [READONLY_SCOPE]
TOKEN_FILE = "token.json"  # noqa: S105 (file name, not a secret)


class DriveAuthError(Exception):
    """Credentials are missing, invalid, or broader than read-only."""


@dataclass(frozen=True)
class DriveAuth:
    credentials: Credentials
    mode: AuthMode
    email: str | None
    """Identity the agent reads as, when known before the first API call."""
    description: str


def authenticate(drive: DriveSettings, data_dir: Path, *, interactive: bool = True) -> DriveAuth:
    """Return read-only Drive credentials for the configured auth mode."""
    if drive.auth_mode == "service_account":
        return _service_account(drive)
    return _oauth(drive, data_dir / TOKEN_FILE, interactive=interactive)


def build_drive_service(credentials: Credentials) -> Any:
    """Drive v3 client. Uses the discovery document bundled with the client library."""
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


# --- service account -------------------------------------------------------


def _service_account(drive: DriveSettings) -> DriveAuth:
    if drive.credentials_file is None:
        return _application_default(drive.impersonate_user)

    info = _read_json(drive.credentials_file, "service account key")
    if "installed" in info or "web" in info:
        raise DriveAuthError(
            f"{drive.credentials_file} is an OAuth client file, not a service account key. "
            "Set GOOGLE_AUTH_MODE=oauth and GOOGLE_OAUTH_CLIENT_FILE to use it, or download a "
            "JSON key from IAM & Admin -> Service Accounts -> Keys."
        )
    if info.get("type") != "service_account":
        raise DriveAuthError(f"{drive.credentials_file} is not a service account key file")

    try:
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    except (ValueError, KeyError) as exc:
        raise DriveAuthError(
            f"{drive.credentials_file}: invalid service account key: {exc}"
        ) from exc

    if drive.impersonate_user:
        return DriveAuth(
            credentials=creds.with_subject(drive.impersonate_user),
            mode="service_account",
            email=drive.impersonate_user,
            description=(
                f"service account {creds.service_account_email} "
                f"impersonating {drive.impersonate_user} (domain-wide delegation)"
            ),
        )
    return DriveAuth(
        credentials=creds,
        mode="service_account",
        email=creds.service_account_email,
        description=f"service account {creds.service_account_email}",
    )


def _application_default(impersonate_user: str | None) -> DriveAuth:
    try:
        creds, _project = google.auth.default(scopes=SCOPES)
    except DefaultCredentialsError as exc:
        raise DriveAuthError(
            "No service account key configured and no Application Default Credentials found. "
            "Set GOOGLE_APPLICATION_CREDENTIALS to the JSON key path, or run on Google Cloud "
            "with a service account attached."
        ) from exc

    email = getattr(creds, "service_account_email", None)
    if impersonate_user:
        if not hasattr(creds, "with_subject"):
            raise DriveAuthError(
                "Domain-wide delegation (GOOGLE_IMPERSONATE_USER) needs a service account key "
                "file; Application Default Credentials cannot impersonate users."
            )
        creds = creds.with_subject(impersonate_user)
        email = impersonate_user
    return DriveAuth(
        credentials=creds,
        mode="service_account",
        email=email if isinstance(email, str) and "@" in email else None,
        description="application default credentials"
        + (f" impersonating {impersonate_user}" if impersonate_user else ""),
    )


# --- OAuth -----------------------------------------------------------------


def _oauth(drive: DriveSettings, token_path: Path, *, interactive: bool) -> DriveAuth:
    creds = _load_token(token_path)

    if creds is not None and not creds.valid and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(token_path, creds)
        except RefreshError:
            creds = None  # revoked or expired refresh token: sign in again

    if creds is None or not creds.valid:
        if not interactive:
            raise DriveAuthError(
                "No valid Google sign-in. Run `drive-agent check` in a terminal to sign in."
            )
        creds = _run_consent_flow(drive.oauth_client_file)
        _save_token(token_path, creds)

    return DriveAuth(
        credentials=creds, mode="oauth", email=None, description="Google account (OAuth)"
    )


def _load_token(token_path: Path) -> UserCredentials | None:
    if not token_path.is_file():
        return None
    try:
        creds: UserCredentials = UserCredentials.from_authorized_user_file(str(token_path))
    except (ValueError, json.JSONDecodeError) as exc:
        raise DriveAuthError(f"{token_path} is corrupt; delete it and sign in again") from exc
    _require_readonly(creds.scopes, token_path)
    return creds


def _run_consent_flow(client_file: Path | None) -> UserCredentials:
    if client_file is None:
        raise DriveAuthError(
            "OAuth mode needs an OAuth client file: set GOOGLE_OAUTH_CLIENT_FILE "
            "(Google Cloud Console -> Credentials -> Create OAuth client ID -> Desktop app)."
        )
    info = _read_json(client_file, "OAuth client file")
    if info.get("type") == "service_account":
        raise DriveAuthError(
            f"{client_file} is a service account key. Set GOOGLE_AUTH_MODE=service_account "
            "and GOOGLE_APPLICATION_CREDENTIALS to use it."
        )
    if "installed" not in info and "web" not in info:
        raise DriveAuthError(f"{client_file} is not an OAuth client file")

    flow = InstalledAppFlow.from_client_config(info, scopes=SCOPES)
    creds: UserCredentials = flow.run_local_server(port=0, open_browser=True)
    _require_readonly(getattr(creds, "granted_scopes", None) or creds.scopes, client_file)
    return creds


def _require_readonly(scopes: Iterable[str] | None, source: Path) -> None:
    granted = set(scopes or ())
    if granted != {READONLY_SCOPE}:
        raise DriveAuthError(
            f"{source}: expected only the drive.readonly scope, got {sorted(granted) or 'none'}. "
            "Delete the cached token and sign in again."
        )


def _save_token(token_path: Path, creds: UserCredentials) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(creds.to_json())
    token_path.chmod(0o600)


def _read_json(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise DriveAuthError(f"{what} not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DriveAuthError(f"{path} is not valid JSON") from exc
    if not isinstance(data, dict):
        raise DriveAuthError(f"{path} is not a {what}")
    return data

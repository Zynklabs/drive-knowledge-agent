"""Turn Google API errors into actionable setup messages."""

from __future__ import annotations

import json
from typing import Any

from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

ENABLE_API_URL = "https://console.cloud.google.com/apis/library/drive.googleapis.com"


def _error_body(exc: HttpError) -> dict[str, Any]:
    try:
        body = json.loads(exc.content.decode("utf-8"))
    except (ValueError, AttributeError, UnicodeDecodeError):
        return {}
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, dict) else {}


def _reasons(error: dict[str, Any]) -> set[str]:
    reasons = {e.get("reason", "") for e in error.get("errors", []) if isinstance(e, dict)}
    reasons |= {d.get("reason", "") for d in error.get("details", []) if isinstance(d, dict)}
    reasons.add(str(error.get("status", "")))
    reasons.discard("")
    return reasons


def explain(exc: Exception, *, identity: str | None = None, folder_id: str | None = None) -> str:
    """A one-paragraph explanation of ``exc`` with the fix, for display to the operator."""
    who = identity or "the configured Google identity"
    if isinstance(exc, RefreshError):
        return _explain_refresh(exc, who)
    if not isinstance(exc, HttpError):
        return f"{type(exc).__name__}: {exc}"

    status = exc.resp.status
    error = _error_body(exc)
    reasons = _reasons(error)
    message = str(error.get("message", "")) or str(exc)
    target = f"folder {folder_id}" if folder_id else "the requested file"

    if (
        reasons & {"accessNotConfigured", "SERVICE_DISABLED"}
        or "has not been used in project" in message
    ):
        return (
            "The Google Drive API is not enabled for this Google Cloud project. "
            f"Enable it at {ENABLE_API_URL}, wait a minute, then retry."
        )
    if "exportSizeLimitExceeded" in reasons:
        return "Google can only export documents up to 10 MB; this one is too large to export."
    if reasons & {"insufficientPermissions", "ACCESS_TOKEN_SCOPE_INSUFFICIENT"}:
        return (
            "The credentials do not include the drive.readonly scope. "
            "For OAuth, delete the cached token and sign in again."
        )
    if "domainPolicy" in reasons:
        return (
            "Your Google Workspace policy blocks this access. Ask an admin to allow sharing "
            f"with {who}, or use domain-wide delegation."
        )
    if status == 404:
        return (
            f"{target[0].upper()}{target[1:]} was not found or is not shared with {who}. "
            "Check the ID, then share the folder with that address as Viewer "
            "(for a Shared Drive, add it as a member with Viewer access)."
        )
    if status == 403 and reasons & {"rateLimitExceeded", "userRateLimitExceeded"}:
        return "Google Drive rate limit reached. Wait a few minutes and retry."
    if status == 429:
        return "Google Drive rate limit reached. Wait a few minutes and retry."
    if status == 403:
        return (
            f"{who} has no access to {target}. Share it with that address as Viewer. "
            f"(Google said: {message})"
        )
    if status == 401:
        return f"Google rejected the credentials for {who} (expired or revoked). {message}"
    return f"Google Drive API error {status}: {message}"


def _explain_refresh(exc: RefreshError, who: str) -> str:
    text = str(exc)
    if "unauthorized_client" in text:
        return (
            "Domain-wide delegation is not authorized for this service account. A Workspace "
            "super-admin must add its client ID with the scope "
            "https://www.googleapis.com/auth/drive.readonly in admin.google.com -> Security -> "
            "API controls -> Domain-wide delegation."
        )
    if "account not found" in text:
        return (
            f"Google does not recognise {who}: the service account was deleted or the key "
            "belongs to another project. Create a new key in IAM & Admin -> Service Accounts."
        )
    if "invalid_grant" in text:
        return (
            f"Google refused to issue a token for {who}: the key may be deleted or disabled, "
            "the impersonated user may not exist, or the system clock is wrong. "
            f"({text})"
        )
    return f"Could not get a Google access token for {who}: {text}"

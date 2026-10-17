import pytest
from google.auth.exceptions import RefreshError

from drive_agent.drive.errors import explain

from .fake_drive import http_error


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (http_error(403, "accessNotConfigured", "Drive API has not been used"), "not enabled"),
        (http_error(403, "", "Drive API has not been used in project 123"), "not enabled"),
        (http_error(403, "insufficientPermissions"), "drive.readonly scope"),
        (http_error(403, "domainPolicy"), "Workspace policy"),
        (http_error(404, "notFound"), "not shared with sa@x"),
        (http_error(403, "rateLimitExceeded"), "rate limit"),
        (http_error(429, ""), "rate limit"),
        (http_error(403, "forbidden", "nope"), "has no access"),
        (http_error(401, "authError", "bad"), "rejected the credentials"),
        (http_error(500, "backendError", "oops"), "error 500: oops"),
        (RefreshError("unauthorized_client: Client is unauthorized"), "Domain-wide delegation"),
        (RefreshError("invalid_grant: Invalid email"), "refused to issue a token"),
        (RefreshError("invalid_grant: Invalid grant: account not found"), "does not recognise"),
        (RefreshError("something else"), "Could not get a Google access token"),
        (ValueError("boom"), "ValueError: boom"),
    ],
)
def test_explain(exc: Exception, expected: str) -> None:
    assert expected in explain(exc, identity="sa@x", folder_id="F1")


def test_not_found_mentions_folder() -> None:
    assert "Folder F1 was not found" in explain(http_error(404, "notFound"), folder_id="F1")


def test_unparseable_body() -> None:
    import httplib2
    from googleapiclient.errors import HttpError

    exc = HttpError(httplib2.Response({"status": 404}), b"<html>not json</html>")
    assert "not found" in explain(exc)

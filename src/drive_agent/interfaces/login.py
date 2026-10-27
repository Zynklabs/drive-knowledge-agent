"""Password check shared by the web app and the admin portal, slowed down against guessing.

Each wrong password costs ``FAILURE_DELAY`` seconds, and after ``LOCKOUT_FAILURES`` wrong
passwords within ``LOCKOUT_SECONDS`` every attempt is refused until the window passes. Counts
are per server process, so opening a new browser session doesn't reset them. For a hosted
deployment, put the app behind SSO (Streamlit's OIDC login or your proxy) as well.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Literal

from pydantic import SecretStr

FAILURE_DELAY = 2.0
LOCKOUT_FAILURES = 10
LOCKOUT_SECONDS = 15 * 60

Outcome = Literal["ok", "wrong", "locked"]


class PasswordGate:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._failures: deque[float] = deque()
        self._lock = threading.Lock()

    def check(self, attempt: str, secret: SecretStr) -> Outcome:
        now = self._clock()
        with self._lock:
            while self._failures and now - self._failures[0] >= LOCKOUT_SECONDS:
                self._failures.popleft()
            if len(self._failures) >= LOCKOUT_FAILURES:
                return "locked"
        if hmac.compare_digest(attempt.encode(), secret.get_secret_value().encode()):
            return "ok"
        with self._lock:
            self._failures.append(now)
        self._sleep(FAILURE_DELAY)
        return "wrong"


GATE = PasswordGate()

MESSAGES: dict[Outcome, str] = {
    "ok": "",
    "wrong": "Wrong password.",
    "locked": "Too many wrong passwords. Try again in 15 minutes.",
}

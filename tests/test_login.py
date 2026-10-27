from pydantic import SecretStr

from drive_agent.interfaces.login import FAILURE_DELAY, LOCKOUT_FAILURES, PasswordGate

SECRET = SecretStr("right")


def make_gate() -> tuple[PasswordGate, list[float], list[float]]:
    now, slept = [0.0], []
    return PasswordGate(clock=lambda: now[0], sleep=slept.append), now, slept


def test_right_password() -> None:
    gate, _, slept = make_gate()
    assert gate.check("right", SECRET) == "ok"
    assert slept == []


def test_wrong_password_is_slowed_down() -> None:
    gate, _, slept = make_gate()
    assert gate.check("wrong", SECRET) == "wrong"
    assert slept == [FAILURE_DELAY]


def test_lockout_after_repeated_failures_then_expiry() -> None:
    gate, now, _ = make_gate()
    for _ in range(LOCKOUT_FAILURES):
        assert gate.check("guess", SECRET) == "wrong"
    assert gate.check("right", SECRET) == "locked"  # even the right one, while locked
    now[0] += 15 * 60
    assert gate.check("right", SECRET) == "ok"

import importlib
import pkgutil
import re

import drive_agent

from .conftest import REPO_ROOT

# Any Drive scope other than the read-only ones is forbidden anywhere in the source.
WRITE_SCOPE = re.compile(r"googleapis\.com/auth/drive(?!\.readonly|\.metadata\.readonly)")


def test_all_modules_import() -> None:
    names = [
        info.name
        for info in pkgutil.walk_packages(drive_agent.__path__, "drive_agent.")
        if info.name != "drive_agent.__main__"
    ]
    assert "drive_agent.core.config" in names
    for name in names:
        importlib.import_module(name)


def test_no_write_scopes_in_source() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "src").rglob("*.py")
        if WRITE_SCOPE.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"non-read-only Drive scope found in: {offenders}"

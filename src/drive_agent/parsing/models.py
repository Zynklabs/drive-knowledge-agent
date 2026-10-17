"""Parser output: a document as an ordered list of sections with their structure."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class ParseError(Exception):
    """The file could not be read (corrupt, password-protected, wrong format)."""


@dataclass(frozen=True, slots=True)
class Section:
    text: str
    heading_path: tuple[str, ...] = ()
    """Enclosing headings, outermost first, e.g. ("Benefits", "Leave")."""
    page: int | None = None
    slide: int | None = None


@dataclass
class ParsedDocument:
    sections: list[Section] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalise line endings and whitespace without changing the words."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.replace("\u00a0", " ")
    text = _TRAILING_SPACE.sub("\n", text)
    return _BLANK_LINES.sub("\n\n", text).strip()

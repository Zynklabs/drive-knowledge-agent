"""File parsers keyed by content format (after Google-native files are exported)."""

from __future__ import annotations

from collections.abc import Callable

from drive_agent.drive.fetcher import ContentFormat
from drive_agent.parsing.models import ParsedDocument, ParseError, Section
from drive_agent.parsing.office import parse_docx, parse_pptx
from drive_agent.parsing.pdf import parse_pdf
from drive_agent.parsing.text import decode, parse_html, parse_json, parse_markdown, parse_text

PARSERS: dict[ContentFormat, Callable[[bytes], ParsedDocument]] = {
    "md": lambda data: parse_markdown(decode(data)),
    "txt": lambda data: parse_text(decode(data)),
    "html": lambda data: parse_html(decode(data)),
    "json": lambda data: parse_json(decode(data)),
    "pdf": parse_pdf,
    "docx": parse_docx,
    "pptx": parse_pptx,
}


def parse(data: bytes, fmt: ContentFormat) -> ParsedDocument:
    """Parse ``data``. Raises ``ParseError`` for unreadable files."""
    try:
        return PARSERS[fmt](data)
    except ParseError:
        raise
    except Exception as exc:  # third-party parsers raise many types on corrupt input
        raise ParseError(f"could not parse {fmt}: {type(exc).__name__}: {exc}") from exc


__all__ = ["PARSERS", "ParseError", "ParsedDocument", "Section", "parse"]

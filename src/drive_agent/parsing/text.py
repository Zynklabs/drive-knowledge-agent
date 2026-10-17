"""Parsers for text formats: Markdown (incl. Google Docs exports), plain text, HTML, JSON."""

from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup

from drive_agent.parsing.models import ParsedDocument, Section, normalize_text

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
# Google Docs' Markdown export embeds images as base64 data URIs; they are noise for Q&A.
_IMAGE_REF_DEF = re.compile(r"^\s*\[[^\]]+\]:\s*<?data:[^\s>]*>?\s*$")
_INLINE_IMAGE = re.compile(r"!\[[^\]]*\](\[[^\]]*\]|\([^)]*\))")
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!<>|~])")
_EMPHASIS = re.compile(r"[*_`]+")


def decode(data: bytes) -> str:
    return data.decode("utf-8-sig", errors="replace")


def _clean_heading(text: str) -> str:
    text = _ESCAPE.sub(r"\1", text)
    text = re.sub(r"\s*\{#[^}]*\}\s*$", "", text)  # trailing {#anchor}
    return _EMPHASIS.sub("", text).strip()


def parse_markdown(text: str) -> ParsedDocument:
    doc = ParsedDocument()
    path: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        body = normalize_text("\n".join(buffer))
        if body:
            doc.sections.append(Section(body, heading_path=tuple(t for _, t in path)))
        buffer.clear()

    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue
        if in_fence:
            buffer.append(line)
            continue
        if _IMAGE_REF_DEF.match(line):
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            level, title = len(heading.group(1)), _clean_heading(heading.group(2))
            path = [(lvl, t) for lvl, t in path if lvl < level]
            if title:
                path.append((level, title))
            continue
        buffer.append(_ESCAPE.sub(r"\1", _INLINE_IMAGE.sub("", line)))
    flush()
    return doc


def parse_text(text: str) -> ParsedDocument:
    body = normalize_text(text)
    return ParsedDocument(sections=[Section(body)] if body else [])


def parse_html(text: str) -> ParsedDocument:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "head"]):
        tag.decompose()
    for level in range(1, 7):
        for heading in soup.find_all(f"h{level}"):
            title = " ".join(heading.get_text(" ").split())
            heading.replace_with(f"\n\n{'#' * level} {title}\n\n" if title else "")
    return parse_markdown(soup.get_text("\n"))


def parse_json(text: str) -> ParsedDocument:
    try:
        pretty = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        doc = parse_text(text)
        doc.warnings.append("invalid JSON; indexed as plain text")
        return doc
    return parse_text(pretty)

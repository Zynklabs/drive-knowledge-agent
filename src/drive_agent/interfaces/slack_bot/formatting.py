"""Turn agent replies into Slack mrkdwn and Block Kit."""

from __future__ import annotations

import re
from typing import Any

from drive_agent.core.agent import AgentReply
from drive_agent.graph.rag import Citation

SECTION_LIMIT = 2900  # Slack's hard limit is 3000 characters per section
CONTEXT_ELEMENTS = 10  # Slack allows at most 10 elements per context block
FALLBACK_CHARS = 300
FEEDBACK_UP = "feedback_up"
FEEDBACK_DOWN = "feedback_down"

TABLE_WIDTH = 60
"""Widest table shown as an aligned monospace block; wider ones wrap on phones and lose their
alignment, so each row becomes one bullet line instead."""

_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*(?![\s*])(.+?)(?<![\s*])\*(?![*\w])")
_STRIKE = re.compile(r"~~(.+?)~~")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$", re.MULTILINE)
_BULLET = re.compile(r"^( *)[-*+]\s+", re.MULTILINE)
_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
_MENTION = re.compile(r"<@[A-Z0-9]+>")
FENCE = "```"


def escape(text: str) -> str:
    """Slack requires &, < and > to be escaped in message text."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_mrkdwn(markdown: str) -> str:
    """Convert the Markdown Claude writes into Slack mrkdwn.

    Slack has no tables, Markdown lists or ``*italic*``, so those would otherwise show up as
    raw pipes, dashes and asterisks. Code blocks are left as they are.
    """
    parts = _convert_tables(markdown).split(FENCE)
    for i in range(0, len(parts), 2):  # even parts are outside code blocks
        parts[i] = _inline_mrkdwn(escape(parts[i]))
    for i in range(1, len(parts), 2):
        parts[i] = escape(parts[i])
    return FENCE.join(parts)


def _inline_mrkdwn(text: str) -> str:
    text = _LINK.sub(lambda m: f"<{m.group(2)}|{m.group(1)}>", text)
    text = _RULE.sub("", text)
    text = _BULLET.sub(lambda m: "    " * (len(m.group(1)) // 2) + "• ", text)
    text = _ITALIC.sub(r"_\1_", text)  # before bold, which turns ** into Slack's single *
    text = _BOLD.sub(r"*\1*", text)
    text = _STRIKE.sub(r"~\1~", text)
    return _HEADING.sub(r"*\1*", text)


def _cells(line: str) -> list[str]:
    return [_BOLD.sub(r"\1", cell.strip()) for cell in line.strip().strip("|").split("|")]


def _convert_tables(markdown: str) -> str:
    """Replace each Markdown table (header row, divider row, body rows) with a Slack form."""
    lines = markdown.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and _TABLE_ROW.match(lines[i]) and _TABLE_DIVIDER.match(lines[i + 1]):
            header = _cells(lines[i])
            rows: list[list[str]] = []
            i += 2
            while i < len(lines) and _TABLE_ROW.match(lines[i]):
                rows.append(_cells(lines[i]))
                i += 1
            out.append(_render_table(header, rows))
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _render_table(header: list[str], rows: list[list[str]]) -> str:
    columns = len(header)
    rows = [(row + [""] * columns)[:columns] for row in rows]
    widths = [max(len(r[c]) for r in [header, *rows]) for c in range(columns)]
    if sum(widths) + 3 * (columns - 1) <= TABLE_WIDTH:

        def line(row: list[str]) -> str:
            cells = (cell.ljust(width) for cell, width in zip(row, widths, strict=True))
            return " | ".join(cells).rstrip()

        rule = "-+-".join("-" * width for width in widths)
        body = "\n".join([line(header), rule, *(line(row) for row in rows)])
        return f"{FENCE}\n{body}\n{FENCE}"
    bullets = []
    for first, *rest in rows:
        details = " · ".join(
            f"{name}: {value}" if name else value
            for name, value in zip(header[1:], rest, strict=True)
            if value
        )
        bullets.append(f"- **{first}**" + (f" — {details}" if details else ""))
    return "\n".join(bullets)


def strip_mentions(text: str) -> str:
    return _MENTION.sub("", text).strip()


def split_sections(text: str, limit: int = SECTION_LIMIT) -> list[str]:
    """Split on paragraph, then line, boundaries so each piece fits one section block.

    A code block cut in two is closed at the end of one piece and reopened in the next.
    """
    pieces = _split(text, limit - 2 * (len(FENCE) + 1))
    for i in range(len(pieces) - 1):
        if pieces[i].count(FENCE) % 2:
            pieces[i] += f"\n{FENCE}"
            pieces[i + 1] = f"{FENCE}\n{pieces[i + 1]}"
    return pieces


def _split(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            pieces.append(current)
        while len(paragraph) > limit:
            cut = paragraph.rfind("\n", 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(paragraph[:cut])
            paragraph = paragraph[cut:].lstrip("\n")
        current = paragraph
    if current:
        pieces.append(current)
    return pieces or [""]


def _source_label(citation: Citation) -> str:
    name = citation.get("file_name", "source")
    if "page" in citation:
        name += f" · p.{citation['page']}"
    elif "slide" in citation:
        name += f" · slide {citation['slide']}"
    label = escape(f"[{citation['number']}] {name}")
    url = citation.get("web_url")
    return f"<{url}|{label.replace('|', '¦')}>" if url else label


def reply_blocks(reply: AgentReply, *, feedback_value: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": piece}}
        for piece in split_sections(to_mrkdwn(reply.answer))
    ]
    citations = sorted(reply.citations, key=lambda c: c["number"])
    if citations:
        labels = [_source_label(c) for c in citations]
        if len(labels) > CONTEXT_ELEMENTS:
            hidden = len(labels) - CONTEXT_ELEMENTS + 1
            labels = [*labels[: CONTEXT_ELEMENTS - 1], f"+{hidden} more"]
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": label} for label in labels],
            }
        )
    if reply.status == "answered":
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "action_id": FEEDBACK_UP,
                        "text": {"type": "plain_text", "text": "👍 Helpful"},
                        "value": feedback_value,
                    },
                    {
                        "type": "button",
                        "action_id": FEEDBACK_DOWN,
                        "text": {"type": "plain_text", "text": "👎 Not helpful"},
                        "value": feedback_value,
                    },
                ],
            }
        )
    return blocks


def fallback_text(answer: str) -> str:
    """Plain text for notifications and clients that can't show blocks."""
    return answer if len(answer) <= FALLBACK_CHARS else answer[: FALLBACK_CHARS - 1] + "…"

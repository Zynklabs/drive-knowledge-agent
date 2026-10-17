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

_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$", re.MULTILINE)
_MENTION = re.compile(r"<@[A-Z0-9]+>")


def escape(text: str) -> str:
    """Slack requires &, < and > to be escaped in message text."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_mrkdwn(markdown: str) -> str:
    """Convert the Markdown Claude writes into Slack mrkdwn."""
    text = escape(markdown)
    text = _LINK.sub(lambda m: f"<{m.group(2)}|{m.group(1)}>", text)
    text = _BOLD.sub(r"*\1*", text)
    return _HEADING.sub(r"*\1*", text)


def strip_mentions(text: str) -> str:
    return _MENTION.sub("", text).strip()


def split_sections(text: str, limit: int = SECTION_LIMIT) -> list[str]:
    """Split on paragraph, then line, boundaries so each piece fits one section block."""
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

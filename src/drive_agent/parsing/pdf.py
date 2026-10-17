"""PDF text extraction with pypdf (BSD-licensed; PyMuPDF is avoided because it is AGPL)."""

from __future__ import annotations

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from drive_agent.parsing.models import ParsedDocument, ParseError, Section, normalize_text

# Bullet glyphs: • ● ○ ◦ ▪ ▫ ■ □ ‣ hyphen-bullet ► ▶ ➢ ✓ ✔
_BULLETS = frozenset(
    "\u2022\u25cf\u25cb\u25e6\u25aa\u25ab\u25a0\u25a1\u2023\u2043\u25ba\u25b6\u27a2\u2713\u2714"
)
_NO_SPACE_BEFORE = frozenset(",.;:!?)%")


def join_fragments(text: str) -> str:
    """Repair PDF layout artifacts: bullets and lone punctuation extracted onto their own lines.

    ``"●\\nBuilt APIs"`` becomes ``"● Built APIs"``;
    ``"Python\\n&\\nGo"`` becomes ``"Python & Go"``.
    """
    out: list[str] = []
    bullet: str | None = None
    glue_next = False
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            out.append("")
            glue_next = False
            continue
        if line in _BULLETS:
            bullet = line
            continue
        if bullet is not None:
            line, bullet = f"{bullet} {line}", None
        elif len(line) == 1 and not line.isalnum() and out and out[-1]:
            sep = "" if line in _NO_SPACE_BEFORE else " "
            out[-1] = f"{out[-1]}{sep}{line}"
            glue_next = True
            continue
        if glue_next and out and out[-1]:
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
        glue_next = False
    if bullet is not None:
        out.append(bullet)
    return "\n".join(out)


def parse_pdf(data: bytes) -> ParsedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ParseError("password-protected PDF")
        pages = list(reader.pages)
    except PdfReadError as exc:
        raise ParseError(f"unreadable PDF: {exc}") from exc

    doc = ParsedDocument()
    empty = 0
    for number, page in enumerate(pages, start=1):
        text = normalize_text(join_fragments(page.extract_text() or ""))
        if text:
            doc.sections.append(Section(text, page=number))
        else:
            empty += 1

    if pages and empty == len(pages):
        doc.warnings.append("no extractable text (scanned PDF?); OCR is not supported yet")
    elif empty:
        doc.warnings.append(f"{empty} of {len(pages)} pages have no extractable text")
    return doc

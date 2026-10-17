"""Word (.docx) and PowerPoint (.pptx) parsers. Google Slides are exported to .pptx first."""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterator
from typing import Any

import docx
from docx.table import Table
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from drive_agent.parsing.models import ParsedDocument, ParseError, Section, normalize_text

_HEADING_STYLE = re.compile(r"^Heading (\d)$")


def _table_rows(table: Any) -> list[str]:
    rows = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = " ".join(cell.text.split())
            if not cells or text != cells[-1]:  # merged cells repeat their text
                cells.append(text)
        if any(cells):
            rows.append(" | ".join(cells))
    return rows


def parse_docx(data: bytes) -> ParsedDocument:
    try:
        document = docx.Document(io.BytesIO(data))
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise ParseError(f"unreadable Word document: {exc}") from exc

    doc = ParsedDocument()
    path: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        body = normalize_text("\n".join(buffer))
        if body:
            doc.sections.append(Section(body, heading_path=tuple(t for _, t in path)))
        buffer.clear()

    for block in document.iter_inner_content():
        if isinstance(block, Table):
            buffer.extend(_table_rows(block))
            continue
        text = block.text.strip()
        style = block.style.name if block.style is not None else ""
        match = _HEADING_STYLE.match(style or "")
        level = 0 if style == "Title" else int(match.group(1)) if match else None
        if level is None or not text:
            buffer.append(text)
            continue
        flush()
        path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, text)]
    flush()
    return doc


def _iter_shapes(shapes: Any) -> Iterator[Any]:
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_shapes(shape.shapes)
        else:
            yield shape


def parse_pptx(data: bytes) -> ParsedDocument:
    try:
        presentation = Presentation(io.BytesIO(data))
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise ParseError(f"unreadable presentation: {exc}") from exc

    doc = ParsedDocument()
    for number, slide in enumerate(presentation.slides, start=1):
        title_shape = slide.shapes.title
        title = " ".join(title_shape.text_frame.text.split()) if title_shape is not None else ""
        parts: list[str] = []
        for shape in _iter_shapes(slide.shapes):
            if title_shape is not None and shape.shape_id == title_shape.shape_id:
                continue
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            elif getattr(shape, "has_table", False):
                parts.extend(_table_rows(shape.table))
        if slide.has_notes_slide:
            notes_frame = slide.notes_slide.notes_text_frame
            notes = normalize_text(notes_frame.text) if notes_frame is not None else ""
            if notes:
                parts.append(f"Speaker notes:\n{notes}")
        body = normalize_text("\n".join([title, *parts]))
        if body:
            doc.sections.append(Section(body, heading_path=(title,) if title else (), slide=number))
    return doc

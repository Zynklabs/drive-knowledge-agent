"""Split parsed sections into overlapping chunks, emitted as LangChain ``Document`` objects.

Sizes are measured in approximate tokens (about 4 characters per token), which is close
enough for Claude/Voyage/OpenAI tokenizers and needs no tokenizer download.
"""

from __future__ import annotations

import math
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from drive_agent.core.config import IngestSettings
from drive_agent.drive.models import DriveFile
from drive_agent.parsing.models import ParsedDocument, Section

CHARS_PER_TOKEN = 4
# Paragraphs, then lines, then sentence ends, so chunks rarely break mid-sentence.
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _context_line(file: DriveFile, section: Section) -> str:
    parts = [file.name, *section.heading_path]
    if section.page is not None:
        parts.append(f"page {section.page}")
    if section.slide is not None:
        parts.append(f"slide {section.slide}")
    return " > ".join(parts)


def _metadata(file: DriveFile, section: Section, index: int) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "chunk_id": f"{file.id}:{index}",
        "chunk_index": index,
        "file_id": file.id,
        "file_name": file.name,
        "kind": file.kind,
        "mime_type": file.mime_type,
        "path": file.path,
        "web_url": file.web_url,
        "section": " > ".join(section.heading_path) or None,
        "page": section.page,
        "slide": section.slide,
        "modified_time": file.modified_time,
        "md5": file.md5,
        "owner": file.owner,
        "drive_id": file.drive_id,
        "folder_root": file.root_folder_id,
    }
    # Several vector stores (e.g. Chroma) reject None metadata values.
    return {key: value for key, value in meta.items() if value is not None}


def chunk_document(
    file: DriveFile, parsed: ParsedDocument, settings: IngestSettings
) -> list[Document]:
    """Chunk every section of ``parsed``. Chunk IDs are stable: ``<file_id>:<n>``."""
    chunks: list[Document] = []
    for section in parsed.sections:
        header = _context_line(file, section)
        budget = max(settings.chunk_tokens - approx_tokens(header) - 1, 50)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=min(settings.chunk_overlap, budget // 2),
            length_function=approx_tokens,
            separators=SEPARATORS,
            keep_separator="end",
            strip_whitespace=True,
        )
        for piece in splitter.split_text(section.text):
            chunks.append(
                Document(
                    page_content=f"{header}\n\n{piece}",
                    metadata=_metadata(file, section, len(chunks)),
                )
            )
    return chunks

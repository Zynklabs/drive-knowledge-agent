"""Map Drive MIME types (with a file-extension fallback) to the file kinds we index."""

from __future__ import annotations

from pathlib import PurePosixPath

from drive_agent.core.config import FileKind

FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"

MIME_KINDS: dict[str, FileKind] = {
    "application/vnd.google-apps.document": "docs",
    "application/vnd.google-apps.presentation": "slides",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/x-markdown": "md",
    "text/html": "html",
    "application/json": "json",
}

EXTENSION_KINDS: dict[str, FileKind] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
}

# Spreadsheets and tabular data are out of scope (can be huge, chunk poorly).
OUT_OF_SCOPE_MIMES = frozenset(
    {
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.ms-excel.sheet.macroEnabled.12",
        "text/csv",
        "text/tab-separated-values",
    }
)
OUT_OF_SCOPE_EXTENSIONS = frozenset({".xlsx", ".xls", ".xlsm", ".csv", ".tsv"})


def _extension(name: str) -> str:
    return PurePosixPath(name).suffix.lower()


def is_out_of_scope(name: str, mime_type: str) -> bool:
    return mime_type in OUT_OF_SCOPE_MIMES or _extension(name) in OUT_OF_SCOPE_EXTENSIONS


def file_kind(name: str, mime_type: str) -> FileKind | None:
    """The indexable kind of a file, or None if unsupported.

    The MIME type wins; the extension is only consulted when the MIME type is unknown
    (e.g. a Markdown file uploaded as ``application/octet-stream``). Google-native types
    are never guessed from the name.
    """
    if mime_type in MIME_KINDS:
        return MIME_KINDS[mime_type]
    if mime_type.startswith("application/vnd.google-apps."):
        return None
    return EXTENSION_KINDS.get(_extension(name))

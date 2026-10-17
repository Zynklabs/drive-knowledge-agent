from drive_agent.chunking import approx_tokens, chunk_document
from drive_agent.core.config import IngestSettings
from drive_agent.drive.models import DriveFile
from drive_agent.parsing import ParsedDocument, Section

FILE = DriveFile(
    id="F1",
    name="Leave Policy",
    mime_type="application/vnd.google-apps.document",
    kind="docs",
    path="HR/Leave Policy",
    root_folder_id="ROOT",
    web_url="https://docs.google.com/document/d/F1/edit",
    modified_time="2026-09-30T10:12:00Z",
)


def sentences(n: int) -> str:
    return " ".join(f"Sentence number {i} describes one leave rule in detail." for i in range(n))


def test_small_sections_one_chunk_each_with_context_and_metadata() -> None:
    parsed = ParsedDocument(
        sections=[
            Section("Intro."),
            Section("24 days.", heading_path=("Benefits", "Leave")),
            Section("Page text.", page=3),
        ]
    )
    chunks = chunk_document(FILE, parsed, IngestSettings())
    assert [c.page_content for c in chunks] == [
        "Leave Policy\n\nIntro.",
        "Leave Policy > Benefits > Leave\n\n24 days.",
        "Leave Policy > page 3\n\nPage text.",
    ]
    assert chunks[1].metadata == {
        "chunk_id": "F1:1",
        "chunk_index": 1,
        "file_id": "F1",
        "file_name": "Leave Policy",
        "kind": "docs",
        "mime_type": "application/vnd.google-apps.document",
        "path": "HR/Leave Policy",
        "web_url": "https://docs.google.com/document/d/F1/edit",
        "section": "Benefits > Leave",
        "modified_time": "2026-09-30T10:12:00Z",
        "folder_root": "ROOT",
    }
    assert chunks[2].metadata["page"] == 3
    assert all(v is not None for c in chunks for v in c.metadata.values())


def test_long_section_is_split_within_budget_with_overlap() -> None:
    settings = IngestSettings(chunk_tokens=100, chunk_overlap=20)
    parsed = ParsedDocument(sections=[Section(sentences(60), heading_path=("Rules",))])
    chunks = chunk_document(FILE, parsed, settings)

    assert len(chunks) > 5
    assert [c.metadata["chunk_id"] for c in chunks] == [f"F1:{i}" for i in range(len(chunks))]
    for chunk in chunks:
        assert approx_tokens(chunk.page_content) <= settings.chunk_tokens
        assert chunk.page_content.startswith("Leave Policy > Rules\n\n")
        body = chunk.page_content.split("\n\n", 1)[1]
        assert body.endswith("."), body  # split at sentence ends
    first, second = (c.page_content.split("\n\n", 1)[1] for c in chunks[:2])
    assert first.split(". ")[-1] in second  # overlap carries context forward


def test_chunk_ids_are_deterministic() -> None:
    parsed = ParsedDocument(sections=[Section(sentences(40)), Section("tail", slide=2)])
    settings = IngestSettings(chunk_tokens=120, chunk_overlap=10)
    first = chunk_document(FILE, parsed, settings)
    second = chunk_document(FILE, parsed, settings)
    assert [(c.page_content, c.metadata) for c in first] == [
        (c.page_content, c.metadata) for c in second
    ]
    assert first[-1].metadata["slide"] == 2


def test_empty_document_has_no_chunks() -> None:
    assert chunk_document(FILE, ParsedDocument(), IngestSettings()) == []

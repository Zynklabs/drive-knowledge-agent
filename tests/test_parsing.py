import json

import pytest

from drive_agent.parsing import ParseError, Section, parse
from drive_agent.parsing.models import normalize_text
from drive_agent.parsing.text import parse_html, parse_markdown

from .doc_factory import encrypt_pdf, make_docx, make_pdf, make_pptx

# --- markdown (incl. Google Docs export) ---------------------------------------


def test_markdown_heading_paths() -> None:
    doc = parse_markdown(
        "Intro text.\n\n# Benefits\n\n## Leave\n\n24 days.\n\n## Health\n\nFamily cover.\n\n"
        "# Conduct\n\nBe kind.\n"
    )
    assert doc.sections == [
        Section("Intro text."),
        Section("24 days.", heading_path=("Benefits", "Leave")),
        Section("Family cover.", heading_path=("Benefits", "Health")),
        Section("Be kind.", heading_path=("Conduct",)),
    ]


def test_markdown_ignores_headings_in_code_fences() -> None:
    doc = parse_markdown("# Setup\n\n```bash\n# not a heading\nrun\n```\n")
    assert doc.sections == [Section("```bash\n# not a heading\nrun\n```", heading_path=("Setup",))]


def test_google_docs_export_noise_is_removed() -> None:
    exported = (
        "# **Leave Policy** {#leave-policy}\n\n"
        "Employees get 24 days\\. See ![][image1] the table\\!\n\n"
        "![logo](data:image/png;base64,AAAA)\n\n"
        "[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB>\n"
    )
    doc = parse_markdown(exported)
    assert doc.sections == [
        Section("Employees get 24 days. See  the table!", heading_path=("Leave Policy",))
    ]
    assert "base64" not in json.dumps([s.text for s in doc.sections])


def test_normalize_text() -> None:
    assert normalize_text("a\r\nb  \n\n\n\nc\x00\u00a0d ") == "a\nb\n\nc d"


# --- text, html, json ----------------------------------------------------------


def test_plain_text_and_bom() -> None:
    assert parse(b"\xef\xbb\xbfhello\n", "txt").sections == [Section("hello")]
    assert parse(b"   \n", "txt").sections == []


def test_html_headings_and_noise() -> None:
    doc = parse_html(
        "<html><head><title>x</title><style>p{}</style></head><body>"
        "<script>alert(1)</script><h1>Guide</h1><p>Start here.</p>"
        "<h2>Install</h2><p>Run <b>it</b>.</p></body></html>"
    )
    assert doc.sections == [
        Section("Start here.", heading_path=("Guide",)),
        Section("Run\nit\n.", heading_path=("Guide", "Install")),
    ]


def test_json_pretty_printed() -> None:
    doc = parse(b'{"team":"Platform","heads":3}', "json")
    assert doc.sections[0].text == '{\n  "team": "Platform",\n  "heads": 3\n}'
    assert doc.warnings == []


def test_invalid_json_falls_back_to_text() -> None:
    doc = parse(b"{not json", "json")
    assert doc.sections == [Section("{not json")]
    assert doc.warnings == ["invalid JSON; indexed as plain text"]


# --- pdf -----------------------------------------------------------------------


def test_pdf_pages() -> None:
    doc = parse(make_pdf(["Jane Doe\nSenior Engineer", "Experience: 8 years"]), "pdf")
    assert [(s.page, s.text) for s in doc.sections] == [
        (1, "Jane Doe\nSenior Engineer"),
        (2, "Experience: 8 years"),
    ]
    assert doc.warnings == []


def test_pdf_partially_empty() -> None:
    doc = parse(make_pdf(["Page one", ""]), "pdf")
    assert [s.page for s in doc.sections] == [1]
    assert doc.warnings == ["1 of 2 pages have no extractable text"]


def test_scanned_pdf_warning() -> None:
    doc = parse(make_pdf(["", ""]), "pdf")
    assert doc.sections == []
    assert "scanned PDF" in doc.warnings[0]


def test_password_protected_pdf() -> None:
    with pytest.raises(ParseError, match="password-protected"):
        parse(encrypt_pdf(make_pdf(["secret"]), "pw"), "pdf")


def test_pdf_with_empty_user_password_opens() -> None:
    doc = parse(encrypt_pdf(make_pdf(["open"]), ""), "pdf")
    assert doc.sections[0].text == "open"


@pytest.mark.parametrize("fmt", ["pdf", "docx", "pptx"])
def test_corrupt_binary_files(fmt: str) -> None:
    with pytest.raises(ParseError):
        parse(b"definitely not a real file", fmt)  # type: ignore[arg-type]


# --- docx ----------------------------------------------------------------------


def test_docx_structure_and_tables() -> None:
    doc = parse(make_docx(), "docx")
    assert doc.sections == [
        Section("Preamble before any heading."),
        Section("Welcome to the company.", heading_path=("Employee Handbook",)),
        Section(
            "Employees get 24 days of paid leave.\nType | Days\nSick | 12",
            heading_path=("Employee Handbook", "Benefits", "Leave"),
        ),
        Section(
            "Insurance covers family.", heading_path=("Employee Handbook", "Benefits", "Health")
        ),
        Section("Be kind.", heading_path=("Employee Handbook", "Conduct")),
    ]


# --- pptx (and exported Google Slides) -----------------------------------------


def test_pptx_slides_notes_tables_groups() -> None:
    doc = parse(make_pptx(), "pptx")
    assert doc.sections == [
        Section(
            "Q3 Results\nRevenue grew 40%\nSpeaker notes:\nMention the APAC launch.",
            heading_path=("Q3 Results",),
            slide=1,
        ),
        Section(
            "Hiring plan\nTeam | Heads\nPlatform | 3\nGrouped note",
            heading_path=("Hiring plan",),
            slide=3,
        ),
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("●\nBuilt APIs\n○\nScaled to 1M users", "● Built APIs\n○ Scaled to 1M users"),
        ("Python\n&\nGo", "Python & Go"),
        ("Led a team\n,\nshipped fast", "Led a team, shipped fast"),
        ("Skills\n\n•\n\nRust", "Skills\n\n\n• Rust"),
        ("&\nstarts with symbol", "&\nstarts with symbol"),
        ("Total\n3\nitems", "Total\n3\nitems"),  # single digits/letters are left alone
        ("trailing bullet\n●", "trailing bullet\n●"),
    ],
)
def test_join_pdf_fragments(raw: str, expected: str) -> None:
    from drive_agent.parsing.pdf import join_fragments

    assert join_fragments(raw) == expected


def test_pdf_fragments_rejoined() -> None:
    doc = parse(make_pdf(["Skills\nPython\n&\nGo\n,\nRust"]), "pdf")
    assert doc.sections[0].text == "Skills\nPython & Go, Rust"


def test_office_zip_bomb_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from drive_agent.parsing import office

    monkeypatch.setattr(office, "MAX_UNZIPPED_BYTES", 1000)
    for fmt, data in (("docx", make_docx()), ("pptx", make_pptx())):
        with pytest.raises(ParseError, match="too large once unzipped"):
            parse(data, fmt)  # type: ignore[arg-type]


def test_pdf_text_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    from drive_agent.parsing import pdf

    monkeypatch.setattr(pdf, "MAX_TEXT_CHARS", 10)
    doc = parse(make_pdf(["First page text.", "Second.", "Third."]), "pdf")
    assert [s.page for s in doc.sections] == [1]
    assert doc.warnings == ["text limit reached; pages 2-3 not indexed"]

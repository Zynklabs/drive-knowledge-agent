"""Build small real PDF / DOCX / PPTX files in memory for parser tests."""

from __future__ import annotations

import io

import docx
from pptx import Presentation
from pptx.util import Inches
from pypdf import PdfReader, PdfWriter


def make_pdf(pages: list[str]) -> bytes:
    """A minimal text PDF; an empty string makes a page with no text (like a scan)."""
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        ops = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
        for line in text.split("\n") if text else []:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({escaped}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode() if text else b""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


def encrypt_pdf(data: bytes, password: str) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(data)))
    writer.encrypt(user_password=password, algorithm="RC4-128")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_docx() -> bytes:
    document = docx.Document()
    document.add_paragraph("Preamble before any heading.")
    document.add_heading("Employee Handbook", 0)  # Title style
    document.add_paragraph("Welcome to the company.")
    document.add_heading("Benefits", 1)
    document.add_heading("Leave", 2)
    document.add_paragraph("Employees get 24 days of paid leave.")
    table = document.add_table(rows=2, cols=3)
    table.cell(0, 0).text = "Type"
    table.cell(0, 1).text = "Days"
    table.cell(0, 2).text = "Days"
    table.cell(1, 0).text = "Sick"
    table.cell(1, 1).merge(table.cell(1, 2)).text = "12"
    document.add_heading("Health", 2)
    document.add_paragraph("Insurance covers family.")
    document.add_heading("Conduct", 1)
    document.add_paragraph("Be kind.")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_pptx() -> bytes:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])  # title + content
    slide.shapes.title.text = "Q3 Results"
    slide.placeholders[1].text = "Revenue grew 40%"
    slide.notes_slide.notes_text_frame.text = "Mention the APAC launch."

    prs.slides.add_slide(prs.slide_layouts[6])  # blank slide: no section

    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title only
    slide.shapes.title.text = "Hiring plan"
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    table.cell(0, 0).text = "Team"
    table.cell(0, 1).text = "Heads"
    table.cell(1, 0).text = "Platform"
    table.cell(1, 1).text = "3"
    group = slide.shapes.add_group_shape()
    group.shapes.add_textbox(
        Inches(1), Inches(4), Inches(3), Inches(1)
    ).text_frame.text = "Grouped note"
    buffer = io.BytesIO()
    prs.save(buffer)
    return buffer.getvalue()

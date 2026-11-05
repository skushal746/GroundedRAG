import io

import docx
import pymupdf
import pytest

from app.ingestion.parsers import DOCX_MIME, resolve_parser
from app.ingestion.parsers.docx import parse_docx
from app.ingestion.parsers.pdf import parse_pdf
from app.ingestion.parsers.text import parse_html, parse_markdown, parse_text
from app.ingestion.types import ParseError, UnsupportedFormatError


def _pdf_bytes() -> bytes:
    """3-page PDF: big heading, body text, a bordered table on page 2, running footer."""
    html_pages = [
        "<h1>Guide</h1><p>Intro paragraph about workspaces.</p><h2>Setup</h2><p>Step one.</p>",
        "<h2>Limits</h2><p>See table.</p><table>"
        "<tr><th>Plan</th><th>Storage</th><th>Seats</th></tr>"
        "<tr><td>Team</td><td>100 GB</td><td>5</td></tr>"
        "<tr><td>Business</td><td>1 TB</td><td>10</td></tr></table>",
        "<h2>Security</h2><p>Deleted workspaces can be restored for 30 days.</p>",
    ]
    css = (
        "table { border-collapse: collapse; } th, td { border: 1px solid black; padding: 3px 5px; }"
        " h1 { font-size: 20pt; }"
    )
    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    rect = pymupdf.paper_rect("a4")
    for html in html_pages:
        story = pymupdf.Story(html=html, user_css=css)
        device = writer.begin_page(rect)
        story.place(pymupdf.Rect(50, 50, rect.width - 50, rect.height - 80))
        story.draw(device)
        writer.end_page()
    writer.close()
    doc = pymupdf.open(stream=buf.getvalue(), filetype="pdf")
    for n, page in enumerate(doc, start=1):
        page.insert_text((50, rect.height - 40), f"Acme Guide - Page {n} of 3", fontsize=8)
    return bytes(doc.tobytes())


def test_pdf_keeps_pages_headings_tables_and_drops_footers() -> None:
    parsed = parse_pdf(_pdf_bytes(), "guide.pdf")
    assert parsed.num_pages == 3
    assert parsed.title == "Guide"
    by_text = {b.text: b for b in parsed.blocks}
    assert by_text["Step one."].headings == ("Guide", "Setup")
    assert by_text["Step one."].page == 1
    table = next(b for b in parsed.blocks if b.kind == "table")
    assert table.page == 2
    assert "| Team | 100 GB | 5 |" in table.text
    assert table.headings[-1] == "Limits"
    assert any(b.page == 3 and "30 days" in b.text for b in parsed.blocks)
    assert not any("Page" in b.text and "of 3" in b.text for b in parsed.blocks)


def test_pdf_rejects_garbage() -> None:
    with pytest.raises(ParseError):
        parse_pdf(b"not a pdf", "x.pdf")


def test_pdf_without_text_is_a_parse_error() -> None:
    doc = pymupdf.open()
    doc.new_page()
    with pytest.raises(ParseError, match="OCR"):
        parse_pdf(doc.tobytes(), "scan.pdf")


def _docx_bytes() -> bytes:
    document = docx.Document()
    document.core_properties.title = "Billing Policy"
    document.add_heading("Refunds", level=1)
    document.add_heading("Annual plans", level=2)
    document.add_paragraph("Refundable within 30 days.")
    document.add_paragraph("First bullet", style="List Bullet")
    document.add_heading("Pricing", level=1)
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Plan", "Price"
    table.cell(1, 0).text, table.cell(1, 1).text = "Team", "$12"
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def test_docx_heading_paths_lists_and_tables_in_order() -> None:
    parsed = parse_docx(_docx_bytes(), "billing.docx")
    assert parsed.title == "Billing Policy"
    kinds = [(b.kind, b.headings) for b in parsed.blocks]
    assert kinds == [
        ("paragraph", ("Refunds", "Annual plans")),
        ("list", ("Refunds", "Annual plans")),
        ("table", ("Pricing",)),
    ]
    assert parsed.blocks[1].text == "- First bullet"
    assert "| Team | $12 |" in parsed.blocks[2].text


def test_docx_rejects_garbage() -> None:
    with pytest.raises(ParseError):
        parse_docx(b"PK-not-really", "x.docx")


def test_markdown_atx_setext_fences_and_tables() -> None:
    md = b"""# FAQ

## Billing
Refunds are
possible.

| Plan | Price |
|---|---|
| Team | $12 |

Other
-----
```
code block
```
"""
    parsed = parse_markdown(md, "faq.md")
    assert parsed.title == "FAQ"
    assert [(b.kind, b.headings) for b in parsed.blocks] == [
        ("paragraph", ("FAQ", "Billing")),
        ("table", ("FAQ", "Billing")),
        ("code", ("FAQ", "Other")),
    ]
    assert parsed.blocks[0].text == "Refunds are possible."


def test_text_uses_underlined_headings() -> None:
    txt = "Release Notes\n=============\n\nv1.2\n----\nAdded things.\n\nFixed bugs.\n"
    parsed = parse_text(txt.encode(), "notes.txt")
    assert parsed.title == "Release Notes"
    assert [b.headings for b in parsed.blocks] == [("Release Notes", "v1.2")] * 2


def test_text_empty_is_error() -> None:
    with pytest.raises(ParseError):
        parse_text(b"   \n\n", "empty.txt")


def test_text_decodes_cp1252() -> None:
    parsed = parse_text("caf\xe9 menu".encode("cp1252"), "menu.txt")
    assert parsed.blocks[0].text == "café menu"


def test_html_structure_and_noise_removal() -> None:
    html = b"""<html><head><title>API Guide</title><script>evil()</script></head><body>
    <nav>Home</nav><h1>API Guide</h1><h2>Auth</h2>
    <p>Send a <code>Bearer</code> token.</p>
    <ul><li>Tokens expire after <b>90 days</b>.</li></ul>
    <table><tr><th>Code</th><th>Meaning</th></tr><tr><td>429</td><td>Too many</td></tr></table>
    <footer>(c) Acme</footer></body></html>"""
    parsed = parse_html(html, "api.html")
    assert parsed.title == "API Guide"
    texts = [b.text for b in parsed.blocks]
    assert texts[0] == "Send a Bearer token."
    assert texts[1] == "- Tokens expire after 90 days."
    assert "| 429 | Too many |" in texts[2]
    assert all("evil" not in t and "Acme" not in t and "Home" not in t for t in texts)
    assert parsed.blocks[0].headings == ("API Guide", "Auth")


def test_registry_extension_wins_over_generic_mime() -> None:
    assert resolve_parser("faq.md", "application/octet-stream")[0] == "text/markdown"
    assert resolve_parser("policy.docx", None)[0] == DOCX_MIME
    assert resolve_parser("upload", "text/html")[0] == "text/html"


def test_registry_rejects_unknown_types() -> None:
    with pytest.raises(UnsupportedFormatError, match=r"\.exe"):
        resolve_parser("setup.exe", "application/x-msdownload")
    with pytest.raises(UnsupportedFormatError):
        resolve_parser("noext", None)

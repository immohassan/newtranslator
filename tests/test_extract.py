"""Extraction must capture text, styling, position and images."""
from app.core.extract import extract, extract_docx, extract_pdf
from app.core.qa import QAReport

import pytest


def test_pdf_extracts_blocks_and_styles(sample_pdf):
    doc = extract_pdf(sample_pdf, QAReport())
    assert doc.kind == "pdf"
    page = doc.pages[0]
    assert page.width == pytest.approx(595, abs=1)

    texts = [b.text for b in page.blocks]
    assert any("Quarterly Report" in t for t in texts)
    assert any("Key Results" in t for t in texts)


def test_pdf_headline_size_and_bold(sample_pdf):
    doc = extract_pdf(sample_pdf, QAReport())
    blocks = {b.text.split("\n")[0]: b for b in doc.pages[0].blocks}

    headline = blocks["Quarterly Report"].dominant_style()
    assert headline.size == pytest.approx(24, abs=0.5)
    assert headline.bold, "the 24pt headline must be detected as bold"

    body = [b for b in doc.pages[0].blocks if "engineering team" in b.text][0]
    assert body.dominant_style().size == pytest.approx(11, abs=0.5)
    assert not body.dominant_style().bold


def test_pdf_detects_underline_from_drawn_line(sample_pdf):
    """PDFs have no underline attribute - it must be inferred from the thin
    rule drawn under the text."""
    doc = extract_pdf(sample_pdf, QAReport())
    subtitle = [b for b in doc.pages[0].blocks if "Department" in b.text][0]
    assert any(s.underline for s in subtitle.spans)
    assert any(d.kind == "underline" for d in doc.pages[0].drawings)


def test_pdf_extracts_image_with_position(sample_pdf):
    doc = extract_pdf(sample_pdf, QAReport())
    images = doc.pages[0].images
    assert len(images) == 1
    image = images[0]
    assert image.data, "raw image bytes must be captured"
    assert image.bbox.x0 == pytest.approx(400, abs=2)
    assert image.directional, "a wide, short graphic is an arrow candidate"


def test_pdf_colors_carried_over(sample_pdf):
    doc = extract_pdf(sample_pdf, QAReport())
    subtitle = [b for b in doc.pages[0].blocks if "Department" in b.text][0]
    r, g, b = subtitle.dominant_style().color
    assert b > r and b > g, "the blue subtitle must keep its colour"


def test_docx_extracts_runs_and_styles(sample_docx):
    doc = extract_docx(sample_docx, QAReport())
    assert doc.kind == "docx"
    texts = [p.text for p in doc.paragraphs]
    assert any("Quarterly Report" in t for t in texts)

    headline = [p for p in doc.paragraphs if p.text == "Quarterly Report"][0]
    run = headline.runs[0]
    assert run.bold is True
    assert run.font_size == pytest.approx(24, abs=0.5)


def test_docx_underline_and_color(sample_docx):
    doc = extract_docx(sample_docx, QAReport())
    subtitle = [p for p in doc.paragraphs if "Department" in p.text][0]
    run = subtitle.runs[0]
    assert run.underline is True
    assert run.color == (0x33, 0x33, 0x99)


def test_docx_includes_table_cells(sample_docx):
    doc = extract_docx(sample_docx, QAReport())
    texts = [p.text for p in doc.paragraphs]
    assert "Metric" in texts and "Uptime" in texts
    assert any(p.location == "table" for p in doc.paragraphs)


def test_unsupported_type_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract(str(path), QAReport())


def test_repeated_drawings_are_collapsed():
    """A rule drawn several times over must reach the page once.

    Producers emit the same rule per stroke pass, and bevel one out of two or
    three greys stacked on the same line. The copies coincide in the source so
    they are invisible there, but each is redrawn on rebuild and a mirror
    pulls them apart - one hairline becomes a stack of them.
    """
    from app.core.extract import _dedupe_drawings
    from app.core.models import BBox, DrawingElement

    def rule(y, fill):
        return DrawingElement(bbox=BBox(28, y, 581, y + 1.5), kind="path",
                              fill=fill)

    # The three passes of one bevelled rule.
    bevelled = [rule(115.0, (151, 151, 151)), rule(115.4, (170, 170, 170)),
                rule(116.0, (85, 85, 85))]
    assert len(_dedupe_drawings(bevelled)) == 1

    # Rules at genuinely different places all survive.
    separate = [rule(115, (0, 0, 0)), rule(201, (0, 0, 0)),
                rule(262, (0, 0, 0))]
    assert len(_dedupe_drawings(separate)) == 3

    # A filled shape of real size is artwork: differently coloured copies are
    # part of the design and are all kept.
    panels = [DrawingElement(bbox=BBox(28, 300, 200, 380), kind="path",
                             fill=(c, c, c)) for c in (10, 120, 250)]
    assert len(_dedupe_drawings(panels)) == 3

    # An exact repeat is dropped whatever its size.
    same = [DrawingElement(bbox=BBox(28, 300, 200, 380), kind="path",
                           fill=(10, 10, 10)) for _ in range(3)]
    assert len(_dedupe_drawings(same)) == 1

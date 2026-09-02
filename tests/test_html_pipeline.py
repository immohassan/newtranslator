"""The reflowing HTML pipeline.

The coordinate rebuild redraws each block where it was in the source, so a
translation that grows pushes nothing down and the error accumulates into
overlap toward the bottom of a dense page. This pipeline reads the page's
structure instead of its coordinates and lets a browser lay it out, which
makes reflow, mirroring, bidi and Arabic shaping someone else's problem.
"""
import os
import re

import fitz
import pytest

from app.core.extract import extract_pdf
from app.core.html_pipeline import (
    DocBlock,
    _is_centred,
    _section_rules,
    Fragment,
    _dir_for,
    build_html,
    extract_structure,
    is_available,
    run_html_pipeline,
    suits_html_pipeline,
)
from app.core.models import BBox
from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.qa import QAReport
from app.core.translate import TranslationProvider
import app.core.translate as translate_mod

RESUME = "storage/ca0ecb41492345f8b3647bbb59be0033/input.pdf"
# A CV whose header is set beside a portrait photograph.
PHOTO_CV = "storage/f919949baa8c426fb4549be8a1c95f6a/input.pdf"
needs_photo_cv = pytest.mark.skipif(not os.path.exists(PHOTO_CV),
                                    reason="sample not present")
needs_resume = pytest.mark.skipif(not os.path.exists(RESUME),
                                  reason="sample resume not present")
needs_browser = pytest.mark.skipif(not is_available(),
                                   reason="no headless browser available")


class _Expanding(TranslationProvider):
    """Arabic stand-in about a third longer than its input.

    Length is what matters here: the failure this pipeline exists to fix only
    appears when the translation needs more room than the original.
    """

    name = "expanding"

    def translate_batch(self, texts, direction):
        out = []
        for text in texts:
            stripped = text.strip()
            if not stripped:
                out.append(text)
                continue
            # Dates and other bare numerals come back unchanged, as a real
            # translator leaves them - the layout tests need them to survive
            # so the side they land on can be checked.
            if re.fullmatch(r"[\W\d\s\u200b]+", stripped) or \
                    re.search(r"(19|20)\d\d", stripped):
                out.append(text)
                continue
            words = max(1, int(len(stripped.split()) * 1.3))
            out.append(" ".join(["نصعربي"] * words))
        return out


def _overlaps(path: str) -> int:
    """Lines whose drawn text actually collides, across all pages.

    Measured on *line* boxes rather than block boxes. A flex row nests its
    halves inside a container, so the container and its own children report as
    overlapping blocks while nothing is drawn on top of anything - comparing
    blocks would count that as a collision and never reach zero.
    """
    total = 0
    with fitz.open(path) as pdf:
        for page in pdf:
            lines = []
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    if text:
                        lines.append(fitz.Rect(line["bbox"]))
            for i, box_a in enumerate(lines):
                for box_b in lines[i + 1:]:
                    overlap = box_a & box_b
                    if overlap.is_empty:
                        continue
                    # Consecutive lines of one paragraph touch by a couple of
                    # points: a line box spans the font's full ascent and
                    # descent, which reaches into the row beside it. That is
                    # ordinary typography, not text drawn over text, so the
                    # test is on how deep the two actually interleave.
                    depth = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
                    shorter = min(box_a.height, box_b.height)
                    if shorter > 0 and depth / shorter > 0.5:
                        total += 1
    return total


# -- structure detection ---------------------------------------------------

@needs_resume
def test_structure_finds_every_block_kind():
    """The resume exercises headings, prose, bullets and title/date rows."""
    doc = extract_pdf(RESUME, QAReport())
    kinds = {b.kind for page in doc.pages
             for b in extract_structure(page, QAReport())}
    assert {"heading", "paragraph", "bullet", "two_sided"} <= kinds


@needs_resume
def test_two_sided_rows_pair_title_with_date():
    """"Job Title .... Date" must become one row, not two stray lines.

    The date is set on its own line, right-aligned, and the title is still an
    open paragraph run when it is reached - so the pairing has to look at the
    run being built, not only at the blocks already closed.
    """
    doc = extract_pdf(RESUME, QAReport())
    rows = [b for page in doc.pages
            for b in extract_structure(page, QAReport())
            if b.kind == "two_sided"]
    assert len(rows) >= 4, "the resume has several dated entries"
    for row in rows:
        assert row.fragments and row.right, "a row needs both halves"

    # The trailing half is whatever the producer set flush right on that
    # baseline - usually a date, but "Lahore, Pakistan" sits there too, so the
    # dates are looked for across the set rather than demanded of every row.
    trailing = [" ".join(f.text for f in row.right) for row in rows]
    dated = [t for t in trailing if re.search(r"(19|20)\d\d", t)]
    assert len(dated) >= 4, f"expected several dated rows, got {trailing}"


@needs_resume
def test_bullets_are_detected_and_their_markers_removed():
    """The marker becomes list structure, not text inside the item."""
    doc = extract_pdf(RESUME, QAReport())
    bullets = [b for page in doc.pages
               for b in extract_structure(page, QAReport())
               if b.kind == "bullet"]
    assert bullets
    for block in bullets:
        assert not block.text.lstrip().startswith(("•", "●", "-", "*"))


# -- per-element direction -------------------------------------------------

@pytest.mark.parametrize("text", [
    "immohassan06@gmail.com",
    "linkedin.com/in/hassan",
    "+92-341-6903128",
    "https://example.com/x",
    "LinkedIn",
])
def test_latin_content_is_marked_ltr_on_an_rtl_page(text):
    """One attribute replaces the old per-span coordinate reasoning."""
    assert _dir_for(text, rtl_page=True) == "ltr"


def test_arabic_content_inherits_the_page_direction():
    assert _dir_for("نص عربي طويل هنا", rtl_page=True) == ""


def test_arabic_inside_an_ltr_page_is_marked_rtl():
    assert _dir_for("نص عربي", rtl_page=False) == "rtl"


# -- HTML generation -------------------------------------------------------

def _html_for(blocks, direction="en2ar"):
    return build_html([blocks], direction, BBox(0, 0, 612, 792), QAReport())


def test_semantic_elements_are_emitted():
    """Real elements, not absolutely-positioned divs at copied coordinates."""
    blocks = [
        DocBlock(kind="heading", fragments=[Fragment("Experience")], level=1),
        DocBlock(kind="paragraph", fragments=[Fragment("Some prose here.")]),
        DocBlock(kind="bullet", fragments=[Fragment("First item")]),
        DocBlock(kind="bullet", fragments=[Fragment("Second item")]),
    ]
    markup = _html_for(blocks)
    assert "<h1>" in markup
    assert "<p>" in markup
    assert markup.count("<li>") == 2
    assert "<ul>" in markup and "</ul>" in markup
    assert "position:absolute" not in markup


def test_two_sided_rows_use_one_flex_rule():
    """`justify-content: space-between` is what makes the row mirror."""
    blocks = [DocBlock(kind="two_sided",
                       fragments=[Fragment("Job Title")],
                       right=[Fragment("April 2023")])]
    markup = _html_for(blocks)
    assert "justify-content: space-between" in markup
    # A two-cell row also carries "pair", which pins the trailing date to the
    # far margin and keeps it on one line.
    assert 'class="two-sided pair"' in markup


def test_a_row_of_several_cells_keeps_every_cell_in_the_row():
    """A banner of labelled fields is one row, not a row plus loose text.

    The header of a CV sets three or more cells across a single baseline.
    Grouping only the first and last left the cells between them to be emitted
    as paragraphs under the row.
    """
    blocks = [DocBlock(kind="two_sided",
                       fragments=[Fragment("Place of birth: Lahore")],
                       extra=[[Fragment("Nationality: Pakistani")],
                              [Fragment("Gender: Female")]],
                       right=[Fragment("Phone number:")])]
    markup = _html_for(blocks)
    for cell in ("Place of birth: Lahore", "Nationality: Pakistani",
                 "Gender: Female", "Phone number:"):
        assert cell in markup
    # One row, and it may wrap - a four-cell banner cannot be held on one line
    # the way a "Job Title .... Date" pair is.
    assert markup.count('class="two-sided') == 1
    assert 'class="two-sided row-wrap"' in markup
    assert markup.index("Place of birth") < markup.index("Nationality") \
        < markup.index("Gender") < markup.index("Phone number")


def test_rtl_is_set_on_the_document_not_computed_per_block():
    markup = _html_for([DocBlock(kind="paragraph",
                                 fragments=[Fragment("نص")])])
    assert 'dir="rtl"' in markup
    assert "line-height: 1.5" in markup


def test_ltr_run_inside_an_rtl_document_is_isolated():
    blocks = [DocBlock(kind="paragraph",
                       fragments=[Fragment("راسلنا "),
                                  Fragment("info@example.com")])]
    markup = _html_for(blocks)
    assert 'dir="ltr"' in markup


def test_text_is_passed_unshaped():
    """The browser shapes Arabic; pre-shaping here would double-shape it."""
    logical = "نص عربي للاختبار"
    markup = _html_for([DocBlock(kind="paragraph",
                                 fragments=[Fragment(logical)])])
    assert logical in markup
    # No presentation forms may reach the HTML.
    assert not any("ﭐ" <= c <= "﻿" for c in markup)


def test_markup_is_escaped():
    markup = _html_for([DocBlock(kind="paragraph",
                                 fragments=[Fragment("a < b & c")])])
    assert "&lt;" in markup and "&amp;" in markup


# -- rendering -------------------------------------------------------------

@needs_resume
@needs_browser
def test_no_overlaps_even_when_the_translation_grows(tmp_path):
    """The failure this pipeline exists to fix.

    A block that needs more room pushes what follows it down, so growth cannot
    accumulate into the pile-up the coordinate path produces.
    """
    translate_mod.set_provider(_Expanding())
    out = str(tmp_path / "out.pdf")
    doc = extract_pdf(RESUME, QAReport())
    run_html_pipeline(doc, out, "en2ar", QAReport())
    assert _overlaps(out) == 0


@needs_resume
@needs_browser
def test_longer_text_flows_onto_another_page(tmp_path):
    """It must reflow, not be crushed onto one page or clipped."""
    translate_mod.set_provider(_Expanding())
    out = str(tmp_path / "out.pdf")
    doc = extract_pdf(RESUME, QAReport())
    qa = QAReport()
    run_html_pipeline(doc, out, "en2ar", qa)

    with fitz.open(RESUME) as src, fitz.open(out) as dst:
        assert dst.page_count >= src.page_count
    assert any(e.category == "layout_review" for e in qa.entries)


@needs_resume
@needs_browser
def test_the_date_side_flips_with_the_page_direction(tmp_path):
    """No per-row logic beyond the flex rule."""
    import re

    translate_mod.set_provider(_Expanding())
    sides = {}
    for direction in ("en2ar", "ar2en"):
        out = str(tmp_path / f"{direction}.pdf")
        doc = extract_pdf(RESUME, QAReport())
        run_html_pipeline(doc, out, direction, QAReport())
        with fitz.open(out) as pdf:
            page = pdf[0]
            middle = page.rect.width / 2
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        if re.search(r"20\d\d", span["text"]):
                            sides.setdefault(
                                direction,
                                "left" if span["bbox"][0] < middle else "right")
                            break
    assert sides.get("en2ar") == "left", sides
    assert sides.get("ar2en") == "right", sides


@needs_resume
@needs_browser
def test_pipeline_option_selects_the_html_engine(tmp_path):
    translate_mod.set_provider(_Expanding())
    out = str(tmp_path / "out.pdf")
    qa = QAReport()
    run_pipeline(RESUME, out,
                 TranslationOptions(direction="en2ar", html_engine=True), qa)
    assert any(e.detail.get("engine") == "html"
               for e in qa.entries if e.category == "summary")
    assert _overlaps(out) == 0


@needs_resume
def test_the_html_engine_is_the_default():
    """Reflow is the default; the coordinate path is the explicit opt-out.

    Leaving the coordinate path in charge means dense documents overlap by
    default, which is the failure this pipeline exists to remove.
    """
    assert TranslationOptions().html_engine is True


@needs_resume
def test_the_coordinate_path_is_still_reachable(tmp_path):
    """The old engine stays available for anything that needs it."""
    from tests.fake_provider import FakeProvider

    translate_mod.set_provider(FakeProvider())
    out = str(tmp_path / "out.pdf")
    qa = QAReport()
    run_pipeline(RESUME, out,
                 TranslationOptions(direction="en2ar", html_engine=False), qa)
    engines = [e.detail.get("engine") for e in qa.entries
               if e.category == "summary"]
    assert "html" not in engines


@needs_resume
def test_a_missing_browser_falls_back_and_says_so(tmp_path, monkeypatch):
    """Without a browser the job still completes, and the user is told why.

    Falling back silently would hand back an overlapping document with no
    indication that the reflowing path was unavailable.
    """
    from tests.fake_provider import FakeProvider
    import app.core.html_pipeline as html_pipeline

    monkeypatch.setattr(html_pipeline, "is_available", lambda: False)
    translate_mod.set_provider(FakeProvider())
    out = str(tmp_path / "out.pdf")
    qa = QAReport()
    run_pipeline(RESUME, out,
                 TranslationOptions(direction="en2ar", html_engine=True), qa)

    assert os.path.exists(out), "the job must still produce a document"
    assert any("no headless browser" in e.message.lower() for e in qa.entries)


# -- section rules ---------------------------------------------------------

@needs_resume
def test_section_rules_are_kept():
    """The dividers under each section heading are part of the document."""
    doc = extract_pdf(RESUME, QAReport())
    rules = [b for page in doc.pages
             for b in extract_structure(page, QAReport())
             if b.kind == "rule"]
    assert len(rules) >= 6, "the resume rules one line under each section"


def test_a_wide_thin_line_is_a_section_rule():
    from app.core.models import DrawingElement, Page as P

    page = P(number=0, width=612, height=792,
             drawings=[DrawingElement(bbox=BBox(54, 113, 511, 113),
                                      kind="underline")])
    assert len(_section_rules(page, 54, 533)) == 1


def test_a_short_rule_belongs_to_its_phrase_not_the_page():
    """An underline under one phrase travels with that text, not as a divider."""
    from app.core.models import DrawingElement, Page as P

    page = P(number=0, width=612, height=792,
             drawings=[DrawingElement(bbox=BBox(90, 720, 190, 720),
                                      kind="underline")])
    assert _section_rules(page, 54, 533) == []


def test_a_filled_box_is_not_a_rule():
    from app.core.models import DrawingElement, Page as P

    page = P(number=0, width=612, height=792,
             drawings=[DrawingElement(bbox=BBox(0, 0, 595, 842), kind="rect")])
    assert _section_rules(page, 54, 533) == []


def test_rules_are_emitted_as_hr_elements():
    markup = _html_for([DocBlock(kind="rule")])
    assert "<hr" in markup and "section-rule" in markup


# -- centred text ----------------------------------------------------------

@needs_resume
def test_centred_header_lines_are_detected():
    """The name and contact block are centred and must stay centred."""
    doc = extract_pdf(RESUME, QAReport())
    centred = [b for b in extract_structure(doc.pages[0], QAReport())
               if b.centred and b.text.strip()]
    assert centred, "the centred header must be recognised"
    assert any("HASSAN" in b.text.upper() for b in centred)


def test_centring_needs_indent_at_both_ends():
    """A paragraph line that merely falls short of the right edge is not centred.

    Judged on the indent at each end rather than on the midpoint, because a
    full-width line's midpoint sits near the centre too.
    """
    from app.core.models import BBox as B, Line, Span

    def line(x0, x1):
        span = Span("text", "helv", 11.0, (0, 0, 0), False, False, False,
                    B(x0, 100, x1, 114), origin=(x0, 114))
        return Line(spans=[span], bbox=B(x0, 100, x1, 114))

    assert _is_centred(line(200, 400), 54, 533)      # pulled in both sides
    assert not _is_centred(line(54, 400), 54, 533)   # flush left
    assert not _is_centred(line(54, 533), 54, 533)   # full width

    # A middle cell of a row is indented at both ends by the cells either side
    # of it, which is the centring test exactly. Sharing a baseline with
    # another line settles it: this is a cell, not a centred line.
    middle = line(200, 400)
    neighbours = [line(54, 190), middle, line(410, 533)]
    assert not _is_centred(middle, 54, 533, neighbours)
    assert _is_centred(middle, 54, 533, [middle])


def test_side_by_side_text_is_not_mistaken_for_a_table():
    """PyMuPDF infers a table from alignment; a caption is not one.

    A court caption sets the parties down one side and the document's labels
    down the other. Detected as a table it gains borders the source never had
    and its two halves are combed together row by row.
    """
    from app.core.html_pipeline import _is_really_a_grid

    caption = [["Plaintiff,", "STATEMENT"],
               ["", "OF NET WORTH"],
               ["", "DATED:"],
               ["- against -", ""],
               ["", "Index No."],
               ["Defendant.", ""]]
    assert not _is_really_a_grid(caption)

    # A real table relates the cells along each row, and keeps its grid.
    grid = [["(a)", "Plaintiff's date of birth:", "1983-10-07"],
            ["(b)", "Defendant's date of birth:", "1982-09-30"],
            ["(c)", "Date married:", "2021-08-11"]]
    assert _is_really_a_grid(grid)


def test_a_caption_keeps_its_two_columns_whole():
    """The two halves of a caption interleave; read in order they comb together.

    Each column has to be gathered whole, or the output reads "Index No. Date
    Action Commenced: Defendant." - three unrelated lines run together.
    """
    from app.core.models import BBox as B, Line, Span
    from app.core.html_pipeline import _caption_region

    def line(text, x0, x1, y):
        span = Span(text, "helv", 11.0, (0, 0, 0), False, False, False,
                    B(x0, y, x1, y + 12), origin=(x0, y + 12))
        return (None, Line(spans=[span], bbox=B(x0, y, x1, y + 12)))

    lines = [
        line("Plaintiff,", 72, 117, 143),
        line("STATEMENT", 412, 488, 143),
        line("OF NET WORTH", 412, 509, 157),
        line("DATED:", 412, 460, 171),
        line("- against -", 75, 183, 186),
        line("Index No.", 412, 468, 200),
        line("Defendant.", 72, 128, 243),
        # A page footer sits in the same margin and must stay out of it.
        line("Page 1", 516, 540, 753),
    ]
    region = _caption_region(lines, 72, 540)
    assert region, "the caption must be recognised"
    left, right = region
    assert "Page 1" not in " ".join(s.text for l in right for s in l.spans)
    joined = " ".join(s.text for l in right for s in l.spans)
    assert "STATEMENT" in joined and "Index No." in joined


def test_a_page_of_dated_rows_is_not_a_caption():
    """A resume's rows pair off on shared baselines - that is not a caption."""
    from app.core.models import BBox as B, Line, Span
    from app.core.html_pipeline import _caption_region

    def line(text, x0, x1, y):
        span = Span(text, "helv", 11.0, (0, 0, 0), False, False, False,
                    B(x0, y, x1, y + 12), origin=(x0, y + 12))
        return (None, Line(spans=[span], bbox=B(x0, y, x1, y + 12)))

    lines = []
    for i, y in enumerate((100, 140, 180, 220)):
        lines.append(line(f"Job Title {i}", 72, 200, y))
        lines.append(line(f"20{20 + i}", 460, 540, y))
    assert _caption_region(lines, 72, 540) is None


@needs_photo_cv
def test_the_page_is_read_in_the_order_it_is_seen():
    """Blocks follow their place on the page, not the producer's write order.

    A PDF's content stream carries no guarantee of sequence, and a
    template-built CV writes its section titles last. Read as stored, every
    heading piled up at the foot of the page, stranded from the content it
    introduces.
    """
    doc = extract_pdf(PHOTO_CV, QAReport())
    for page in doc.pages:
        blocks = extract_structure(page, QAReport())
        kinds = [b.kind for b in blocks]
        if kinds.count("heading") < 2:
            continue
        # No page may end with a run of headings carrying no content at all.
        tail = kinds[kinds.index("heading", len(kinds) // 2):]
        assert any(k != "heading" for k in tail), \
            f"page {page.number + 1} ends in a stack of orphaned headings"

    # The section titles of the last page each introduce something.
    blocks = extract_structure(doc.pages[-1], QAReport())
    titles = [i for i, b in enumerate(blocks)
              if b.kind == "heading" and b.text.strip().isupper()]
    assert titles, "the fixture has upper-case section titles"
    for i in titles:
        assert i < len(blocks) - 1, "a section title needs content after it"


@needs_photo_cv
def test_an_image_is_placed_where_it_sits_on_the_page():
    """Images follow the text they belong to, not the end of the page.

    Appended after every block, a CV's header portrait landed at the foot of
    the page's content and reflow then carried it onto the following page,
    where it sat in the middle of an unrelated section.
    """
    doc = extract_pdf(PHOTO_CV, QAReport())
    page = doc.pages[0]
    assert any(im.data for im in page.images), "the fixture carries a photo"
    blocks = extract_structure(page, QAReport())
    kinds = [b.kind for b in blocks]
    first_image = kinds.index("image")
    assert first_image < len(kinds) - 1, \
        "a header image must not be emitted after all of the page's text"


def test_an_image_with_text_beside_it_floats():
    """A portrait in the corner keeps the header alongside it.

    Set as a block of its own it splits the header in two and pushes the rest
    of the page down; floated, the text wraps beside it as the source had it.
    """
    from app.core.models import BBox as B, ImageElement, Line, Span
    from app.core.html_pipeline import _float_side

    def line(x0, x1, y):
        span = Span("text", "helv", 11.0, (0, 0, 0), False, False, False,
                    B(x0, y, x1, y + 12), origin=(x0, y + 12))
        return Line(spans=[span], bbox=B(x0, y, x1, y + 12))

    corner = ImageElement(bbox=B(16, 14, 102, 100), data=b"x", ext="png")
    beside = [line(116, 400, 13), line(116, 400, 52), line(116, 400, 75)]
    assert _float_side(corner, beside) == "start"

    # An image with text on both sides was placed inline by the producer;
    # floating it would reorder the page.
    both = beside + [line(10, 14, 52)]
    assert _float_side(corner, both) == ""

    # A full-width figure has nothing beside it and stays a block.
    figure = ImageElement(bbox=B(72, 200, 520, 400), data=b"x", ext="png")
    assert _float_side(figure, beside) == ""


def test_centred_blocks_carry_the_class():
    markup = _html_for([DocBlock(kind="paragraph",
                                 fragments=[Fragment("Name")], centred=True)])
    assert 'class="centred"' in markup
    assert "text-align: center" in markup


# -- pagination ------------------------------------------------------------

@needs_resume
@needs_browser
def test_pages_fill_before_breaking(tmp_path):
    """Source page boundaries must not be reproduced as forced breaks.

    They record where the *original* ran out of room, which is no longer where
    the translation does - forcing a break there leaves a page half empty and
    pushes its remainder onto the next.
    """
    translate_mod.set_provider(_Expanding())
    out = str(tmp_path / "out.pdf")
    doc = extract_pdf(RESUME, QAReport())
    run_html_pipeline(doc, out, "en2ar", QAReport())

    with fitz.open(out) as pdf:
        fills = []
        for page in pdf:
            bottoms = [b["bbox"][3] for b in page.get_text("dict")["blocks"]
                       if b.get("type") == 0
                       and "".join(s["text"] for l in b.get("lines", [])
                                   for s in l.get("spans", [])).strip()]
            fills.append(max(bottoms) if bottoms else 0)
            height = page.rect.height

        # Every page but the last must be substantially full.
        for index, fill in enumerate(fills[:-1]):
            assert fill > height * 0.75, (
                f"page {index + 1} broke early at y={fill:.0f} of {height:.0f}"
            )


def test_no_forced_break_markup_is_emitted():
    markup = build_html([[DocBlock(kind="paragraph",
                                   fragments=[Fragment("one")])],
                         [DocBlock(kind="paragraph",
                                   fragments=[Fragment("two")])]],
                        "en2ar", BBox(0, 0, 612, 792), QAReport())
    assert "break-before: page" not in markup


# -- headings must not be swallowed by the body ----------------------------

@needs_resume
def test_every_section_heading_is_recognised():
    """A section title set slightly bold must not fold into the paragraph.

    Requiring *every* span of the line to be bold let a single trailing space
    or zero-width joiner - carried as its own unbold span - veto the heading,
    so "Education" ended up as the tail of the summary paragraph.
    """
    doc = extract_pdf(RESUME, QAReport())
    headings = [b.text.strip().replace("\u200b", "")
                for page in doc.pages
                for b in extract_structure(page, QAReport())
                if b.kind == "heading"]
    for wanted in ("Professional Summary", "Education",
                   "Professional Experience"):
        assert any(wanted in h for h in headings), \
            f"{wanted!r} was not read as a heading: {headings}"


def test_a_heading_with_an_unbold_trailing_space_still_counts():
    """Weighted by characters, so one stray span cannot veto the line."""
    from app.core.models import BBox as B, Line, Span
    from app.core.html_pipeline import _classify

    bold = Span("Education", "Roboto-Bold", 12.0, (0, 0, 0), True, False,
                False, B(54, 233, 130, 247), origin=(54, 247))
    trailing = Span(" ", "Roboto", 12.0, (0, 0, 0), False, False, False,
                    B(130, 233, 133, 247), origin=(130, 247))
    line = Line(spans=[bold, trailing], bbox=B(54, 233, 133, 247))
    kind, _ = _classify(line, 11.0, 612, 54, 533)
    assert kind == "heading"


def test_mostly_unbold_text_is_not_a_heading():
    from app.core.models import BBox as B, Line, Span
    from app.core.html_pipeline import _classify

    plain = Span("a long run of ordinary body text", "Roboto", 12.0,
                 (0, 0, 0), False, False, False, B(54, 233, 400, 247),
                 origin=(54, 247))
    emphasis = Span("bit", "Roboto-Bold", 12.0, (0, 0, 0), True, False, False,
                    B(400, 233, 420, 247), origin=(400, 247))
    line = Line(spans=[plain, emphasis], bbox=B(54, 233, 420, 247))
    kind, _ = _classify(line, 11.0, 612, 54, 533)
    assert kind != "heading"


# -- rows split across two source lines ------------------------------------

@needs_resume
def test_rows_set_as_two_lines_are_paired():
    """"Institute ... Lahore, Pakistan" is two lines sharing one baseline.

    PyMuPDF reports them separately. Left unpaired they become two paragraphs
    and the right-hand half wraps into a narrow column of its own.
    """
    doc = extract_pdf(RESUME, QAReport())
    rows = [b for page in doc.pages
            for b in extract_structure(page, QAReport())
            if b.kind == "two_sided"]
    trailing = [" ".join(f.text for f in row.right).strip() for row in rows]
    assert any("Lahore" in t for t in trailing), trailing
    assert any("April 2023" in t for t in trailing), trailing


@needs_resume
def test_a_trailing_half_never_becomes_its_own_paragraph():
    """The date must not survive as a standalone block."""
    doc = extract_pdf(RESUME, QAReport())
    paragraphs = [b.text.strip() for page in doc.pages
                  for b in extract_structure(page, QAReport())
                  if b.kind == "paragraph"]
    assert not any(p.startswith("April 2023") for p in paragraphs), paragraphs


# -- rules sit under their heading -----------------------------------------

@needs_resume
@needs_browser
def test_a_rule_follows_its_heading(tmp_path):
    """In the source the divider is drawn below the section title."""
    translate_mod.set_provider(_Expanding())
    out = str(tmp_path / "out.pdf")
    doc = extract_pdf(RESUME, QAReport())
    run_html_pipeline(doc, out, "en2ar", QAReport())

    with fitz.open(out) as pdf:
        page = pdf[0]
        rules = sorted(d["rect"].y0 for d in page.get_drawings()
                       if d["rect"].height < 3 and d["rect"].width > 200)
        texts = sorted(line["bbox"][1]
                       for block in page.get_text("dict")["blocks"]
                       if block.get("type") == 0
                       for line in block.get("lines", []))
        assert rules, "the page must carry section rules"
        # Every rule has text above it and text below it - it divides, rather
        # than sitting at the very top of the page.
        for y in rules:
            assert any(t < y for t in texts), f"rule at {y:.0f} has nothing above"
            assert any(t > y for t in texts), f"rule at {y:.0f} has nothing below"


# -- choosing an engine per document ---------------------------------------

FLYER = "storage/99d02f6d29574e8f90123aaac2832ec8/input.pdf"
PORTFOLIO = "storage/445191b1fa0b452f87ca42413fb79698/input.pdf"


@pytest.mark.skipif(not os.path.exists(FLYER), reason="sample not present")
def test_a_designed_page_is_not_reflowed():
    """A flyer's panels, logos and icons *are* the document.

    Re-flowing it as semantic HTML keeps the words and discards the design, so
    it is redrawn in place instead. It is also the case that needs reflow
    least: there is too little text for growth to accumulate.
    """
    doc = extract_pdf(FLYER, QAReport())
    assert not suits_html_pipeline(doc)


@needs_resume
def test_a_text_document_is_reflowed():
    doc = extract_pdf(RESUME, QAReport())
    assert suits_html_pipeline(doc)


@pytest.mark.skipif(not os.path.exists(PORTFOLIO), reason="sample not present")
def test_a_document_with_images_but_little_artwork_is_reflowed():
    """Photographs are content in the flow, not page furniture."""
    doc = extract_pdf(PORTFOLIO, QAReport())
    assert suits_html_pipeline(doc)


def test_an_empty_document_is_not_reflowed():
    from app.core.models import Document as D

    assert not suits_html_pipeline(D(source_path="x", kind="pdf"))


@pytest.mark.skipif(not os.path.exists(FLYER), reason="sample not present")
def test_the_flyer_keeps_its_artwork_through_the_pipeline(tmp_path):
    """End to end: routing must actually reach the coordinate engine."""
    from tests.fake_provider import FakeProvider

    translate_mod.set_provider(FakeProvider())
    out = str(tmp_path / "out.pdf")
    qa = QAReport()
    run_pipeline(FLYER, out, TranslationOptions(direction="en2ar"), qa)

    with fitz.open(FLYER) as src, fitz.open(out) as dst:
        before = len(src[0].get_drawings())
        after = len(dst[0].get_drawings())
        assert dst.page_count == src.page_count, "a flyer must stay one page"
    assert after > before * 0.9, f"artwork was lost: {before} -> {after}"
    assert any(e.detail.get("engine") == "coordinate" for e in qa.entries)


@needs_resume
@needs_browser
def test_the_resume_still_reaches_the_html_engine(tmp_path):
    translate_mod.set_provider(_Expanding())
    out = str(tmp_path / "out.pdf")
    qa = QAReport()
    run_pipeline(RESUME, out, TranslationOptions(direction="en2ar"), qa)
    assert any(e.detail.get("engine") == "html"
               for e in qa.entries if e.category == "summary")
    assert _overlaps(out) == 0


# -- tables and numbered lists ---------------------------------------------

MIGRATION = "storage/df0fbf7346c24b1cbfe7386c53f3fc7b/input.pdf"
needs_migration = pytest.mark.skipif(not os.path.exists(MIGRATION),
                                     reason="sample not present")


def _structure(path):
    doc = extract_pdf(path, QAReport())
    with fitz.open(path) as source:
        return [extract_structure(page, QAReport(), source[page.number])
                for page in doc.pages]


@needs_migration
def test_tables_are_kept_as_tables():
    """A grid is the one structure that cannot survive being flattened.

    Its cells only mean anything beside the cells they sit with, so the rows
    are lifted out whole rather than becoming a run of loose paragraphs.
    """
    tables = [b for page in _structure(MIGRATION) for b in page
              if b.kind == "table"]
    assert tables, "the document's tables must be recognised"
    biggest = max(tables, key=lambda t: len(t.rows))
    assert len(biggest.rows) >= 5
    assert all(len(row) == 2 for row in biggest.rows)


@needs_migration
def test_a_table_does_not_also_appear_as_paragraphs():
    """Cell text is withheld from the rest of the page once lifted out.

    Checked on a string that appears *only* inside the table. "Logic Mapping"
    would not do: the document also lists it as a numbered step in the body,
    so finding it outside the table proves nothing.
    """
    pages = _structure(MIGRATION)
    loose = " ".join(b.text for page in pages for b in page
                     if b.kind in ("paragraph", "bullet"))
    assert "6–7 hours" not in loose, "a table cell leaked into the body text"


@needs_migration
def test_table_rules_are_not_emitted_as_section_dividers():
    """A table's own borders belong to the table, not to the page.

    Left in, they came out as loose horizontal rules between the rows.
    """
    pages = _structure(MIGRATION)
    rules = [b for page in pages for b in page if b.kind == "rule"]
    assert not rules, f"{len(rules)} table border(s) leaked as section rules"


def test_tables_render_as_real_table_elements():
    block = DocBlock(kind="table", header=True,
                     rows=[["Task", "Estimated Time"],
                           ["Logic Mapping", "1-2 hours"]])
    markup = _html_for([block])
    assert "<table>" in markup and "</table>" in markup
    assert "<thead>" in markup and "<th>" in markup
    assert markup.count("<td>") == 2
    assert "border-collapse: collapse" in markup


@needs_migration
@needs_browser
def test_table_cells_are_translated(tmp_path):
    """Cells are plain strings, so they need writing back after the batch."""
    translate_mod.set_provider(_Expanding())
    doc = extract_pdf(MIGRATION, QAReport())
    out = str(tmp_path / "out.pdf")
    run_html_pipeline(doc, out, "en2ar", QAReport())

    with fitz.open(out) as pdf:
        text = "".join(page.get_text() for page in pdf)
    assert "Logic Mapping" not in text, "a cell was left untranslated"


# -- numbered lists --------------------------------------------------------

@needs_migration
def test_numbered_items_are_marked_ordered():
    items = [b for page in _structure(MIGRATION) for b in page
             if b.kind == "bullet" and b.ordered]
    # The document numbers six migration steps. The separator after each
    # number is a zero-width space, not an ordinary one, which is what used to
    # hide them from the ordered-list test.
    assert len(items) >= 6, f"expected the numbered steps, got {len(items)}"
    assert not any(b.text.strip()[:2].rstrip(".").isdigit() for b in items), \
        "the source number must be stripped - the list renumbers itself"


def test_ordered_items_render_as_an_ol():
    """The browser numbers an <ol>, so the source marker is dropped."""
    blocks = [DocBlock(kind="bullet", fragments=[Fragment("First")],
                       ordered=True),
              DocBlock(kind="bullet", fragments=[Fragment("Second")],
                       ordered=True)]
    markup = _html_for(blocks)
    assert "<ol>" in markup and "</ol>" in markup
    assert "<ul>" not in markup
    assert markup.count("<li>") == 2


def test_a_numbered_run_does_not_continue_into_a_bulleted_one():
    blocks = [DocBlock(kind="bullet", fragments=[Fragment("one")],
                       ordered=True),
              DocBlock(kind="bullet", fragments=[Fragment("dot")])]
    markup = _html_for(blocks)
    assert "</ol>" in markup and "<ul>" in markup


def test_the_source_number_is_not_written_back_into_the_item():
    """Otherwise the item is numbered twice - once by us, once by the list."""
    from app.core.models import BBox as B, Line, Span
    from app.core.html_pipeline import _classify, _fragments

    span = Span("4. Remaining Development", "Roboto", 11.0, (0, 0, 0),
                False, False, False, B(72, 100, 300, 114), origin=(72, 114))
    line = Line(spans=[span], bbox=B(72, 100, 300, 114))
    kind, meta = _classify(line, 11.0, 612, 72, 533)
    assert kind == "bullet" and meta.get("ordered")
    text = "".join(f.text for f in _fragments([span], strip_marker=True))
    assert not text.strip().startswith("4.")


# -- lists written without marker characters -------------------------------

GEAR = "storage/0fbed848b5ee4b1bbd66c2835636b4de/input.pdf"
needs_gear = pytest.mark.skipif(not os.path.exists(GEAR),
                                reason="sample not present")


@needs_gear
def test_entries_set_on_their_own_lines_stay_separate():
    """A list can be written without a single bullet character.

    This document gives every item its own one-line block. Running those
    together makes a paragraph out of what the reader sees as separate
    entries, which is what crushed the gear list into prose.
    """
    pages = _structure(GEAR)
    entries = [b for page in pages for b in page if b.standalone]
    assert len(entries) > 50, f"expected the gear entries, got {len(entries)}"
    joined = [b for page in pages for b in page
              if b.kind == "paragraph" and not b.standalone
              and b.text.count("✅") > 1]
    assert not joined, "separate entries were merged into a paragraph"


@needs_gear
def test_standalone_entries_render_as_list_items():
    pages = _structure(GEAR)
    markup = build_html(pages, "en2ar", BBox(0, 0, 612, 792), QAReport())
    assert markup.count("<li>") > 50
    assert "<ul>" in markup


def test_a_wrapped_paragraph_is_not_turned_into_a_list():
    """Only lines that stand alone in their block become entries.

    A paragraph's lines share a block, so they still join into one <p>.
    """
    from app.core.models import BBox as B, Line, Page as P, Span, TextBlock

    def line(text, y):
        span = Span(text, "Roboto", 11.0, (0, 0, 0), False, False, False,
                    B(72, y, 500, y + 13), origin=(72, y + 13))
        return Line(spans=[span], bbox=B(72, y, 500, y + 13))

    block = TextBlock(lines=[line("first line of the paragraph", 100),
                             line("second line of the paragraph", 116)],
                      bbox=B(72, 100, 500, 129))
    page = P(number=0, width=612, height=792, blocks=[block])
    blocks = extract_structure(page, QAReport())
    assert len([b for b in blocks if b.kind == "paragraph"]) == 1
    assert not any(b.standalone for b in blocks)


# -- headings that are not bold --------------------------------------------

@needs_gear
def test_a_large_unbold_line_is_a_heading():
    """"Hydration & Nutrition" is 15pt Regular against 11pt body.

    Requiring bold folded every section title into the list below it.
    """
    headings = [b.text.strip() for page in _structure(GEAR) for b in page
                if b.kind == "heading"]
    for wanted in ("Hydration & Nutrition", "Clothing System", "Accessories"):
        assert any(wanted in h for h in headings), \
            f"{wanted!r} was not read as a heading"


def test_large_body_text_is_not_mistaken_for_a_heading():
    """A long run stays prose however it is set."""
    from app.core.models import BBox as B, Line, Span
    from app.core.html_pipeline import _classify

    long_text = ("a long sentence of body text that simply happens to be set "
                 "at a larger size than the rest of the page around it.")
    span = Span(long_text, "Roboto", 15.0, (0, 0, 0), False, False, False,
                B(72, 100, 540, 118), origin=(72, 118))
    line = Line(spans=[span], bbox=B(72, 100, 540, 118))
    kind, _ = _classify(line, 11.0, 612, 72, 540)
    assert kind != "heading"


def test_a_large_sentence_is_not_a_heading():
    from app.core.models import BBox as B, Line, Span
    from app.core.html_pipeline import _classify

    span = Span("This is a short sentence.", "Roboto", 15.0, (0, 0, 0),
                False, False, False, B(72, 100, 300, 118), origin=(72, 118))
    line = Line(spans=[span], bbox=B(72, 100, 300, 118))
    kind, _ = _classify(line, 11.0, 612, 72, 540)
    assert kind != "heading"


class _Verbatim(TranslationProvider):
    """Returns its input unchanged.

    Layout tests need distinguishable text: a provider that replaces every
    entry with the same repeated word makes separate list items look identical
    once rendered, which hides exactly the bug these tests are for.
    """

    name = "verbatim"

    def translate_batch(self, texts, direction):
        return list(texts)


@needs_gear
@needs_browser
def test_list_entries_survive_the_whole_pipeline(tmp_path):
    """Checked through `run_pipeline`, not by calling the structure pass.

    The layout preparation that runs before the rebuild - fragment merging,
    paragraph stacking - exists for the coordinate path. It fuses the
    one-line blocks this document uses for its entries, so calling
    `extract_structure` directly passes while a real job still produces
    paragraphs. Only the full path proves the fix.
    """
    translate_mod.set_provider(_Verbatim())
    out = str(tmp_path / "out.pdf")
    run_pipeline(GEAR, out, TranslationOptions(direction="en2ar"), QAReport())

    with fitz.open(out) as pdf:
        page = pdf[0]
        multi = 0
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            text = "".join(s["text"] for l in block.get("lines", [])
                           for s in l.get("spans", []))
            # Two ticked entries in one block means they were run together.
            if text.count("✅") > 1:
                multi += 1
        assert multi == 0, f"{multi} block(s) merged separate list entries"


@needs_gear
@needs_browser
def test_the_html_engine_sees_unmerged_blocks(tmp_path):
    """The engine choice must come before the coordinate-path preparation."""
    translate_mod.set_provider(_Verbatim())
    out = str(tmp_path / "out.pdf")
    qa = QAReport()
    run_pipeline(GEAR, out, TranslationOptions(direction="en2ar"), qa)

    assert any(e.detail.get("engine") == "html"
               for e in qa.entries if e.category == "summary")
    # None of the coordinate-path preparation should have reported work.
    assert not any("joined so each sentence" in e.message for e in qa.entries)


# -- entries opened by a tick or a cross -----------------------------------

AI_DOC = "storage/845152bd39444c968c994f7fbbac42d7/input.pdf"
needs_ai_doc = pytest.mark.skipif(not os.path.exists(AI_DOC),
                                  reason="sample not present")


@pytest.mark.parametrize("text", [
    "✅ Use the exact numbers provided",
    "❌ Recalculate metrics",
    "☑ Checked item",
    "✗ Rejected item",
])
def test_a_tick_or_cross_opens_a_list_item(text):
    """These mark an entry just as a bullet does."""
    from app.core.models import BBox as B, Line, Span
    from app.core.html_pipeline import _classify

    span = Span(text, "Roboto", 11.0, (0, 0, 0), False, False, False,
                B(72, 100, 400, 114), origin=(72, 114))
    line = Line(spans=[span], bbox=B(72, 100, 400, 114))
    kind, meta = _classify(line, 11.0, 612, 72, 533)
    assert kind == "bullet"
    assert meta.get("keep_marker"), "a tick carries meaning and must be kept"


def test_a_tick_marker_is_not_stripped():
    """"✅" and "❌" say "do" and "do not".

    Unlike a bullet, which is decoration the renderer replaces, dropping these
    would change what the line means.
    """
    from app.core.models import BBox as B, Span
    from app.core.html_pipeline import _fragments

    span = Span("✅ Use the exact numbers", "Roboto", 11.0, (0, 0, 0),
                False, False, False, B(72, 100, 400, 114), origin=(72, 114))
    kept = "".join(f.text for f in _fragments([span], strip_marker=False))
    assert kept.strip().startswith("✅")


def test_an_ordinary_bullet_is_still_stripped():
    """The list draws its own marker, so the source one would double up."""
    from app.core.models import BBox as B, Span
    from app.core.html_pipeline import _fragments

    span = Span("• Ordinary item", "Roboto", 11.0, (0, 0, 0), False, False,
                False, B(72, 100, 400, 114), origin=(72, 114))
    stripped = "".join(f.text for f in _fragments([span], strip_marker=True))
    assert not stripped.strip().startswith("•")


@needs_ai_doc
def test_tick_entries_inside_one_block_are_separated():
    """This document sets all five entries as lines of a single block.

    The gear list gave each entry its own block; here they share one, so the
    standalone-block rule does not apply and the marker is what identifies
    them.
    """
    pages = _structure(AI_DOC)
    entries = [b for page in pages for b in page
               if b.kind == "bullet" and ("✅" in b.text or "❌" in b.text)]
    assert len(entries) >= 10, f"expected the rule entries, got {len(entries)}"
    for entry in entries:
        assert entry.text.count("✅") + entry.text.count("❌") == 1, \
            f"entries were run together: {entry.text[:60]!r}"


@needs_ai_doc
@needs_browser
def test_tick_entries_survive_the_whole_pipeline(tmp_path):
    translate_mod.set_provider(_Verbatim())
    out = str(tmp_path / "out.pdf")
    run_pipeline(AI_DOC, out, TranslationOptions(direction="en2ar"), QAReport())

    with fitz.open(out) as pdf:
        merged = 0
        for page in pdf:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                text = "".join(s["text"] for l in block.get("lines", [])
                               for s in l.get("spans", []))
                if text.count("✅") + text.count("❌") > 1:
                    merged += 1
        assert merged == 0, f"{merged} block(s) merged separate entries"

"""No-break spaces and bidi ordering in mixed Arabic/Latin text.

Two failure reports drove these tests. Only one of them was a defect in this
pipeline; the other turned out to be an artefact of how PyMuPDF reports text.
Both are pinned here so the distinction is not lost again.
"""
import fitz
import pytest

from app.core import fonts as fontlib
from app.core.extract import extract_pdf
from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.qa import QAReport
from app.core.rebuild_pdf import _wrap_lines
from app.core.shape_arabic import normalize_spaces, shape, shape_lines

ARABIC_FONT = "fonts/NotoNaskhArabic-Regular.ttf"
AR_WORDS = "دروس اللغة الإنجليزية المجانية"


# -- Bug 3: no-break spaces ------------------------------------------------

def test_normalize_spaces_replaces_nbsp():
    assert normalize_spaces("دروس اللغة") == "دروس اللغة"


def test_normalize_spaces_handles_other_fixed_width_spaces():
    assert normalize_spaces("a b c") == "a b c"
    # Word joiner and BOM carry no width and are dropped outright.
    assert normalize_spaces("a⁠b﻿c") == "abc"


def test_normalize_spaces_is_safe_on_ordinary_text():
    for text in ("", "plain text", AR_WORDS):
        assert normalize_spaces(text) == text


def test_nbsp_prevents_wrapping_without_normalisation():
    """Why nbsp matters: it makes a whole line one unbreakable word.

    `_wrap_lines` splits on U+0020, so an nbsp-joined string cannot wrap at
    all - it overflows its box and collides with whatever sits below. This is
    the mechanism behind the reported overlap, so it is asserted directly.
    """
    joined = AR_WORDS.replace(" ", " ")
    unwrappable = _wrap_lines(joined, ARABIC_FONT, 14, 60, shaped=True)
    wrappable = _wrap_lines(normalize_spaces(joined), ARABIC_FONT, 14, 60,
                            shaped=True)
    assert len(unwrappable) == 1, "nbsp text cannot be broken"
    assert len(wrappable) > 1, "normalised text wraps to fit the box"


def test_extraction_strips_nbsp(tmp_path):
    """A space drawn with an Arabic face comes back as U+00A0 from PyMuPDF.

    The character is not something this pipeline writes - it arrives on the way
    in - so extraction is where it has to be normalised, before it can reach
    the translator or the wrapping logic.
    """
    src = str(tmp_path / "nbsp.pdf")
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_font(fontname="ar", fontfile=ARABIC_FONT)
    page.insert_textbox(fitz.Rect(20, 20, 380, 120), shape(AR_WORDS),
                        fontname="ar", fontfile=ARABIC_FONT, fontsize=14,
                        align=fitz.TEXT_ALIGN_RIGHT)
    doc.save(src)
    doc.close()

    # The raw extractor does report the no-break space...
    with fitz.open(src) as pdf:
        assert " " in pdf[0].get_text()

    # ...but nothing downstream of our extraction ever sees it.
    page_obj = extract_pdf(src, QAReport()).pages[0]
    for block in page_obj.blocks:
        assert " " not in block.text
        for span in block.spans:
            assert " " not in span.text


def test_no_nbsp_reaches_the_translator(mixed_pdf):
    """Every string handed to the rebuild step is free of no-break spaces."""
    for page in extract_pdf(mixed_pdf, QAReport()).pages:
        for block in page.blocks:
            assert " " not in block.text
            assert " " not in block.text and "﻿" not in block.text


# -- Bug 4: bidi ordering --------------------------------------------------

MIXED_LINES = [
    "Arabic |العربية",
    "60 .الإدارية الخمسة",
    "المكاتب الخمسة 60 في المدينة",
]


@pytest.mark.parametrize("text", MIXED_LINES)
def test_bidi_runs_once_per_whole_line(text):
    """Shaping a full line must not equal shaping it word by word.

    `shape()` already calls reshape() + get_display() once on the complete
    logical string. Running bidi per fragment loses the algorithm's context and
    scrambles the order - this asserts the two differ, so a future refactor
    that starts shaping fragments is caught.
    """
    whole = shape(text)
    per_word = " ".join(shape(w) for w in text.split(" "))
    assert whole != per_word


def test_numeral_run_keeps_its_place_in_an_rtl_line():
    """An embedded numeral stays adjacent to the word it belongs with."""
    shaped = shape("المكاتب الخمسة 60 في المدينة")
    assert "60" in shaped
    # In visual order an RTL line starts at the right, so the Latin numeral
    # must not have been pushed to the visual start of the string.
    assert not shaped.startswith("60")


def test_wrapping_happens_on_logical_text_before_shaping():
    """Wrap points are chosen on logical text, then each line is shaped once.

    Shaping first and wrapping afterwards would break the line in visual space
    and reverse the paragraph, so this pins the order of the two steps.
    """
    text = "المكاتب الخمسة 60 في المدينة الكبرى للاختبار"
    lines = _wrap_lines(text, ARABIC_FONT, 12, 150, shaped=True)
    assert len(lines) > 1
    for line in lines:
        assert " " not in line
    # Every wrapped line is shaped as a unit, not per word.
    assert shape_lines(lines) == "\n".join(shape(l) for l in lines)


def test_mixed_script_line_renders_all_its_glyphs(tmp_path):
    """A mixed line must use a font covering both scripts.

    Noto Naskh has no Latin glyphs, so drawing "Arabic |العربية" in it silently
    renders the Latin half as notdef boxes. Font resolution has to notice.
    """
    text = "Arabic |العربية"
    resolved = fontlib.resolve_for_text(text, bold=False, italic=False,
                                        qa=QAReport())
    assert fontlib.covers(text, resolved.path), \
        "resolved font must cover both scripts"

    out = str(tmp_path / "mixed.pdf")
    doc = fitz.open()
    page = doc.new_page(width=500, height=200)
    page.insert_font(fontname=resolved.name, fontfile=resolved.path)
    page.insert_textbox(fitz.Rect(20, 20, 480, 80), shape(text),
                        fontname=resolved.name, fontfile=resolved.path,
                        fontsize=14, align=fitz.TEXT_ALIGN_RIGHT)
    doc.save(out)
    doc.close()

    with fitz.open(out) as pdf:
        rendered = pdf[0].get_text()
    assert "\x00" not in rendered, "missing glyphs were drawn as notdef"
    assert "Arabic" in rendered


def test_mixed_line_glyphs_are_drawn_in_visual_order(tmp_path):
    """The Latin run must sit where bidi puts it on the page.

    Read via get_text() the characters come back in the order the content
    stream stores them, which is not the order they appear on the page. Sorting
    the glyphs by their x origin gives the true visual order.
    """
    text = "المكاتب الخمسة 60 في المدينة"
    resolved = fontlib.resolve_for_text(text, bold=False, italic=False,
                                        qa=QAReport())
    out = str(tmp_path / "order.pdf")
    doc = fitz.open()
    page = doc.new_page(width=500, height=200)
    page.insert_font(fontname=resolved.name, fontfile=resolved.path)
    page.insert_textbox(fitz.Rect(20, 20, 480, 80), shape(text),
                        fontname=resolved.name, fontfile=resolved.path,
                        fontsize=14, align=fitz.TEXT_ALIGN_RIGHT)
    doc.save(out)
    doc.close()

    with fitz.open(out) as pdf:
        raw = pdf[0].get_text("rawdict")
    chars = [(c["c"], c["origin"][0])
             for b in raw["blocks"] for l in b.get("lines", [])
             for s in l.get("spans", []) for c in s["chars"]]
    visual = "".join(c for c, _ in sorted(chars, key=lambda t: t[1]))
    # An RTL line begins at the right-hand side, so the numeral cannot be the
    # leftmost thing on the page.
    assert not visual.strip().startswith("60")
    assert "60" in visual


# -- the reported overlap, end to end --------------------------------------

def test_no_block_overlaps_in_rebuilt_output(mixed_pdf, tmp_path):
    """The reported collision: two text boxes physically intersecting.

    Checked on the real rebuilt file rather than on internal state, because
    that is where the original report found it.
    """
    out = str(tmp_path / "out.pdf")
    run_pipeline(mixed_pdf, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), QAReport())

    with fitz.open(out) as pdf:
        boxes = []
        for block in pdf[0].get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            text = "".join(s["text"] for l in block.get("lines", [])
                           for s in l.get("spans", []))
            if text.strip():
                boxes.append((fitz.Rect(block["bbox"]), text.strip()))

    for i, (box_a, text_a) in enumerate(boxes):
        for box_b, text_b in boxes[i + 1:]:
            overlap = box_a & box_b
            if overlap.is_empty:
                continue
            smaller = min(abs(box_a), abs(box_b))
            share = abs(overlap) / smaller if smaller else 0
            assert share <= 0.15, \
                f"blocks overlap: {text_a[:30]!r} vs {text_b[:30]!r}"

"""A fragment inside a paragraph becomes part of its text.

A parenthetical gloss, or a number lifted into its own block, is part of the
sentence around it - but extraction gives it a box of its own. Kept as a box it
has to be *placed*, and after translation the surrounding lines no longer leave
a hole the right size for it, so it lands on top of them.

Folding it into the paragraph removes the placement problem rather than solving
it: from then on it wraps with the text and cannot collide with the lines it
belongs to.
"""
import os

import fitz
import pytest

from app.core.extract import extract_pdf
from app.core.merge import (
    INLINE_MAX_WORDS,
    inline_embedded_fragments,
    merge_paragraph_lines,
)
from app.core.models import BBox, Line, Page, Span, TextBlock
from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.qa import QAReport
from app.core.translate import TranslationProvider
import app.core.translate as translate_mod

REAL_PDF = "storage/d6984ec04a4c4f0d9f7a4ac1232243fc/input.pdf"
GLOSS = "(Mayor's Office of Immigrant Affairs, MOIA)"


def _span(text, x0, y0, x1, y1, size=20.0):
    return Span(text, "helv", size, (0, 0, 0), True, False, False,
                BBox(x0, y0, x1, y1), origin=(x0, y1))


def _paragraph(lines):
    """`lines` is a list of (text, x0, y0, x1, y1)."""
    built = [Line(spans=[_span(*l)], bbox=BBox(*l[1:])) for l in lines]
    box = built[0].bbox
    for line in built[1:]:
        box = box.union(line.bbox)
    return TextBlock(lines=built, bbox=box)


def _fragment(text, x0, y0, x1, y1, size=13.0):
    return TextBlock(lines=[Line(spans=[_span(text, x0, y0, x1, y1, size)],
                                 bbox=BBox(x0, y0, x1, y1))],
                     bbox=BBox(x0, y0, x1, y1))


def _page(blocks):
    return Page(number=0, width=612, height=792, blocks=list(blocks))


# -- folding a fragment in -------------------------------------------------

def test_gloss_inside_a_paragraph_is_folded_in():
    para = _paragraph([("first line", 160, 233, 570, 268),
                       ("second line", 160, 258, 570, 292),
                       ("third line", 160, 282, 570, 316)])
    gloss = _fragment(GLOSS, 310, 264, 570, 287)
    page = _page([para, gloss])

    assert inline_embedded_fragments(page, "ar2en") == 1
    assert len(page.blocks) == 1, "the fragment must no longer be its own block"
    assert GLOSS in page.blocks[0].text


def test_the_fragment_lands_in_reading_order():
    """It goes after the line it sat below, so the sentence still reads right."""
    para = _paragraph([("first line", 160, 233, 570, 268),
                       ("second line", 160, 258, 570, 292),
                       ("third line", 160, 282, 570, 316)])
    gloss = _fragment(GLOSS, 310, 264, 570, 287)
    page = _page([para, gloss])
    inline_embedded_fragments(page, "ar2en")

    text = page.blocks[0].text
    assert text.index("second line") < text.index(GLOSS) < text.index("third line")


def test_a_fragment_outside_the_paragraph_is_left_alone():
    para = _paragraph([("first line", 160, 233, 400, 268),
                       ("second line", 160, 258, 400, 292)])
    aside = _fragment("Aside", 450, 264, 570, 287)
    page = _page([para, aside])
    assert inline_embedded_fragments(page, "ar2en") == 0
    assert len(page.blocks) == 2


def test_a_long_block_is_not_swallowed():
    """Only a short run is inlined; a paragraph beside one is its own text."""
    para = _paragraph([("first line", 160, 233, 570, 268),
                       ("second line", 160, 258, 570, 292),
                       ("third line", 160, 282, 570, 316)])
    long_text = " ".join(["word"] * (INLINE_MAX_WORDS + 5))
    other = _fragment(long_text, 310, 264, 570, 287)
    page = _page([para, other])
    assert inline_embedded_fragments(page, "ar2en") == 0


def test_a_single_line_block_is_not_a_host():
    """Folding into a one-line block would just concatenate two labels."""
    label = _paragraph([("Email us", 160, 233, 300, 268)])
    other = _fragment("(MOIA)", 200, 240, 280, 260)
    page = _page([label, other])
    assert inline_embedded_fragments(page, "ar2en") == 0


def test_fragment_spans_keep_their_own_style():
    """It becomes a run of the paragraph, not a restyled copy.

    The gloss keeps its own size and font, so a smaller parenthetical still
    reads as a parenthetical after the fold.
    """
    para = _paragraph([("first line", 160, 233, 570, 268),
                       ("second line", 160, 258, 570, 292),
                       ("third line", 160, 282, 570, 316)])
    gloss = _fragment(GLOSS, 310, 264, 570, 287, size=13.0)
    page = _page([para, gloss])
    inline_embedded_fragments(page, "ar2en")

    sizes = {round(s.size, 1) for s in page.blocks[0].spans if s.text.strip()}
    assert sizes == {20.0, 13.0}


def test_spacing_is_added_around_the_fragment():
    para = _paragraph([("first line", 160, 233, 570, 268),
                       ("second line", 160, 258, 570, 292),
                       ("third line", 160, 282, 570, 316)])
    gloss = _fragment(GLOSS, 310, 264, 570, 287)
    page = _page([para, gloss])
    inline_embedded_fragments(page, "ar2en")
    assert f"second line {GLOSS}" in page.blocks[0].text


# -- the real document -----------------------------------------------------

@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_gloss_and_number_are_inlined():
    """The reported page: a gloss and a stray "60", both inside the sentence."""
    page = extract_pdf(REAL_PDF, QAReport()).pages[0]
    merge_paragraph_lines(page, "ar2en")
    assert inline_embedded_fragments(page, "ar2en") == 2

    body = [b for b in page.blocks if b.bbox and 220 < b.bbox.y0 < 345]
    assert len(body) == 1, "the sentence and its fragments must be one block"
    assert "MOIA" in body[0].text
    assert "60" in body[0].text


class _Realistic(TranslationProvider):
    """Stands in for a translator, keeping the reported sentence intact."""

    name = "realistic"

    def translate_batch(self, texts, direction):
        out = []
        for text in texts:
            if "يمتلك مكتب العمدة" in text:
                out.append(
                    "The Mayor's Office of Immigrant Affairs in New York City "
                    "(Mayor's Office of Immigrant Affairs, MOIA) has over 60 "
                    "English learning centers across the five boroughs."
                )
            elif any("؀" <= c <= "ۿ" for c in text):
                out.append(" ".join(["English"] * max(1, len(text.split()))))
            else:
                out.append(text)
        return out


@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_page_has_no_text_overlaps_after_inlining(tmp_path):
    """The collision is gone because there is no separate box left to place."""
    translate_mod.set_provider(_Realistic())
    out = str(tmp_path / "out.pdf")
    run_pipeline(REAL_PDF, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), QAReport())

    with fitz.open(out) as pdf:
        page = pdf[0]
        boxes = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            text = "".join(s["text"] for l in block.get("lines", [])
                           for s in l.get("spans", [])).strip()
            if text:
                boxes.append((fitz.Rect(block["bbox"]), text))

        for i, (box_a, text_a) in enumerate(boxes):
            for box_b, text_b in boxes[i + 1:]:
                overlap = box_a & box_b
                if overlap.is_empty:
                    continue
                share = abs(overlap) / min(abs(box_a), abs(box_b))
                assert share <= 0.15, \
                    f"text overlaps: {text_a[:26]!r} <-> {text_b[:26]!r}"


@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_paragraph_renders_flush_left(tmp_path):
    """Every line of the reflowed sentence starts at the same edge."""
    translate_mod.set_provider(_Realistic())
    out = str(tmp_path / "out.pdf")
    run_pipeline(REAL_PDF, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), QAReport())

    with fitz.open(out) as pdf:
        for block in pdf[0].get_text("rawdict")["blocks"]:
            if block.get("type") != 0 or not (200 < block["bbox"][1] < 370):
                continue
            lefts = []
            for line in block.get("lines", []):
                visible = [c for s in line["spans"] for c in s["chars"]
                           if not c["c"].isspace()]
                if visible:
                    lefts.append(round(visible[0]["origin"][0], 1))
            if len(lefts) < 2:
                continue
            edge = min(lefts)
            for x in lefts[:-1]:
                assert x == pytest.approx(edge, abs=0.5), \
                    f"ragged left edge: {lefts}"

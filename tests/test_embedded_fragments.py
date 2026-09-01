"""A foreign-script fragment inside a paragraph travels with that paragraph.

A parenthetical gloss - "(Mayor's Office of Immigrant Affairs, MOIA)" set
inside an Arabic sentence - reads as part of that sentence. On an ar2en job it
is already in the target language, so the content guard would freeze it; but
freezing its *position* while the sentence around it mirrors strands it on the
far side of the page, on top of whatever is there.
"""
import os

import pytest

from app.core.extract import extract_pdf
from app.core.mirror import (
    MirrorMode,
    find_host_paragraph,
    is_anchored_label,
    mirror_document_pages,
)
from app.core.models import BBox, ImageElement, Line, Page, Span, TextBlock
from app.core.qa import QAReport

W, H = 612.0, 792.0
REAL_PDF = "storage/d6984ec04a4c4f0d9f7a4ac1232243fc/input.pdf"
ARABIC = "يمتلك مكتب العمدة لشؤون المهاجرين في مدينة نيويورك"
GLOSS = "(Mayor's Office of Immigrant Affairs, MOIA)"


def _block(text, x0, y0, x1, y1):
    span = Span(text, "helv", 11.0, (0, 0, 0), False, False, False,
                BBox(x0, y0, x1, y1), origin=(x0, y1))
    block = TextBlock(lines=[Line(spans=[span], bbox=BBox(x0, y0, x1, y1))],
                      bbox=BBox(x0, y0, x1, y1))
    from app.core.mirror import block_is_rtl
    block.mirror = block_is_rtl(block)
    return block


def _page(blocks, images=()):
    return Page(number=0, width=W, height=H, blocks=list(blocks),
                images=list(images))


# -- finding the host ------------------------------------------------------

def test_gloss_finds_the_paragraph_it_sits_in():
    host = _block(ARABIC, 100, 250, 400, 290)
    gloss = _block(GLOSS, 410, 260, 570, 282)
    page = _page([host, gloss])
    assert find_host_paragraph(gloss, page) is host


def test_a_fragment_with_no_paragraph_beside_it_has_no_host():
    gloss = _block(GLOSS, 410, 600, 570, 622)
    other = _block(ARABIC, 100, 100, 400, 140)
    assert find_host_paragraph(gloss, _page([gloss, other])) is None


def test_a_pinned_block_is_never_chosen_as_host():
    """A host must be something that actually moves.

    An anchored caption stays where it is, so following it would leave the
    fragment exactly as stranded as freezing it would.
    """
    icon = ImageElement(bbox=BBox(60, 250, 100, 290))
    caption = _block("اعرف", 110, 258, 170, 280)
    gloss = _block(GLOSS, 180, 258, 340, 280)
    page = _page([caption, gloss], images=[icon])
    assert is_anchored_label(caption, page)
    assert find_host_paragraph(gloss, page) is not caption


# -- travelling with the host ---------------------------------------------

def test_gloss_moves_by_the_same_shift_as_its_host():
    host = _block(ARABIC, 100, 250, 400, 290)
    gloss = _block(GLOSS, 410, 260, 570, 282)
    page = _page([host, gloss])
    host_before = host.bbox.as_tuple()
    gloss_before = gloss.bbox.as_tuple()

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")

    shift = host.bbox.x0 - host_before[0]
    assert shift != 0, "the host must have moved"
    assert gloss.bbox.x0 == pytest.approx(gloss_before[0] + shift)
    assert gloss.bbox.width == pytest.approx(gloss_before[2] - gloss_before[0])


def test_gloss_is_shifted_not_reflected():
    """It keeps its own left-to-right run inside the paragraph that moved."""
    host = _block(ARABIC, 100, 250, 400, 290)
    span_a = Span("(Mayor's ", "helv", 11.0, (0, 0, 0), False, False, False,
                  BBox(410, 260, 480, 282), origin=(410, 282))
    span_b = Span("Office)", "helv", 11.0, (0, 0, 0), False, False, False,
                  BBox(480, 260, 550, 282), origin=(480, 282))
    gloss = TextBlock(lines=[Line(spans=[span_a, span_b],
                                 bbox=BBox(410, 260, 550, 282))],
                      bbox=BBox(410, 260, 550, 282))
    gloss.mirror = False
    page = _page([host, gloss])

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")

    # Both spans moved by one shared offset, so their order is unchanged.
    assert span_a.bbox.x0 < span_b.bbox.x0
    assert span_b.bbox.x0 - span_a.bbox.x0 == pytest.approx(70)


def test_a_standalone_english_block_still_freezes():
    """Only an embedded fragment follows a host; an isolated one is preserved."""
    gloss = _block(GLOSS, 410, 600, 570, 622)
    page = _page([gloss])
    before = gloss.bbox.as_tuple()
    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")
    assert gloss.bbox.as_tuple() == before


# -- body text must not be mistaken for a caption -------------------------

def test_a_paragraph_line_near_a_graphic_is_not_a_caption():
    """The regression that let a sentence line freeze beside an icon.

    Lines of one paragraph interleave vertically - a line's box includes
    ascenders and descenders reaching into the row beside it - while a caption
    and the item below it have clear space between them. Distance to the
    nearest graphic cannot separate the two on a real page.
    """
    icon = ImageElement(bbox=BBox(50, 250, 114, 300))
    line_one = _block(ARABIC, 158, 258, 306, 292)
    line_two = _block(ARABIC, 156, 282, 400, 316)   # overlaps line_one
    page = _page([line_one, line_two], images=[icon])
    assert not is_anchored_label(line_one, page)


def test_a_caption_above_a_separate_item_is_still_a_caption():
    """Clear space below it, so it is not part of a stack."""
    icon = ImageElement(bbox=BBox(200, 715, 240, 755))
    caption = _block("Learn more", 249, 716, 313, 738)
    below = _block("nyc.gov/wespeaknyc", 195, 740, 313, 757)
    page = _page([caption, below], images=[icon])
    assert is_anchored_label(caption, page)


# -- the real document -----------------------------------------------------

@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_gloss_travels_with_its_paragraph():
    """The reported case, from the document itself."""
    page = extract_pdf(REAL_PDF, QAReport()).pages[0]
    gloss = next(b for b in page.blocks if "MOIA" in b.text)
    assert find_host_paragraph(gloss, page) is not None, \
        "the gloss must find the paragraph it sits in"

    before = gloss.bbox.as_tuple()
    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")

    # It must have travelled - the bug was that it stayed put while the page
    # moved around it - and travelled as a rigid block, not been reflected.
    assert gloss.bbox.as_tuple() != before, \
        "the gloss was left behind by its paragraph"
    assert gloss.bbox.width == pytest.approx(before[2] - before[0]), \
        "the gloss must be shifted, not reflected"
    # It moved leftwards, the same way the Arabic body did.
    assert gloss.bbox.x0 < before[0]


@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_body_line_is_not_frozen_as_a_caption():
    """"في مدينة نيويورك" is a line of the sentence, not an icon caption."""
    page = extract_pdf(REAL_PDF, QAReport()).pages[0]
    line = next(b for b in page.blocks if "في مدينة" in b.text)
    assert not is_anchored_label(line, page)


# -- a fragment with nowhere to go is reported, not shuffled ---------------

def test_a_fragment_nudges_clear_of_its_neighbour():
    """Where the row has space, the fragment is moved into it."""
    host = _block(ARABIC, 60, 250, 200, 290)
    neighbour = _block("جار", 210, 258, 260, 282)
    gloss = _block("(MOIA)", 265, 260, 320, 282)
    page = _page([host, neighbour, gloss])
    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")
    overlap = (min(gloss.bbox.x1, neighbour.bbox.x1)
               - max(gloss.bbox.x0, neighbour.bbox.x0))
    assert overlap <= 0, "the fragment must not sit on its neighbour"


@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
def test_unplaceable_fragment_is_reported():
    """The reported page has no room for the gloss at any position.

    It is 264pt wide and the largest free gap on its row is 154pt, so no
    horizontal nudge or vertical drop can clear it - each would only land it on
    a different neighbour. It stays with its sentence and QA says so, rather
    than the pipeline pretending the layout is fine.
    """
    page = extract_pdf(REAL_PDF, QAReport()).pages[0]
    qa = QAReport()
    mirror_document_pages([page], MirrorMode.FULL, qa, "ar2en")
    assert any("fragment inside a paragraph" in e.message for e in qa.entries), \
        "an unplaceable fragment must be reported"

"""Short captions stay put; prose mirrors.

A caption like "Email us" or "Learn more" is positioned relative to the icon it
labels, not to the page's reading direction. Reflecting it across the page
detaches it from that icon and, in a crowded footer, drops it on top of
whatever sits opposite - a QR code, in the reported document.

Being short is not enough on its own to identify one: a heading is short too,
and a heading belongs to the flow and must mirror with it. What marks a caption
is that it is *attached* to a graphic beside it.
"""
import os

import fitz
import pytest

from app.core.extract import extract_pdf
from app.core.merge import split_contact_lines
from app.core.mirror import (
    MirrorMode,
    is_anchored_label,
    is_contact_detail,
    is_short_label,
    mirror_document_pages,
)
from app.core.models import (
    BBox,
    DrawingElement,
    ImageElement,
    Line,
    Page,
    Span,
    TextBlock,
)
from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.qa import QAReport

W, H = 612.0, 792.0
REAL_PDF = "storage/d6984ec04a4c4f0d9f7a4ac1232243fc/input.pdf"


def _block(text, x0, y0, x1, y1):
    span = Span(text, "helv", 10.0, (0, 0, 0), False, False, False,
                BBox(x0, y0, x1, y1), origin=(x0, y1))
    return TextBlock(lines=[Line(spans=[span], bbox=BBox(x0, y0, x1, y1))],
                     bbox=BBox(x0, y0, x1, y1))


def _page(blocks, images=(), drawings=()):
    return Page(number=0, width=W, height=H, blocks=list(blocks),
                images=list(images), drawings=list(drawings))


# -- the "is it short" half ------------------------------------------------

@pytest.mark.parametrize("text", [
    "Email us", "Learn more", "اعرف المزيد", "راسلنا عبر البريد الإلكتروني",
    "We offer:", "nyc.gov/wespeaknyc",
])
def test_captions_are_short(text):
    assert is_short_label(text)


@pytest.mark.parametrize("text", [
    "Each center offers free in-person English lessons for beginners.",
    "Uptime reached 99.9 percent. Response time fell by 40 percent.",
    "line one\nline two",
    "",
])
def test_prose_is_not_short(text):
    assert not is_short_label(text)


def test_a_sentence_is_never_a_label():
    """Sentence punctuation marks prose however few words it has."""
    assert not is_short_label("It works.")


# -- the "is it attached" half ---------------------------------------------

def test_a_caption_beside_an_icon_is_anchored():
    icon = ImageElement(bbox=BBox(200, 715, 240, 755))
    block = _block("Email us", 250, 720, 320, 740)
    assert is_anchored_label(block, _page([block], images=[icon]))


def test_a_heading_is_not_anchored_even_though_it_is_short():
    """The regression this guards: headings must keep mirroring.

    "Key Results" is as short as "Email us", but it is set across its column
    and belongs to the reading flow, so it has to travel with the layout.
    """
    block = _block("Key Results", 72, 300, 520, 320)
    assert is_short_label(block.text)
    assert not is_anchored_label(block, _page([block]))


def test_a_caption_with_no_graphic_nearby_is_not_anchored():
    block = _block("Email us", 250, 720, 320, 740)
    assert not is_anchored_label(block, _page([block]))


def test_a_caption_far_from_its_graphic_is_not_anchored():
    icon = ImageElement(bbox=BBox(50, 100, 90, 140))
    block = _block("Email us", 450, 720, 520, 740)
    assert not is_anchored_label(block, _page([block], images=[icon]))


def test_a_caption_beside_vector_artwork_is_anchored():
    """The footer icons in the reported file are vector paths, not images."""
    art = [DrawingElement(bbox=BBox(200 + i * 6, 715, 206 + i * 6, 750),
                          kind="path", fill=(0, 0, 0)) for i in range(6)]
    block = _block("Learn more", 250, 720, 320, 740)
    assert is_anchored_label(block, _page([block], drawings=art))


def test_a_wide_block_is_not_a_caption():
    """Set across the page, so it is content rather than a label."""
    icon = ImageElement(bbox=BBox(30, 715, 70, 755))
    block = _block("Email us", 80, 720, 560, 740)
    assert not is_anchored_label(block, _page([block], images=[icon]))


# -- behaviour through the mirror pass -------------------------------------

def test_anchored_label_holds_its_position():
    icon = ImageElement(bbox=BBox(200, 715, 240, 755))
    block = _block("Email us", 250, 720, 320, 740)
    page = _page([block], images=[icon])
    before = block.bbox.as_tuple()

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")

    assert block.bbox.as_tuple() == before, "a caption must not be reflected"


def test_prose_beside_a_graphic_still_mirrors():
    """Attachment only exempts captions - a paragraph still travels."""
    icon = ImageElement(bbox=BBox(30, 300, 70, 340))
    prose = _block(
        "Each center offers free in-person English lessons for beginners.",
        80, 300, 520, 340)
    page = _page([prose], images=[icon])
    before = prose.bbox.as_tuple()

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")

    assert prose.bbox.as_tuple() != before


def test_a_held_label_is_still_translated():
    """Holding position must not exempt the text from translation."""
    icon = ImageElement(bbox=BBox(200, 715, 240, 755))
    block = _block("راسلنا", 250, 720, 320, 740)
    page = _page([block], images=[icon])
    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")
    # Mirroring never touches content; the block is still the one to translate.
    assert page.blocks[0].translated is None
    assert page.blocks[0].text.strip() == "راسلنا"


# -- the real document -----------------------------------------------------

@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_footer_labels_hold_position():
    """"Email us" and "Learn more" from the reported file."""
    page = extract_pdf(REAL_PDF, QAReport()).pages[0]
    labels = [b for b in page.blocks
              if b.bbox and b.bbox.y0 > 690 and is_anchored_label(b, page)]
    assert len(labels) >= 2, "both footer captions must be recognised"
    before = {id(b): b.bbox.as_tuple() for b in labels}

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")

    for block in labels:
        assert block.bbox.as_tuple() == before[id(block)]


@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_footer_label_no_longer_lands_on_the_qr_code(tmp_path):
    """The reported collision: the caption mirrored onto the QR code."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(REAL_PDF, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), QAReport())

    with fitz.open(out) as pdf:
        page = pdf[0]
        # The QR block is the dense cluster of tiny paths on the left; a lone
        # stray path further right is not part of it.
        cells = [d["rect"] for d in page.get_drawings()
                 if d["rect"].y0 > 700 and d["rect"].width < 6]
        assert len(cells) > 50, "the QR code must still be on the page"
        left = min(r.x0 for r in cells)
        qr = fitz.Rect(left, min(r.y0 for r in cells),
                       max(r.x1 for r in cells if r.x0 < left + 120),
                       max(r.y1 for r in cells))

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0 or block["bbox"][1] < 700:
                continue
            text = "".join(s["text"] for l in block.get("lines", [])
                           for s in l.get("spans", []))
            if not text.strip():
                continue
            overlap = fitz.Rect(block["bbox"]) & qr
            assert overlap.is_empty, \
                f"footer text overlaps the QR code: {text[:30]!r}"


# -- URLs and email addresses never move -----------------------------------

@pytest.mark.parametrize("text", [
    "nyc.gov/wespeaknyc",
    "wespeaknyc@cityhall.nyc.gov",
    "https://example.com/path",
    "www.example.org",
    "info@example.gov",
    "example.com",
])
def test_contact_details_are_recognised(text):
    assert is_contact_detail(text)


@pytest.mark.parametrize("text", [
    "Email us", "Learn more", "اعرف المزيد",
    "Each center offers free lessons for beginners.",
    "Version 1 released", "",
])
def test_ordinary_text_is_not_a_contact_detail(text):
    assert not is_contact_detail(text)


def test_a_url_holds_its_position_with_no_icon_nearby():
    """Contact detail holds whether or not a graphic sits close to it.

    It is printed content: never translated, so it never reflows, and set
    under the icon it belongs to.
    """
    block = _block("nyc.gov/wespeaknyc", 195, 740, 313, 757)
    page = _page([block])
    before = block.bbox.as_tuple()
    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")
    assert block.bbox.as_tuple() == before


def test_prose_mentioning_a_domain_still_mirrors():
    """Only the detail itself is pinned, not a sentence that cites one."""
    block = _block(
        "Each center offers free in-person lessons for beginners today",
        72, 300, 520, 340)
    page = _page([block])
    before = block.bbox.as_tuple()
    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")
    assert block.bbox.as_tuple() != before


# -- side-by-side contact lines are separated ------------------------------

def _two_line_block(first, second):
    def span(text, x0, y0, x1, y1):
        return Span(text, "helv", 10.0, (0, 0, 0), False, False, False,
                    BBox(x0, y0, x1, y1), origin=(x0, y1))
    lines = [
        Line(spans=[span(first, 195, 740, 313, 757)],
             bbox=BBox(195, 740, 313, 757)),
        Line(spans=[span(second, 407, 740, 576, 757)],
             bbox=BBox(407, 740, 576, 757)),
    ]
    return TextBlock(lines=lines, bbox=BBox(195, 740, 576, 757))


def test_side_by_side_contact_lines_are_split():
    """A URL and an email under different icons must not share a block.

    Joined, they mirror as one unit, which carries each away from its own
    icon - and drops the far one onto whatever sits opposite.
    """
    block = _two_line_block("nyc.gov/wespeaknyc", "wespeaknyc@cityhall.nyc.gov")
    page = _page([block])
    assert split_contact_lines(page) == 1
    assert len(page.blocks) == 2
    assert [round(b.bbox.x0) for b in page.blocks] == [195, 407]


def test_a_wrapped_paragraph_is_never_split():
    """Lines that share a horizontal span are stacked text, not separate items."""
    def span(text, y0):
        return Span(text, "helv", 10.0, (0, 0, 0), False, False, False,
                    BBox(72, y0, 400, y0 + 14), origin=(72, y0 + 14))
    lines = [Line(spans=[span("first line of the paragraph", 300)],
                  bbox=BBox(72, 300, 400, 314)),
             Line(spans=[span("second line of the paragraph", 316)],
                  bbox=BBox(72, 316, 400, 330))]
    block = TextBlock(lines=lines, bbox=BBox(72, 300, 400, 330))
    page = _page([block])
    assert split_contact_lines(page) == 0
    assert len(page.blocks) == 1


def test_a_short_label_beside_a_url_also_splits():
    """Footer items need not all be contact details.

    A row often pairs a label with a link - "Learn more" over
    "nyc.gov/wespeaknyc" - each under its own icon. Requiring every line to be
    a URL or an email left rows like "Learn more" + "Email us" joined, so they
    mirrored as a unit and landed on top of the links below them.
    """
    block = _two_line_block("nyc.gov/wespeaknyc", "Learn more")
    page = _page([block])
    assert split_contact_lines(page) == 1
    assert len(page.blocks) == 2


def test_prose_beside_a_url_does_not_split():
    """Running text is not a footer item, however it is positioned."""
    block = _two_line_block(
        "nyc.gov/wespeaknyc",
        "Each center offers free in-person English lessons for beginners.")
    page = _page([block])
    assert split_contact_lines(page) == 0
    assert len(page.blocks) == 1


# -- the real document -----------------------------------------------------

@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_contact_details_hold_their_positions():
    """The URL and email from the reported footer, each under its own icon."""
    page = extract_pdf(REAL_PDF, QAReport()).pages[0]
    assert split_contact_lines(page) == 1

    contacts = [b for b in page.blocks
                if b.bbox and is_contact_detail(b.text)]
    assert len(contacts) >= 2
    before = {id(b): b.bbox.as_tuple() for b in contacts}

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")

    for block in contacts:
        assert block.bbox.as_tuple() == before[id(block)], \
            f"contact detail moved: {block.text[:30]!r}"


@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_contact_details_stay_clear_of_the_qr_code(tmp_path):
    out = str(tmp_path / "out.pdf")
    run_pipeline(REAL_PDF, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), QAReport())
    with fitz.open(out) as pdf:
        page = pdf[0]
        cells = [d["rect"] for d in page.get_drawings()
                 if d["rect"].y0 > 700 and d["rect"].width < 6]
        left = min(r.x0 for r in cells)
        qr = fitz.Rect(left, min(r.y0 for r in cells),
                       max(r.x1 for r in cells if r.x0 < left + 120),
                       max(r.y1 for r in cells))
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0 or block["bbox"][1] < 700:
                continue
            text = "".join(s["text"] for l in block.get("lines", [])
                           for s in l.get("spans", []))
            if not is_contact_detail(text):
                continue
            assert (fitz.Rect(block["bbox"]) & qr).is_empty, \
                f"contact detail overlaps the QR code: {text[:30]!r}"

        # Each detail must also still sit under its own icon, which only holds
        # if the two were separated: joined, they mirror as one unit and the
        # far one is carried across the page.
        url = [b for b in page.get_text("dict")["blocks"]
               if b.get("type") == 0 and b["bbox"][1] > 700
               and "nyc.gov/wespeaknyc" in "".join(
                   s["text"] for l in b.get("lines", [])
                   for s in l.get("spans", []))]
        assert url, "the URL must still be on the page"
        assert url[0]["bbox"][0] > qr.x1, \
            "the URL was carried across the page onto the QR side"

        # The URL and the email sit under different icons. Re-reading the
        # rebuilt file re-groups them into one block, so the check is on where
        # the two lines were actually drawn: joined, they collapse into a
        # single narrow box and the second loses its place under its own icon.
        drawn = [l["bbox"] for b in page.get_text("dict")["blocks"]
                 if b.get("type") == 0 and b["bbox"][1] > 700
                 for l in b.get("lines", [])
                 if is_contact_detail("".join(s["text"] for s in l["spans"]))]
        assert len(drawn) >= 2, "both contact details must be on the page"
        spread = max(b[2] for b in drawn) - min(b[0] for b in drawn)
        assert spread > 300, (
            "URL and email collapsed together instead of staying under their "
            f"own icons (spread {spread:.0f}pt)"
        )

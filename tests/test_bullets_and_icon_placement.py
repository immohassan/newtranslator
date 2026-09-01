"""Positional graphic mirroring and list-marker attachment.

Two layout rules that both come down to "where does this element actually sit":

  * a graphic on one side of the page carries a left/right relationship and
    moves when the page flips; one in the footer or centred does not;
  * a list marker belongs to the item beside it, not to a row of its own.
"""
import fitz
import pytest

from app.core.extract import extract_pdf
from app.core.merge import (
    BULLET_MARKER,
    attach_bullets,
    drop_vector_bullets,
    canonical_marker,
    is_bullet_text,
    strip_leading_marker,
)
from app.core.mirror import (
    MirrorMode,
    mirror_document_pages,
    should_mirror_element,
)
from app.core.models import BBox
from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.qa import QAReport

W, H = 595.0, 842.0


# -- which graphics move ---------------------------------------------------

@pytest.mark.parametrize("name,box", [
    ("left margin icon", BBox(30, 300, 60, 330)),
    ("right margin icon", BBox(535, 300, 565, 330)),
    ("left body rule", BBox(72, 136, 255, 138)),
    ("right pull quote", BBox(430, 400, 560, 460)),
])
def test_side_graphics_mirror(name, box):
    assert should_mirror_element(box, W, H) is True, name


@pytest.mark.parametrize("name,box", [
    ("footer envelope", BBox(300, 760, 330, 785)),
    ("footer globe", BBox(400, 760, 430, 785)),
    ("footer QR code", BBox(470, 745, 540, 800)),
    ("centred logo", BBox(270, 400, 325, 440)),
    ("centred masthead", BBox(250, 10, 345, 40)),
])
def test_footer_and_centred_graphics_stay_put(name, box):
    assert should_mirror_element(box, W, H) is False, name


def test_a_wide_left_rule_is_not_mistaken_for_centred():
    """Reaching past the middle is not the same as being centred.

    A left-margin rule can extend beyond the page's centre line and is still
    left-side content. Only an element whose own centre sits in the band, or
    which straddles the middle evenly, counts as centred.
    """
    assert should_mirror_element(BBox(72, 136, 340, 138), W, H) is True


def test_full_width_divider_is_treated_as_centred():
    assert should_mirror_element(BBox(40, 500, 555, 502), W, H) is False


def test_degenerate_page_size_defaults_to_mirroring():
    """A page with no usable dimensions must not silently drop the transform."""
    assert should_mirror_element(BBox(0, 0, 10, 10), 0, 0) is True


def test_footer_icons_hold_position_through_the_mirror_pass(bulleted_pdf):
    """End to end on a real page: side icons move, footer icons do not."""
    page = extract_pdf(bulleted_pdf, QAReport()).pages[0]
    before = {id(im): im.bbox.as_tuple() for im in page.images}
    expected = {id(im): should_mirror_element(im.bbox, page.width, page.height)
                for im in page.images}
    assert any(expected.values()), "fixture needs a side graphic"
    assert not all(expected.values()), "fixture needs a footer graphic"

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "en2ar")

    for image in page.images:
        was = before[id(image)]
        if expected[id(image)]:
            assert image.bbox.x0 == pytest.approx(page.width - was[2])
        else:
            assert image.bbox.as_tuple() == was, "footer graphic was moved"


# -- list markers ----------------------------------------------------------

@pytest.mark.parametrize("marker", ["•", "·", "▪", "◦", "-", "*", "1.", "2)", "a."])
def test_markers_are_recognised(marker):
    assert is_bullet_text(marker)


@pytest.mark.parametrize("text", [
    "", "   ", "Free training materials", "A", "12345", "Note.",
])
def test_ordinary_text_is_not_a_marker(text):
    assert not is_bullet_text(text)


def test_marker_is_joined_to_its_item(bulleted_pdf):
    """A marker must share a line with the text it introduces.

    Left on its own it is laid out as a line of its own, which is what put the
    dots on the row above their items in the reported output.
    """
    page = extract_pdf(bulleted_pdf, QAReport()).pages[0]
    assert attach_bullets(page) > 0

    for block in page.blocks:
        for line in block.lines:
            if is_bullet_text(line.text):
                pytest.fail(f"marker still on its own line: {line.text!r}")


def test_marker_and_item_end_up_on_one_line(bulleted_pdf):
    page = extract_pdf(bulleted_pdf, QAReport()).pages[0]
    attach_bullets(page)
    items = [b.text for b in page.blocks if "training materials" in b.text]
    assert items, "expected the bulleted item"
    assert items[0].startswith(("•", "·")), items[0]
    assert "\n" not in items[0].strip()


def test_numbered_markers_are_joined_too(bulleted_pdf):
    page = extract_pdf(bulleted_pdf, QAReport()).pages[0]
    attach_bullets(page)
    joined = [b.text for b in page.blocks if "First step" in b.text]
    assert joined and joined[0].strip().startswith("1.")


def test_attaching_is_idempotent(bulleted_pdf):
    """Running the pass twice must not double the markers."""
    page = extract_pdf(bulleted_pdf, QAReport()).pages[0]
    attach_bullets(page)
    texts = [b.text for b in page.blocks]
    assert attach_bullets(page) == 0
    assert [b.text for b in page.blocks] == texts


def test_a_lone_marker_is_kept_not_dropped():
    """A marker with nothing to attach to must survive as content."""
    from app.core.models import BBox as B, Line, Page, Span, TextBlock

    span = Span("•", "helv", 11.0, (0, 0, 0), False, False, False,
                B(72, 100, 80, 112), origin=(72, 112))
    block = TextBlock(lines=[Line(spans=[span], bbox=B(72, 100, 80, 112))],
                      bbox=B(72, 100, 80, 112))
    page = Page(number=0, width=W, height=H, blocks=[block])
    attach_bullets(page)
    assert page.blocks, "a lone marker must not be silently deleted"


def test_bullets_render_inline_after_rebuild(bulleted_pdf, tmp_path):
    """The rendered page must not contain a line that is only a marker."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(bulleted_pdf, out,
                 TranslationOptions(direction="en2ar", mirror=True, html_engine=False), QAReport())
    with fitz.open(out) as pdf:
        for block in pdf[0].get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                if text and is_bullet_text(text):
                    pytest.fail(f"marker rendered on its own line: {text!r}")


def test_pipeline_reports_bullet_joins(bulleted_pdf, tmp_path):
    out = str(tmp_path / "out.pdf")
    qa = QAReport()
    run_pipeline(bulleted_pdf, out,
                 TranslationOptions(direction="en2ar", mirror=True, html_engine=False), qa)
    assert any("list marker" in e.message for e in qa.entries)


# -- markers are replaced, not repositioned --------------------------------

@pytest.mark.parametrize("text,expected", [
    ("\u00b7Free training materials", "Free training materials"),
    ("\u2022 A diverse community", "A diverse community"),
    ("1. First step", "First step"),
    ("2) Second", "Second"),
])
def test_leading_marker_is_split_off(text, expected):
    _, rest, found = strip_leading_marker(text)
    assert found and rest.strip() == expected


def test_text_without_a_marker_is_untouched():
    marker, rest, found = strip_leading_marker("No marker here")
    assert not found and rest == "No marker here" and marker == ""


def test_a_bare_marker_is_not_treated_as_an_item():
    """Nothing would be left of it, so it is not a list item."""
    _, _, found = strip_leading_marker("\u2022")
    assert not found


@pytest.mark.parametrize("source,expected", [
    ("\u00b7 item", BULLET_MARKER),
    ("\u25aa item", BULLET_MARKER),
    ("- item", BULLET_MARKER),
    ("1. item", "1."),
    ("2) item", "2)"),
    ("a. item", "a."),
])
def test_canonical_marker(source, expected):
    assert canonical_marker(source) == expected


def test_every_bullet_uses_the_same_marker(bulleted_pdf):
    """The point of replacing rather than repositioning: they now match.

    Source markers arrive at assorted sizes, baselines and offsets. After
    normalisation every bulleted item on the page starts with one glyph.
    """
    page = extract_pdf(bulleted_pdf, QAReport()).pages[0]
    attach_bullets(page)
    bulleted = [b.text for b in page.blocks if b.text.startswith(BULLET_MARKER)]
    assert len(bulleted) >= 4
    for text in bulleted:
        assert text.startswith(f"{BULLET_MARKER} "), repr(text)


def test_marker_glued_to_its_text_is_separated(bulleted_pdf):
    """The reported collision: marker and text extracted as one run.

    A 5pt marker set tight against 11pt text reads as "\u00b7Tight marker item",
    which renders as the dot colliding with the first letter.
    """
    page = extract_pdf(bulleted_pdf, QAReport()).pages[0]
    attach_bullets(page)
    items = [b.text for b in page.blocks if "Tight marker item" in b.text]
    assert items == [f"{BULLET_MARKER} Tight marker item"], items


def test_odd_sized_marker_span_is_dropped(bulleted_pdf):
    """The marker's own size and baseline go with it.

    A marker span set at 5pt beside 11pt text is what pushes the dot off the
    line. Removing that span leaves the item as one run at one size.
    """
    page = extract_pdf(bulleted_pdf, QAReport()).pages[0]
    attach_bullets(page)
    for block in page.blocks:
        if not block.text.startswith(BULLET_MARKER):
            continue
        sizes = {round(sp.size, 1) for sp in block.spans if sp.text.strip()}
        assert len(sizes) == 1, f"marker kept its own size: {sizes}"


def test_numbering_survives_normalisation(bulleted_pdf):
    """An ordered list keeps its sequence - a bullet would lose it."""
    page = extract_pdf(bulleted_pdf, QAReport()).pages[0]
    attach_bullets(page)
    ordered = [b.text for b in page.blocks
               if b.text.startswith(("1.", "2."))]
    assert sorted(ordered) == ["1. First step", "2. Second step"], ordered


def test_normalised_bullets_render_inline_and_uniform(bulleted_pdf, tmp_path):
    """On the rebuilt page every bullet sits on its item's own line."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(bulleted_pdf, out,
                 TranslationOptions(direction="en2ar", mirror=True, html_engine=False), QAReport())
    with fitz.open(out) as pdf:
        for block in pdf[0].get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                if text and is_bullet_text(text):
                    pytest.fail(f"marker alone on a line: {text!r}")


# -- markers the source drew as vector paths -------------------------------

def _dot(x0, y0, size=3.6, fill=(31, 78, 162), kind="path"):
    from app.core.models import DrawingElement
    return DrawingElement(bbox=BBox(x0, y0, x0 + size, y0 + size),
                          kind=kind, fill=fill)


def _page_with(drawings, blocks):
    from app.core.models import Page
    return Page(number=0, width=W, height=H, blocks=blocks, drawings=drawings)


def _text_block(text, x0, y0, x1, y1):
    from app.core.models import Line, Span, TextBlock
    span = Span(text, "helv", 11.0, (0, 0, 0), False, False, False,
                BBox(x0, y0, x1, y1), origin=(x0, y1))
    return TextBlock(lines=[Line(spans=[span], bbox=BBox(x0, y0, x1, y1))],
                     bbox=BBox(x0, y0, x1, y1))


def test_vector_bullet_beside_text_is_dropped():
    """A small filled dot next to a line of text is a list marker."""
    page = _page_with([_dot(60, 100)],
                      [_text_block("Free training materials", 80, 96, 300, 112)])
    assert drop_vector_bullets(page) == 1
    assert page.drawings == []


def test_vector_bullet_at_a_block_edge_is_dropped():
    """A whole list often extracts as one block, with the markers inside it.

    The marker frequently defines the block's own edge, so the containment test
    needs a point of slack or the two coordinates miss by a rounding error.
    """
    page = _page_with([_dot(572.4, 487.4)],
                      [_text_block("item", 144.9, 447.1, 576.0, 563.8)])
    assert drop_vector_bullets(page) == 1


def test_a_large_filled_shape_is_not_a_bullet():
    """An icon or logo element must survive."""
    page = _page_with([_dot(60, 100, size=24)],
                      [_text_block("Some text", 80, 96, 300, 112)])
    assert drop_vector_bullets(page) == 0
    assert len(page.drawings) == 1


def test_an_elongated_shape_is_not_a_bullet():
    """A rule or dash is not a marker, however thin."""
    from app.core.models import DrawingElement
    dash = DrawingElement(bbox=BBox(60, 100, 90, 103), kind="path",
                          fill=(0, 0, 0))
    page = _page_with([dash], [_text_block("text", 100, 96, 300, 112)])
    assert drop_vector_bullets(page) == 0


def test_an_unfilled_path_is_not_a_bullet():
    page = _page_with([_dot(60, 100, fill=None)],
                      [_text_block("text", 80, 96, 300, 112)])
    assert drop_vector_bullets(page) == 0


def test_a_dot_far_from_any_text_is_left_alone():
    """A dot in a chart or logo is not beside a line of text."""
    page = _page_with([_dot(60, 700)],
                      [_text_block("text", 80, 96, 300, 112)])
    assert drop_vector_bullets(page) == 0


def test_a_page_without_text_keeps_its_drawings():
    page = _page_with([_dot(60, 100)], [])
    assert drop_vector_bullets(page) == 0
    assert len(page.drawings) == 1


def test_tab_only_line_counts_as_a_marker():
    """Where the bullet was a vector path, the text layer holds only a tab.

    The tab is what the item's text follows, so it marks the item - and the
    canonical bullet is written onto the item's own span, since a tab carries
    no glyph of its own.
    """
    from app.core.models import Line, Page, Span, TextBlock
    def span(text, x0, y0):
        return Span(text, "helv", 11.0, (0, 0, 0), False, False, False,
                    BBox(x0, y0, x0 + 60, y0 + 12), origin=(x0, y0 + 12))
    block = TextBlock(
        lines=[Line(spans=[span("\t", 70, 100)], bbox=BBox(70, 100, 76, 112)),
               Line(spans=[span("Free training materials", 80, 100)],
                    bbox=BBox(80, 100, 300, 112))],
        bbox=BBox(70, 100, 300, 112))
    page = Page(number=0, width=W, height=H, blocks=[block])
    assert attach_bullets(page) >= 1
    assert page.blocks[0].text.strip().startswith(f"{BULLET_MARKER} ")
    assert "\t" not in page.blocks[0].text


# -- the real reported document -------------------------------------------

REAL_PDF = "storage/d6984ec04a4c4f0d9f7a4ac1232243fc/input.pdf"


@pytest.mark.skipif(not __import__("os").path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_document_bullets_are_normalised(tmp_path):
    """The reported file: four dots drawn as 3.6pt filled paths.

    They cannot travel with their items - they belong to no text block - so
    they keep the source coordinates while the text reflows and remirrors
    around them, which is what put them between rows and on top of letters.
    """
    page = extract_pdf(REAL_PDF, QAReport()).pages[0]
    before = len(page.drawings)
    assert drop_vector_bullets(page) == 4
    assert len(page.drawings) == before - 4, "only the markers were removed"

    assert attach_bullets(page) >= 4
    items = [b for b in page.blocks if BULLET_MARKER in b.text]
    assert items, "list items must carry a text marker"


@pytest.mark.skipif(not __import__("os").path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_document_leaves_no_stray_dots(tmp_path):
    """End to end: no small filled path survives beside the list rows."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(REAL_PDF, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), QAReport())
    with fitz.open(out) as pdf:
        strays = [d for d in pdf[0].get_drawings()
                  if d["rect"].width < 9 and d["rect"].height < 9
                  and 430 < d["rect"].y0 < 600]
    assert not strays, f"{len(strays)} stray marker(s) left on the page"


def test_text_landing_on_held_artwork_is_reported():
    """A graphic that holds position is an obstacle, not empty space.

    A footer icon row or QR code deliberately keeps its place, so text mirrored
    into that space overprints it. Neither is moved - the graphic is anchored
    to the page and the text has nowhere better to go on a full page - but the
    clash is reported rather than silently rendered.
    """
    from app.core.models import DrawingElement, Line, Page, Span, TextBlock

    # A QR-like block of small filled paths in the footer strip.
    # One contiguous graphic in the footer strip, on the left of the page.
    art = [DrawingElement(bbox=BBox(50 + i * 6, 760, 56 + i * 6, 790),
                          kind="path", fill=(0, 0, 0)) for i in range(10)]
    # Text on the right, which mirrors onto that graphic's position.
    span = Span("Email us", "helv", 10.0, (0, 0, 0), False, False, False,
                BBox(485, 763, 545, 787), origin=(485, 787))
    block = TextBlock(lines=[Line(spans=[span], bbox=BBox(485, 763, 545, 787))],
                      bbox=BBox(485, 763, 545, 787))
    page = Page(number=0, width=W, height=H, blocks=[block], drawings=art)

    qa = QAReport()
    mirror_document_pages([page], MirrorMode.FULL, qa, "ar2en")

    assert any("stays in place" in e.message for e in qa.entries), \
        "an overlap with held artwork must be reported"


@pytest.mark.skipif(not __import__("os").path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_document_reports_the_footer_collision(tmp_path):
    """The reported QR-over-"Email us" clash reaches the QA report."""
    out = str(tmp_path / "out.pdf")
    qa = QAReport()
    run_pipeline(REAL_PDF, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), qa)
    assert any("stays in place" in e.message for e in qa.entries)


# -- a re-anchored block must not freeze artwork ---------------------------

def test_reanchored_block_does_not_pin_artwork():
    """`preserved_boxes` means "did not move" - a re-anchored block did.

    An LTR block inside a mirrored page has its box reflected to find its new
    anchor. Treating that new box as preserved froze every graphic under it:
    on an Arabic page translated en2ar the body icons stayed on the side they
    started, overlapping the text that had moved.
    """
    from app.core.models import DrawingElement, Line, Page, Span, TextBlock

    # A wide English paragraph, which is re-anchored rather than reflected.
    span = Span("The Mayor's Office of Immigrant Affairs in New York City",
                "helv", 12.0, (0, 0, 0), False, False, False,
                BBox(30, 232, 446, 309), origin=(30, 309))
    block = TextBlock(lines=[Line(spans=[span], bbox=BBox(30, 232, 446, 309))],
                      bbox=BBox(30, 232, 446, 309))
    block.mirror = False

    # An icon on the far side, well clear of the block's *source* box.
    icon = [DrawingElement(bbox=BBox(498 + i * 12, 251, 508 + i * 12, 300),
                           kind="path", fill=(0, 0, 0)) for i in range(4)]
    page = Page(number=0, width=612, height=792, blocks=[block],
                drawings=icon)
    before = [d.bbox.x0 for d in icon]

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "en2ar")

    assert all(abs(d.bbox.x0 - was) > 1 for d, was in zip(icon, before)), \
        "the icon was pinned by a block that had itself moved"
    assert all(d.bbox.x0 < 306 for d in icon), \
        "the icon should have crossed to the other side of the page"


ROUNDTRIP_PDF = "storage/1b5a6ab6d24944f18453b9b3a526bc84/input.pdf"


@pytest.mark.skipif(not __import__("os").path.exists(ROUNDTRIP_PDF),
                    reason="sample document not present")
def test_real_body_icons_mirror_on_en2ar():
    """The reported page: body icons stranded on the right, over the text."""
    page = extract_pdf(ROUNDTRIP_PDF, QAReport()).pages[0]
    icon = [d for d in page.drawings if 245 <= d.bbox.y0 < 310]
    assert icon, "fixture expectation: a body icon around y=250-310"
    before = [d.bbox.x0 for d in icon]

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "en2ar")

    for drawing, was in zip(icon, before):
        assert abs(drawing.bbox.x0 - was) > 1, "body icon did not mirror"
    assert all(d.bbox.x1 < page.width / 2 for d in icon), \
        "the icon started on the right and must end on the left"

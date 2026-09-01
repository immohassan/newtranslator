"""Horizontal mirroring maths and image flipping."""
import pytest

from app.core.extract import extract_pdf
from app.core.mirror import (
    MirrorMode,
    flip_image_bytes,
    mirror_bbox,
    mirror_document_pages,
    resolve_mode,
)
from app.core.models import BBox
from app.core.qa import QAReport


def test_mirror_formula():
    """new_x0 = W - x1 and new_x1 = W - x0; y untouched."""
    out = mirror_bbox(BBox(100, 200, 300, 250), 595)
    assert out.x0 == 595 - 300
    assert out.x1 == 595 - 100
    assert out.y0 == 200 and out.y1 == 250


def test_mirror_preserves_width():
    box = BBox(72, 100, 380, 140)
    assert mirror_bbox(box, 595).width == pytest.approx(box.width)


def test_mirror_is_its_own_inverse():
    """ar2en uses the same transform to undo an en2ar mirror."""
    box = BBox(72, 100, 380, 140)
    assert mirror_bbox(mirror_bbox(box, 595), 595).as_tuple() == box.as_tuple()


def test_full_mirror_moves_every_element(sample_pdf):
    doc = extract_pdf(sample_pdf, QAReport())
    page = doc.pages[0]
    image_before = page.images[0].bbox.x0
    block_before = page.blocks[0].bbox.x0

    mirror_document_pages(doc.pages, MirrorMode.FULL, QAReport())

    assert page.images[0].bbox.x0 != image_before
    assert page.blocks[0].bbox.x0 != block_before
    # The arrow started on the right; after mirroring it must sit on the left.
    assert page.images[0].bbox.x0 < page.width / 2


def test_align_mode_leaves_positions(sample_pdf):
    doc = extract_pdf(sample_pdf, QAReport())
    before = [b.bbox.as_tuple() for b in doc.pages[0].blocks]
    mirror_document_pages(doc.pages, MirrorMode.ALIGN, QAReport())
    assert [b.bbox.as_tuple() for b in doc.pages[0].blocks] == before


def test_spans_and_drawings_mirror_too(sample_pdf):
    doc = extract_pdf(sample_pdf, QAReport())
    page = doc.pages[0]
    span_before = page.blocks[0].spans[0].bbox.x0
    drawing_before = page.drawings[0].bbox.x0

    mirror_document_pages(doc.pages, MirrorMode.FULL, QAReport())

    assert page.blocks[0].spans[0].bbox.x0 != span_before
    assert page.drawings[0].bbox.x0 != drawing_before


def test_resolve_mode():
    assert resolve_mode(True) is MirrorMode.FULL
    assert resolve_mode(False) is MirrorMode.ALIGN


def test_flip_image_changes_pixels():
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (10, 4), "white")
    img.putpixel((0, 0), (255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    original = buf.getvalue()

    flipped = flip_image_bytes(original)
    result = Image.open(BytesIO(flipped))
    assert result.getpixel((9, 0)) == (255, 0, 0), "red pixel must move to the right"


def test_flip_bad_bytes_returns_input():
    """A broken image must not take the whole job down."""
    assert flip_image_bytes(b"not an image") == b"not an image"

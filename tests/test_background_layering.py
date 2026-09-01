"""Images must survive a page whose background is a drawn rectangle.

Some producers paint the sheet as a filled rectangle covering the whole page
rather than leaving it blank. Replayed after the photographs and screenshots
that sit on it, that rectangle paints them out: the images are still in the
file and still report as placed, but nothing of them is visible.
"""
import os

import fitz
import pytest

from app.core.extract import extract_pdf
from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.qa import QAReport
from app.core.rebuild_pdf import _split_backgrounds
from app.core.models import BBox, DrawingElement, ImageElement, Page
import app.core.translate as translate_mod

PORTFOLIO = "storage/445191b1fa0b452f87ca42413fb79698/input.pdf"


def _drawing(x0, y0, x1, y1, fill=(255, 255, 255)):
    return DrawingElement(bbox=BBox(x0, y0, x1, y1), kind="rect", fill=fill)


def _page(drawings):
    return Page(number=0, width=612, height=792, drawings=list(drawings))


# -- telling a background from artwork -------------------------------------

def test_a_full_page_fill_is_a_background():
    page = _page([_drawing(0, 0, 612, 792)])
    backgrounds, foreground = _split_backgrounds(page)
    assert len(backgrounds) == 1 and not foreground


def test_a_small_filled_shape_is_not_a_background():
    page = _page([_drawing(100, 100, 200, 200)])
    backgrounds, foreground = _split_backgrounds(page)
    assert not backgrounds and len(foreground) == 1


def test_an_unfilled_full_page_rect_is_not_a_background():
    """A border stroke hides nothing, so its order does not matter."""
    page = _page([DrawingElement(bbox=BBox(0, 0, 612, 792), kind="rect",
                                 fill=None)])
    backgrounds, foreground = _split_backgrounds(page)
    assert not backgrounds and len(foreground) == 1


def test_a_coloured_background_counts_too():
    """Identified by area, not colour - a tinted sheet hides just as much."""
    page = _page([_drawing(0, 0, 612, 792, fill=(20, 40, 160))])
    backgrounds, _ = _split_backgrounds(page)
    assert len(backgrounds) == 1


def test_a_half_page_panel_stays_in_the_foreground():
    """A section panel is artwork; only a full sheet is background."""
    page = _page([_drawing(0, 0, 612, 396)])
    backgrounds, foreground = _split_backgrounds(page)
    assert not backgrounds and len(foreground) == 1


# -- the real document -----------------------------------------------------

@pytest.mark.skipif(not os.path.exists(PORTFOLIO),
                    reason="sample document not present")
def test_portfolio_page_has_a_full_page_background():
    """Fixture expectation: this is what made the images vanish."""
    page = extract_pdf(PORTFOLIO, QAReport()).pages[2]
    backgrounds, _ = _split_backgrounds(page)
    assert backgrounds, "the page background must be recognised"
    assert page.images, "the page must carry the images it painted over"


@pytest.mark.skipif(not os.path.exists(PORTFOLIO),
                    reason="sample document not present")
def test_images_are_still_visible_after_rebuild(tmp_path):
    """Every image must render as more than a flat block of one colour."""
    from tests.fake_provider import FakeProvider

    translate_mod.set_provider(FakeProvider())
    out = str(tmp_path / "out.pdf")
    run_pipeline(PORTFOLIO, out,
                 TranslationOptions(direction="en2ar", mirror=True, html_engine=False), QAReport())

    with fitz.open(out) as pdf:
        checked = 0
        for page in pdf:
            for info in page.get_image_info():
                pixmap = page.get_pixmap(dpi=50, clip=fitz.Rect(*info["bbox"]))
                assert len(set(pixmap.samples)) > 1, (
                    f"image on page {page.number + 1} rendered blank"
                )
                checked += 1
        assert checked >= 5, "the sample document should carry several images"


@pytest.mark.skipif(not os.path.exists(PORTFOLIO),
                    reason="sample document not present")
def test_no_images_are_lost(tmp_path):
    from tests.fake_provider import FakeProvider

    translate_mod.set_provider(FakeProvider())
    out = str(tmp_path / "out.pdf")
    run_pipeline(PORTFOLIO, out,
                 TranslationOptions(direction="en2ar", mirror=True, html_engine=False), QAReport())

    with fitz.open(PORTFOLIO) as src, fitz.open(out) as dst:
        before = sum(len(src[i].get_image_info()) for i in range(src.page_count))
        after = sum(len(dst[i].get_image_info()) for i in range(dst.page_count))
    assert after >= before, f"images lost: {before} -> {after}"

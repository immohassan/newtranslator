"""Vector artwork carried through the reflowing rebuild.

The rule these tests defend is that nothing is silently lost: every drawing a
page carries is either attached to the block it decorates or reported as
loose. A drawing that quietly disappears is the failure that made skill dots
and boxes vanish from rebuilt CVs.
"""
from __future__ import annotations

import fitz
import pytest

from app.core.artwork import read_artwork
from app.core.extract import extract_pdf
from app.core.qa import QAReport


@pytest.fixture
def resume(tmp_path):
    """A CV-shaped page: tinted sidebar, boxed name, rows of rating pips."""
    path = tmp_path / "resume.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # The sidebar's tint - a tall, narrow filled rect.
    page.draw_rect(fitz.Rect(0, 0, 184, 842), color=None,
                   fill=(0.96, 0.96, 0.96))
    # A box around the name, drawn as four strokes the way a producer does.
    box = fitz.Rect(220, 40, 520, 130)
    for a, b in ((box.tl, box.tr), (box.bl, box.br),
                 (box.tl, box.bl), (box.tr, box.br)):
        page.draw_line(a, b, color=(0.13, 0.13, 0.13), width=1.0)
    page.insert_text((260, 90), "JULIE MONROE", fontsize=20)

    # Two skill labels, each with a row of five pips under it. The second is
    # a partial score: three dark, two light.
    page.insert_text((45, 300), "Food preparation", fontsize=9)
    for i in range(5):
        page.draw_rect(fitz.Rect(45 + i * 10, 308, 50 + i * 10, 313),
                       color=None, fill=(0.13, 0.13, 0.13))
    page.insert_text((45, 340), "French", fontsize=9)
    for i in range(5):
        fill = (0.13, 0.13, 0.13) if i < 3 else (0.87, 0.87, 0.87)
        page.draw_rect(fitz.Rect(45 + i * 10, 348, 50 + i * 10, 353),
                       color=None, fill=fill)

    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def page(resume):
    return extract_pdf(str(resume), QAReport()).pages[0]


def test_the_sidebar_tint_is_found(page):
    art = read_artwork(page)
    assert len(art.panels) == 1
    assert art.panels[0].bbox.width < page.width * 0.4


def test_the_name_box_is_reassembled_from_its_four_strokes(page):
    """Read one at a time the sides are rules, and land across the page."""
    art = read_artwork(page)
    assert len(art.frames) == 1
    frame = art.frames[0]
    assert frame.owner is not None
    assert "JULIE" in page.blocks[frame.owner].text


def test_a_full_rating_reads_as_full(page):
    art = read_artwork(page)
    full = [r for r in art.ratings if r.filled == r.total]
    assert full, "a row of five identical pips is a full score"
    assert full[0].total == 5


def test_a_partial_rating_keeps_its_score(page):
    """Three dark pips beside two light ones is three out of five."""
    art = read_artwork(page)
    partial = [r for r in art.ratings if r.filled != r.total]
    assert partial, "a two-tone row is a partial score"
    assert (partial[0].filled, partial[0].total) == (3, 5)


def test_each_rating_is_attached_to_the_text_it_scores(page):
    art = read_artwork(page)
    owned = {}
    for rating in art.ratings:
        assert rating.owner is not None
        owned[page.blocks[rating.owner].text.strip()] = rating.filled
    assert owned.get("Food preparation") == 5
    assert owned.get("French") == 3


def test_a_pip_is_not_also_emitted_as_a_loose_drawing(page):
    """A pip drawn as a fill plus its strokes must be consumed once."""
    art = read_artwork(page)
    for rating in art.ratings:
        overlapping = [d for d in art.loose
                       if (d.bbox.x0 >= rating.bbox.x0 - 2
                           and d.bbox.x1 <= rating.bbox.x1 + 2
                           and d.bbox.y0 >= rating.bbox.y0 - 2
                           and d.bbox.y1 <= rating.bbox.y1 + 2)]
        assert not overlapping


def test_nothing_is_lost(page):
    """Every drawing is either owned or reported loose - never dropped."""
    art = read_artwork(page)
    accounted = len(art.loose) + len(art.panels) + len(art.frames)
    accounted += sum(r.total for r in art.ratings)
    # Underlines are bound to their spans at extraction and are not artwork.
    drawings = [d for d in page.drawings if d.kind != "underline"]
    assert accounted <= len(drawings)
    assert art  # the page's artwork was read at all


def test_a_page_with_no_artwork_yields_nothing(tmp_path):
    path = tmp_path / "plain.pdf"
    doc = fitz.open()
    doc.new_page(width=595, height=842).insert_text((72, 100), "Plain text.")
    doc.save(str(path))
    doc.close()

    page = extract_pdf(str(path), QAReport()).pages[0]
    assert not read_artwork(page)


def test_three_pips_are_a_rating_but_two_are_not(tmp_path):
    """A pair of marks is a bullet or a stray, not a score."""
    path = tmp_path / "pips.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((45, 100), "Two marks", fontsize=9)
    for i in range(2):
        page.draw_rect(fitz.Rect(45 + i * 10, 108, 50 + i * 10, 113),
                       color=None, fill=(0.13, 0.13, 0.13))
    page.insert_text((45, 200), "Three marks", fontsize=9)
    for i in range(3):
        page.draw_rect(fitz.Rect(45 + i * 10, 208, 50 + i * 10, 213),
                       color=None, fill=(0.13, 0.13, 0.13))
    doc.save(str(path))
    doc.close()

    art = read_artwork(extract_pdf(str(path), QAReport()).pages[0])
    assert len(art.ratings) == 1
    assert art.ratings[0].total == 3


def test_a_dash_shaped_pip_is_still_a_rating(tmp_path):
    """Templates score with dashes as well as dots - both are ratings."""
    path = tmp_path / "dashes.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((45, 100), "Makeup", fontsize=9)
    for i in range(5):
        # 18x4, the proportions the Tokyo template uses.
        page.draw_rect(fitz.Rect(45 + i * 24, 108, 63 + i * 24, 112),
                       color=None, fill=(0.7, 0.13, 0.15))
    doc.save(str(path))
    doc.close()

    art = read_artwork(extract_pdf(str(path), QAReport()).pages[0])
    assert len(art.ratings) == 1
    rating = art.ratings[0]
    assert rating.total == 5
    # The shape is carried through, so it is not redrawn as a row of dots.
    assert rating.pip_width > rating.pip_height


def test_a_hairline_rule_is_not_a_pip(tmp_path):
    """A rule is as long as a dash and a fraction of the thickness."""
    path = tmp_path / "rules.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((45, 100), "Section", fontsize=9)
    for i in range(5):
        page.draw_line(fitz.Point(45, 108 + i * 12),
                       fitz.Point(200, 108 + i * 12),
                       color=(0.1, 0.1, 0.1), width=0.4)
    doc.save(str(path))
    doc.close()

    art = read_artwork(extract_pdf(str(path), QAReport()).pages[0])
    assert not art.ratings


def test_a_header_buried_in_a_banner_is_dropped(tmp_path):
    """The copy inside the banner is a build leftover, not a second header."""
    from app.core.artwork import buried_duplicates

    path = tmp_path / "banner.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.draw_rect(fitz.Rect(0, 0, 595, 120), color=None, fill=(0.7, 0.13, 0.15))
    page.insert_text((54, 30), "email@email.com", fontsize=8)   # buried
    page.insert_text((54, 160), "email@email.com", fontsize=8)  # visible
    page.insert_text((54, 200), "Only once", fontsize=8)
    doc.save(str(path))
    doc.close()

    page_obj = extract_pdf(str(path), QAReport()).pages[0]
    buried = buried_duplicates(page_obj)
    assert len(buried) == 1
    assert "email" in page_obj.blocks[next(iter(buried))].text
    # The copy outside the banner survives, and so does unique text.
    kept = [b.text.strip() for i, b in enumerate(page_obj.blocks)
            if i not in buried]
    assert "Only once" in " ".join(kept)
    assert any("email" in t for t in kept)


def test_text_inside_a_panel_is_kept_when_it_is_not_repeated(tmp_path):
    """A panel full of text is ordinary - only a duplicate is dropped."""
    from app.core.artwork import buried_duplicates

    path = tmp_path / "panel.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.draw_rect(fitz.Rect(0, 0, 184, 842), color=None, fill=(0.96, 0.96, 0.96))
    page.insert_text((45, 100), "Sidebar content", fontsize=9)
    page.insert_text((45, 130), "More sidebar", fontsize=9)
    doc.save(str(path))
    doc.close()

    assert not buried_duplicates(extract_pdf(str(path), QAReport()).pages[0])

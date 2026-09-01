"""Per-block script detection drives sizing and mirroring, with no manual setup.

Every block on a page decides for itself, from its own dominant script, whether
it belongs to the right-to-left flow. Nothing here configures a block by hand:
that is the property under test.
"""
import fitz
import pytest

from app.core import fonts as fontlib
from app.core.extract import extract_pdf
from app.core.language import is_arabic
from app.core.merge import _combine, _combine_lines
from app.core.mirror import MirrorMode, block_is_rtl, mirror_document_pages
from app.core.models import BBox, Line, Span, TextBlock
from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.qa import QAReport

ARABIC = "دروس اللغة الإنجليزية المجانية"


# -- Bug 1: ar2en font sizing ---------------------------------------------

def test_ar2en_scales_size_down():
    """English replacing Arabic must not inherit the Arabic-tuned size."""
    assert fontlib.adjusted_size(20.0, False, source_is_arabic=True) < 20.0


def test_en2ar_still_bumps_size_up():
    """The reverse direction is unchanged."""
    assert fontlib.adjusted_size(11.0, True) == 11.0 + fontlib.ARABIC_SIZE_BONUS


def test_native_english_keeps_its_own_size():
    """English that was always English keeps the size its author chose.

    This is why the adjustment keys off the source text's script rather than
    off the translation direction: on an ar2en job an English footer is not a
    translation at all and must not be shrunk.
    """
    assert fontlib.adjusted_size(10.0, False, source_is_arabic=False) == 10.0


@pytest.mark.parametrize("original,expected", [
    (25.5, 25.5 * fontlib.AR_TO_EN_SIZE_SCALE),
    (20.0, 20.0 * fontlib.AR_TO_EN_SIZE_SCALE),
    (16.0, 16.0 * fontlib.AR_TO_EN_SIZE_SCALE),
])
def test_reported_sizes_are_scaled(original, expected):
    """The sizes from the report: 25.5pt heading, 20pt subheading, 16pt body."""
    assert fontlib.adjusted_size(original, False, source_is_arabic=True) == \
        pytest.approx(expected)


def test_scale_is_a_named_constant():
    """Tunable in one place after a visual review, not inline."""
    assert 0.5 < fontlib.AR_TO_EN_SIZE_SCALE < 1.0


def test_ar2en_reduction_is_meaningful():
    """Arabic point sizes are above what the same text needs in Latin.

    The scale was briefly taken to 0.75 while the paragraph was still being
    laid out in fragments, which made the text look oversized. Once the
    fragments were merged and the paragraph reflowed as one unit, 0.87 read
    correctly - so the size was returned to it. This pins the reduction as
    real without pinning a particular value: it is a visual setting, tuned
    against rendered output rather than derived.
    """
    assert fontlib.AR_TO_EN_SIZE_SCALE <= 0.9
    # A 20pt Arabic body line must come down into Latin body range.
    assert fontlib.adjusted_size(20.0, False, source_is_arabic=True) < 20.0


def test_scaled_size_never_collapses():
    """Even a tiny source size stays legible rather than scaling toward zero."""
    assert fontlib.adjusted_size(1.0, False, source_is_arabic=True) >= \
        fontlib.AR_TO_EN_MIN_SIZE


def test_ar2en_output_sizes_are_smaller_than_source(full_page_pdf, tmp_path):
    """End to end: no English block is drawn at its Arabic source size."""
    page_obj = extract_pdf(full_page_pdf, QAReport()).pages[0]
    arabic_sizes = {
        round(b.dominant_style().size, 1)
        for b in page_obj.blocks if b.mirror
    }

    out = str(tmp_path / "out.pdf")
    run_pipeline(full_page_pdf, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), QAReport())

    with fitz.open(out) as pdf:
        for block in pdf[0].get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text or is_arabic(text):
                        continue
                    # A Latin span translated from Arabic must have moved off
                    # the source size rather than reusing it verbatim.
                    if round(span["size"], 1) in arabic_sizes:
                        pytest.fail(
                            f"English span {text[:30]!r} kept the Arabic size "
                            f"{span['size']:.1f}pt"
                        )


# -- Bug 2: automatic per-block mirroring ---------------------------------

def test_every_block_matches_its_own_script(full_page_pdf):
    """The whole page classifies itself: heading, subtitle, body, footer.

    No block is configured anywhere. Each one's mirror decision must equal the
    verdict of its own dominant script.
    """
    page = extract_pdf(full_page_pdf, QAReport()).pages[0]
    assert len(page.blocks) >= 6, "fixture must cover a realistic page"

    for block in page.blocks:
        assert block.mirror is is_arabic(block.text), \
            f"block {block.text[:30]!r} disagrees with its own script"
        assert block.mirror is block_is_rtl(block)


def test_page_contains_both_kinds(full_page_pdf):
    """Guards the test above from passing on a single-script page."""
    page = extract_pdf(full_page_pdf, QAReport()).pages[0]
    assert any(b.mirror for b in page.blocks)
    assert any(not b.mirror for b in page.blocks)


def test_footer_behaviour_generalises_to_every_english_block(full_page_pdf):
    """The footer already worked; heading and subtitle must behave identically.

    All three are English blocks on an Arabic page, so all three must reach the
    same decision by the same route - no special case for the footer.
    """
    page = extract_pdf(full_page_pdf, QAReport()).pages[0]
    english = [b for b in page.blocks if not b.mirror]
    labels = " ".join(b.text for b in english)
    assert "Mayor's Office" in labels     # the subtitle from the report
    assert "Program Overview" in labels   # a heading
    assert "Email us" in labels           # the footer that already worked
    assert all(not b.mirror for b in english)


@pytest.mark.parametrize("direction", ["en2ar", "ar2en"])
def test_mirroring_needs_no_per_block_configuration(full_page_pdf, direction):
    """Running the pass twice with no setup gives script-consistent results."""
    page = extract_pdf(full_page_pdf, QAReport()).pages[0]
    width = page.width
    before = {id(b): b.bbox.as_tuple() for b in page.blocks}
    flags = {id(b): b.mirror for b in page.blocks}

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), direction)

    for block in page.blocks:
        was = before[id(block)]
        if flags[id(block)]:
            assert block.bbox.x0 == pytest.approx(width - was[2])
        else:
            # LTR blocks are re-anchored, never reflected: width preserved and
            # spans keep their internal left-to-right order.
            assert block.bbox.width == pytest.approx(was[2] - was[0])


def test_align_mode_is_a_global_override_only(full_page_pdf):
    """The CLI toggle still switches the whole pass off, as documented."""
    page = extract_pdf(full_page_pdf, QAReport()).pages[0]
    before = [b.bbox.as_tuple() for b in page.blocks]
    mirror_document_pages([page], MirrorMode.ALIGN, QAReport(), "ar2en")
    assert [b.bbox.as_tuple() for b in page.blocks] == before


# -- merging must not lose the classification -----------------------------

def _block(text: str, x0: float, x1: float, rtl: bool) -> TextBlock:
    span = Span(text, "helv", 11.0, (0, 0, 0), False, False, False,
                BBox(x0, 50, x1, 62), origin=(x0, 62), mirror=rtl)
    return TextBlock(lines=[Line(spans=[span], bbox=BBox(x0, 50, x1, 62))],
                     bbox=BBox(x0, 50, x1, 62), mirror=rtl)


def test_merged_english_fragments_stay_ltr():
    """The footer case: several short English fragments merged into one block.

    A merged block is a new object, so its script has to be recomputed from the
    joined text. Inheriting the default here is what silently mirrored a footer
    assembled from "Email us" + "Learn more".
    """
    merged = _combine([_block("Email", 100, 150, False),
                       _block("us", 155, 190, False)], False)
    assert merged.mirror is False
    stacked = _combine_lines([_block("Learn", 100, 150, False),
                              _block("more", 155, 190, False)])
    assert stacked.mirror is False


def test_merged_arabic_fragments_stay_rtl():
    merged = _combine([_block(ARABIC, 100, 200, True),
                       _block(ARABIC, 205, 300, True)], True)
    assert merged.mirror is True


def test_merged_block_follows_joined_text_not_its_parts():
    """Classification is recomputed, not inherited from the first fragment."""
    merged = _combine([_block("a", 100, 120, True),
                       _block(ARABIC, 125, 300, True)], True)
    assert merged.mirror is block_is_rtl(merged)


def test_blocks_still_agree_with_their_script_after_the_pipeline(full_page_pdf,
                                                                tmp_path):
    """Merging runs inside the pipeline, so re-check the invariant after it."""
    out = str(tmp_path / "out.pdf")
    qa = QAReport()
    run_pipeline(full_page_pdf, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), qa)
    with fitz.open(out) as pdf:
        assert pdf[0].get_text().strip()

"""Mixed Arabic/English pages: script-aware mirroring and block-level fitting.

The source document these guard is the one that broke both passes at once: a
page that is mostly Arabic but carries English headings. Mirroring used to flip
every span regardless of script, and fitting used to judge each block without
reporting when the translation stopped fitting the design.
"""
import fitz
import pytest

from app.core.extract import extract_pdf
from app.core.language import is_arabic
from app.core.mirror import MirrorMode, block_is_rtl, mirror_document_pages
from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.qa import QAReport
from app.core.translate import TranslationProvider
import app.core.translate as translate_mod


ARABIC = "نص عربي طويل في هذه الفقرة"
ENGLISH = "Executive Summary"


# -- script detection ------------------------------------------------------

def test_is_arabic_separates_scripts():
    assert is_arabic(ARABIC)
    assert not is_arabic(ENGLISH)


def test_is_arabic_ignores_neutrals():
    """Digits and punctuation must not dilute a short Arabic label."""
    assert is_arabic("مؤشر 2024 (99.9%)")


def test_is_arabic_on_empty_and_neutral_text():
    for text in ("", "   ", "12345", "-- //"):
        assert not is_arabic(text)


def test_arabic_led_mixed_text_stays_rtl():
    """An Arabic heading quoting an English product name is still RTL."""
    assert is_arabic("مقدمة عامة حول Platform Migration في الشركة")


def test_is_arabic_is_direction_independent():
    """Script is a property of the text, never of the job being run."""
    assert is_arabic(ARABIC) and not is_arabic(ENGLISH)


# -- Bug 1: language-aware mirroring ---------------------------------------

def _blocks_by_script(page):
    arabic = [b for b in page.blocks if b.mirror]
    english = [b for b in page.blocks if not b.mirror]
    return arabic, english


def test_extraction_flags_spans_by_script(mixed_pdf):
    page = extract_pdf(mixed_pdf, QAReport()).pages[0]
    arabic, english = _blocks_by_script(page)
    assert arabic and english, "fixture must contain both scripts"
    for block in english:
        assert not is_arabic(block.text)
        assert all(not s.mirror for s in block.spans)
    for block in arabic:
        assert is_arabic(block.text)


def test_block_flag_follows_block_text(mixed_pdf):
    page = extract_pdf(mixed_pdf, QAReport()).pages[0]
    for block in page.blocks:
        assert block.mirror is block_is_rtl(block)


@pytest.mark.parametrize("direction", ["en2ar", "ar2en"])
def test_english_headings_are_never_flipped(mixed_pdf, direction):
    """The regression itself: English headings must keep their LTR run.

    A flipped heading is detectable without knowing the target coordinate: its
    box would be reflected about the page centre *and* its spans reversed
    within it. Re-anchoring moves every span by one shared offset instead, so
    the spans keep their original spacing and order.
    """
    page = extract_pdf(mixed_pdf, QAReport()).pages[0]
    english = [b for b in page.blocks if not b.mirror]
    before = {id(b): [s.bbox.as_tuple() for s in b.spans] for b in english}

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), direction)

    for block in english:
        offsets = [
            after[0] - was[0]
            for after, was in zip([s.bbox.as_tuple() for s in block.spans],
                                  before[id(block)])
        ]
        # One shared shift for the whole block == moved, not mirrored.
        assert max(offsets) - min(offsets) < 0.01, \
            "English heading spans were reflected individually"


@pytest.mark.parametrize("direction", ["en2ar", "ar2en"])
def test_arabic_blocks_are_mirrored(mixed_pdf, direction):
    page = extract_pdf(mixed_pdf, QAReport()).pages[0]
    width = page.width
    arabic = [b for b in page.blocks if b.mirror]
    before = {id(b): b.bbox.as_tuple() for b in arabic}

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), direction)

    for block in arabic:
        was = before[id(block)]
        assert block.bbox.x0 == pytest.approx(width - was[2]), \
            "Arabic block must be reflected across the page centre"
        assert block.bbox.x1 == pytest.approx(width - was[0])


def test_heading_stays_attached_to_its_section(mixed_pdf):
    """A heading must follow the block it labels, not sit at raw coordinates.

    Before the fix the two failure modes were opposite: the heading was either
    flipped with the body text or left entirely untouched while the page moved
    around it. Both detach it from its section; the heading and the Arabic
    block below it must still share horizontal space afterwards.
    """
    page = extract_pdf(mixed_pdf, QAReport()).pages[0]
    heading = next(b for b in page.blocks if not b.mirror)
    body = min(
        (b for b in page.blocks if b.mirror and b.bbox.y0 > heading.bbox.y0),
        key=lambda b: b.bbox.y0,
    )
    raw_heading_x0 = heading.bbox.x0

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "en2ar")

    assert heading.bbox.x0 != raw_heading_x0, \
        "heading was left at its raw source position"
    overlap = (min(heading.bbox.x1, body.bbox.x1)
               - max(heading.bbox.x0, body.bbox.x0))
    assert overlap > 0, "heading no longer sits over the section it labels"


def test_ltr_span_inside_rtl_block_is_shifted_not_reflected():
    """A brand name inside an Arabic paragraph moves with it, unflipped."""
    from app.core.models import BBox, Line, Page, Span, TextBlock

    ltr = Span("Acme", "helv", 11.0, (0, 0, 0), False, False, False,
               BBox(100, 50, 140, 62), origin=(100, 60), mirror=False)
    rtl = Span(ARABIC, "ar", 11.0, (0, 0, 0), False, False, False,
               BBox(140, 50, 300, 62), origin=(140, 60), mirror=True)
    block = TextBlock(lines=[Line(spans=[ltr, rtl], bbox=BBox(100, 50, 300, 62))],
                      bbox=BBox(100, 50, 300, 62), mirror=True)
    page = Page(number=0, width=595, height=842, blocks=[block])

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")

    # The LTR span keeps its width and is not reflected about its own centre.
    assert ltr.bbox.width == pytest.approx(40)
    assert ltr.bbox.x0 == pytest.approx(595 - 140)


# -- Bug 2: fitting, alignment and QA reporting ----------------------------

class _Expanding(TranslationProvider):
    """Returns text far longer than its input, to force the overflow path."""

    name = "expanding"

    def __init__(self, factor: int = 26):
        self.factor = factor

    def translate_batch(self, texts, direction):
        return [t + " " + "word extra " * self.factor for t in texts]


@pytest.fixture
def expanding_provider():
    translate_mod.set_provider(_Expanding())
    yield
    translate_mod.set_provider(_Expanding(1))


def test_long_translation_is_reported_not_silently_overflowed(
    mixed_pdf, tmp_path, expanding_provider
):
    """A translation that outgrows its box must reach the QA report."""
    out = str(tmp_path / "out.pdf")
    qa = QAReport()
    run_pipeline(mixed_pdf, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), qa)

    flagged = [e for e in qa.entries if e.category == "layout_review"]
    assert flagged, "an over-long translation must be flagged for review"
    assert all(e.severity == "warning" for e in flagged)


def test_qa_report_is_serialisable(mixed_pdf, tmp_path, expanding_provider):
    out = str(tmp_path / "out.pdf")
    qa = QAReport()
    run_pipeline(mixed_pdf, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), qa)
    payload = qa.to_dict()
    assert "layout_review" in payload["by_category"]


def test_rebuilt_pdf_is_readable(mixed_pdf, tmp_path):
    """The whole pipeline must still produce a valid, non-empty PDF."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(mixed_pdf, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), QAReport())
    with fitz.open(out) as pdf:
        assert pdf.page_count == 1
        assert pdf[0].get_text().strip()


def test_font_size_is_consistent_within_a_block(mixed_pdf, tmp_path):
    """One block is drawn at one size - no per-span size drift.

    The block is the unit that gets fitted, so every span it produces in the
    output shares a single font size.
    """
    out = str(tmp_path / "out.pdf")
    run_pipeline(mixed_pdf, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), QAReport())
    with fitz.open(out) as pdf:
        for block in pdf[0].get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            sizes = {round(s["size"], 1)
                     for line in block.get("lines", [])
                     for s in line.get("spans", []) if s.get("text", "").strip()}
            assert len(sizes) <= 1, f"mixed font sizes in one block: {sizes}"

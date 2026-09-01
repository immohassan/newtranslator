"""Regressions for layout defects found while building the PDF rebuild."""
import fitz
import pytest

from app.core import fonts as fontlib
from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.qa import QAReport
from app.core.shape_arabic import contains_arabic, shape


@pytest.fixture
def multipage_pdf(tmp_path):
    """Portrait page, an empty page, and a landscape page of a different width."""
    path = str(tmp_path / "multi.pdf")
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((72, 100), "Quarterly Report", fontname="hebo", fontsize=20)
    doc.new_page(width=595, height=842)
    p3 = doc.new_page(width=842, height=595)
    p3.insert_text((72, 100), "Key Results", fontname="helv", fontsize=14)
    doc.save(path)
    doc.close()
    return path


def test_arabic_line_height_exceeds_latin():
    """Naskh needs far more vertical room than Latin at the same point size -
    the reason source boxes must grow before the text is shrunk."""
    arabic = fontlib._locate("NotoNaskhArabic-Regular.ttf")
    latin = fontlib._locate("DejaVuSans.ttf")
    assert fontlib.line_height(arabic, 15) > fontlib.line_height(latin, 15) * 1.3


def test_short_heading_does_not_wrap(multipage_pdf, tmp_path):
    """A heading that is only slightly too wide must widen its box instead of
    wrapping: PyMuPDF stacks wrapped lines in logical order, which scrambles
    already-bidi'd right-to-left text."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(multipage_pdf, out, TranslationOptions("en2ar", mirror=True, html_engine=False))

    with fitz.open(out) as pdf:
        page = pdf[2]
        blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        arabic = [b for b in blocks
                  if contains_arabic("".join(s["text"] for l in b["lines"]
                                             for s in l["spans"]))]
        assert arabic
        for block in arabic:
            assert len(block["lines"]) == 1, "the heading must stay on one line"


def test_translated_heading_reads_in_order(multipage_pdf, tmp_path):
    """The heading must come back with its words in the original order.

    PyMuPDF re-reverses visual-order text when extracting, so the round trip
    lands back on the logical string. A wrapped heading would instead come back
    with the two words swapped.
    """
    out = str(tmp_path / "out.pdf")
    run_pipeline(multipage_pdf, out, TranslationOptions("en2ar", mirror=True, html_engine=False))
    with fitz.open(out) as pdf:
        text = pdf[2].get_text().strip().replace("\xa0", " ")
    words = text.split()
    assert len(words) == 2, f"expected two words, got {words}"
    assert words[0].startswith("ﺍﻟﻨ"), "the first word must still be first"


def test_empty_page_survives(multipage_pdf, tmp_path):
    out = str(tmp_path / "out.pdf")
    run_pipeline(multipage_pdf, out, TranslationOptions("en2ar", mirror=True, html_engine=False))
    with fitz.open(out) as pdf:
        assert pdf.page_count == 3
        assert pdf[1].get_text().strip() == ""


def test_landscape_page_mirrors_on_its_own_width(multipage_pdf, tmp_path):
    """Each page mirrors around its own width, not the document's first page."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(multipage_pdf, out, TranslationOptions("en2ar", mirror=True, html_engine=False))
    with fitz.open(out) as pdf:
        page = pdf[2]
        assert page.rect.width == pytest.approx(842, abs=1)
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") == 0:
                assert block["bbox"][2] <= page.rect.width + 1
                assert block["bbox"][2] > page.rect.width / 2, "must be right-aligned"


def test_no_content_drawn_off_any_page(multipage_pdf, tmp_path):
    out = str(tmp_path / "out.pdf")
    run_pipeline(multipage_pdf, out, TranslationOptions("en2ar", mirror=True, html_engine=False))
    with fitz.open(out) as pdf:
        for page in pdf:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                assert block["bbox"][0] >= -1
                assert block["bbox"][2] <= page.rect.width + 1


def test_stale_underline_is_erased(sample_pdf, tmp_path):
    """Vector rules are not removed by redaction, so they must be painted over -
    otherwise the original underline stays behind on the mirrored page."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(sample_pdf, out, TranslationOptions("en2ar", mirror=True, html_engine=False))

    with fitz.open(sample_pdf) as src:
        original = [d for d in src[0].get_drawings()
                    if d["rect"].width > 100 and d["rect"].height < 3]
    assert original, "the fixture must contain a wide thin rule"
    original_rect = original[0]["rect"]

    with fitz.open(out) as pdf:
        # Nothing dark should remain where the original underline was drawn.
        pix = pdf[0].get_pixmap(clip=fitz.Rect(
            original_rect.x0, original_rect.y0 - 1,
            original_rect.x1, original_rect.y1 + 1))
        darkest = min(pix.pixel(x, y)[2]
                      for x in range(pix.width) for y in range(pix.height))
    assert darkest > 200, "the original underline was not cleared"


def test_directional_image_content_is_flipped(sample_pdf, tmp_path):
    """An arrow that pointed right must point left after mirroring."""
    plain = str(tmp_path / "plain.pdf")
    flipped = str(tmp_path / "flipped.pdf")
    run_pipeline(sample_pdf, plain,
                 TranslationOptions("en2ar", mirror=True,
                                    flip_directional_images=False, html_engine=False))
    run_pipeline(sample_pdf, flipped,
                 TranslationOptions("en2ar", mirror=True,
                                    flip_directional_images=True, html_engine=False))

    def arrow_pixels(path):
        with fitz.open(path) as pdf:
            page = pdf[0]
            rect = page.get_image_rects(page.get_images(full=True)[0][0])[0]
            return page.get_pixmap(clip=rect, dpi=72)

    a, b = arrow_pixels(plain), arrow_pixels(flipped)
    assert a.samples != b.samples, "flipping must change the image content"


def test_underline_matches_translated_width(sample_pdf, tmp_path):
    """The redrawn underline tracks the translated text, not the original."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(sample_pdf, out, TranslationOptions("en2ar", mirror=True, html_engine=False))
    with fitz.open(out) as pdf:
        rules = [d for d in pdf[0].get_drawings()
                 if d["rect"].width > 20 and d["rect"].height < 3]
    assert rules, "an underline must be redrawn for the underlined heading"


def test_underline_can_be_disabled(sample_pdf, tmp_path):
    with_ul = str(tmp_path / "with.pdf")
    without = str(tmp_path / "without.pdf")
    run_pipeline(sample_pdf, with_ul,
                 TranslationOptions("en2ar", mirror=True, underline=True, html_engine=False))
    run_pipeline(sample_pdf, without,
                 TranslationOptions("en2ar", mirror=True, underline=False, html_engine=False))

    def rule_count(path):
        with fitz.open(path) as pdf:
            return len([d for d in pdf[0].get_drawings()
                        if d["rect"].width > 20 and d["rect"].height < 3])

    assert rule_count(without) < rule_count(with_ul)


# --- multi-line Arabic: wrap in logical order, shape per line ---------------
def test_shaped_arabic_is_narrower_than_logical():
    """Joined presentation forms are much narrower than unjoined letters, so
    wrapping must measure the shaped form or lines break far too early."""
    from app.core.shape_arabic import shape

    path = fontlib._locate("NotoNaskhArabic-Regular.ttf")
    text = "النتائج الرئيسية الهامة"
    assert fontlib.measure(shape(text), path, 15) < fontlib.measure(text, path, 15)


def test_wrap_lines_measures_shaped_form():
    from app.core.rebuild_pdf import _wrap_lines

    path = fontlib._locate("NotoNaskhArabic-Regular.ttf")
    text = "أكمل فريق الهندسة نقل المنصة قبل الموعد المحدد بوقت كاف"
    logical = _wrap_lines(text, path, 12, 150, shaped=False)
    rendered = _wrap_lines(text, path, 12, 150, shaped=True)
    assert len(rendered) <= len(logical), \
        "measuring the logical string wraps too aggressively"


def test_shape_lines_preserves_line_order():
    """Each line is shaped independently; the sequence of lines is untouched."""
    from app.core.shape_arabic import shape, shape_lines

    lines = ["السطر الأول", "السطر الثاني"]
    out = shape_lines(lines).split("\n")
    assert out == [shape(lines[0]), shape(lines[1])]


def test_long_arabic_paragraph_keeps_all_words(sample_pdf, tmp_path):
    """A wrapped paragraph must render every word exactly once.

    The bug this covers: shaping the whole paragraph and letting the renderer
    wrap it afterwards moved the paragraph's tail onto the first line.
    """
    from app.core import translate as T
    from app.core.shape_arabic import shape

    long_ar = ("أكمل فريق الهندسة نقل المنصة قبل الموعد المحدد وتحسنت موثوقية "
               "النظام وانخفضت التكلفة الإجمالية للتشغيل خلال فترة التقرير")

    class OneParagraph(T.TranslationProvider):
        name = "one-paragraph"

        def translate_batch(self, texts, direction):
            # Only the body block is long enough to wrap; leave the rest alone
            # so nothing else competes for vertical space.
            return [long_ar if len(t) > 60 else t for t in texts]

    T.set_provider(OneParagraph())
    out = str(tmp_path / "out.pdf")
    qa = run_pipeline(sample_pdf, out, TranslationOptions("en2ar", mirror=True, html_engine=False))

    # PyMuPDF reports each rendered line as its own block, so collect the lines
    # of the wrapped paragraph in the order they were drawn (top to bottom).
    with fitz.open(out) as pdf:
        lines = []
        for block in pdf[0].get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                text = "".join(s["text"] for s in line["spans"])
                lines.append((line["bbox"][1], text))
    lines.sort()

    # PyMuPDF returns joined presentation forms without re-applying bidi, so
    # match on the reshaped word rather than the logical or fully shaped one.
    from app.core.shape_arabic import _RESHAPER

    def rendered_form(word):
        return _RESHAPER.reshape(word)

    paragraph = [t for _, t in lines]
    head = next((i for i, t in enumerate(paragraph)
                 if rendered_form(long_ar.split()[0]) in t), None)
    tail = next((i for i, t in enumerate(paragraph)
                 if rendered_form(long_ar.split()[-1]) in t), None)
    assert head is not None, "the paragraph's first word was not rendered"
    assert tail is not None, "the paragraph's last word was not rendered"
    assert head < tail, \
        "the paragraph's tail was rendered above its head - lines were shaped " \
        "before wrapping"

    assert not [e for e in qa.entries if e.category == "text_clipped"]


def test_oversized_translation_is_never_dropped(sample_pdf, tmp_path):
    """A translation far longer than its box must still be drawn.

    Growth is bounded by the neighbouring block, but when nothing fits inside
    that ceiling the block overflows rather than vanishing - losing a heading
    silently is the worst outcome - and QA reports the shortfall.
    """
    from app.core import translate as T

    class TooLong(T.TranslationProvider):
        name = "too-long"

        def translate_batch(self, texts, direction):
            return ["التقرير الفصلي الشامل للنتائج والإنجازات" for _ in texts]

    T.set_provider(TooLong())
    out = str(tmp_path / "out.pdf")
    qa = run_pipeline(sample_pdf, out, TranslationOptions("en2ar", mirror=True, html_engine=False))

    with fitz.open(out) as pdf:
        page = pdf[0]
        blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        width = page.rect.width

    # Every source block still reaches the page...
    assert len(blocks) >= 5, "no block may be silently dropped"
    # ...and nothing is drawn off the edge of it.
    for block in blocks:
        assert block["bbox"][0] >= -1
        assert block["bbox"][2] <= width + 1

    # Whatever compromises were needed are reported, and nothing is recorded
    # as clipped because no content was lost.
    assert not [e for e in qa.entries if e.category == "text_clipped"]


# --- vector artwork (logos, icons) -----------------------------------------
@pytest.fixture
def logo_pdf(tmp_path):
    """A page whose 'logo' is several curved paths, as real branding is."""
    path = str(tmp_path / "logo.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Three glyph-like blobs in a row, drawn consecutively - one logo.
    for i in range(3):
        x = 400 + i * 30
        shape = page.new_shape()
        shape.draw_bezier(fitz.Point(x, 60), fitz.Point(x + 10, 40),
                          fitz.Point(x + 20, 80), fitz.Point(x + 25, 60))
        shape.finish(fill=(0, 0.3, 0.7), color=None)
        shape.commit()
    page.insert_text((72, 200), "Quarterly Report", fontname="hebo", fontsize=20)
    doc.save(path)
    doc.close()
    return path


def test_curved_paths_are_classified_as_artwork(logo_pdf):
    """Curved paths must not be treated as rules or filled rectangles - that is
    what turned every logo on the page into a solid block."""
    from app.core.extract import extract_pdf

    doc = extract_pdf(logo_pdf, QAReport())
    art = [d for d in doc.pages[0].drawings if d.kind == "path"]
    assert art, "bezier artwork must be classified as 'path'"
    assert all(d.items for d in art), "path segments must be preserved"


def test_artwork_is_replayed_not_boxed(logo_pdf, tmp_path):
    """The rebuilt page must contain curves, not a filled bounding box."""
    from app.core import translate as T

    T.set_provider(T.EchoProvider())
    out = str(tmp_path / "out.pdf")
    run_pipeline(logo_pdf, out, TranslationOptions("en2ar", mirror=True, html_engine=False))

    with fitz.open(out) as pdf:
        curves = [d for d in pdf[0].get_drawings()
                  if any(i[0] == "c" for i in d["items"])]
    assert curves, "the logo's curves must survive the rebuild"


def test_logo_parts_keep_their_order_when_mirrored(logo_pdf):
    """A logo is one path per glyph. Mirroring each around the page centre
    individually reverses their order and the logo reads backwards, so the
    group moves as a unit.
    """
    from app.core.extract import extract_pdf
    from app.core.mirror import MirrorMode, mirror_document_pages

    doc = extract_pdf(logo_pdf, QAReport())
    page = doc.pages[0]
    art = sorted([d for d in page.drawings if d.kind == "path"],
                 key=lambda d: d.bbox.x0)
    before = [d.bbox.x0 for d in art]
    widths = [d.bbox.width for d in art]

    mirror_document_pages([page], MirrorMode.FULL, QAReport())

    after = [d.bbox.x0 for d in art]
    # Same left-to-right sequence...
    assert after == sorted(after), "logo parts were reordered by mirroring"
    # ...same internal spacing, same sizes, just moved across the page.
    assert [round(w, 2) for w in widths] == \
           [round(d.bbox.width, 2) for d in art]
    gaps_before = [round(b - a, 2) for a, b in zip(before, before[1:])]
    gaps_after = [round(b - a, 2) for a, b in zip(after, after[1:])]
    assert gaps_before == gaps_after, "the logo's internal spacing changed"
    assert after[0] != before[0], "the logo should have moved"


def test_source_lines_that_already_overlap_do_not_collide(tmp_path):
    """Tightly-led source lines overlap slightly. Growing each box to the
    Arabic line height must not let one line print over the next."""
    from app.core import translate as T

    path = str(tmp_path / "tight.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for i in range(3):
        page.insert_text((72, 200 + i * 25), f"Tight leading line {i}",
                         fontname="helv", fontsize=16)
    doc.save(path)
    doc.close()

    T.set_provider(T.EchoProvider())
    out = str(tmp_path / "out.pdf")
    qa = run_pipeline(path, out, TranslationOptions("en2ar", mirror=True, html_engine=False))

    # The three source lines form one paragraph and are now laid out as a
    # single block, so count rendered *lines* rather than blocks.
    with fitz.open(out) as pdf:
        lines = [l for b in pdf[0].get_text("dict")["blocks"]
                 if b.get("type") == 0 for l in b["lines"]
                 if "".join(s["text"] for s in l["spans"]).strip()]
    assert len(lines) >= 3, "no line may be dropped"

    # Lines of one paragraph must not sit on top of each other.
    tops = sorted(l["bbox"][1] for l in lines)
    for a, b in zip(tops, tops[1:]):
        assert b - a > 1, "paragraph lines overlap"

    assert not [e for e in qa.entries if e.category == "text_clipped"]


# --- paragraph line merging ------------------------------------------------
@pytest.fixture
def split_paragraph_pdf(tmp_path):
    """One paragraph emitted as three separate blocks, as designed PDFs do."""
    path = str(tmp_path / "split.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for i, line in enumerate([
        "Each center offers free in-person English lessons for",
        "beginner and intermediate levels for immigrants,",
        "regardless of immigration status or current level.",
    ]):
        page.insert_text((72, 300 + i * 25), line, fontname="helv", fontsize=16)
    doc.save(path)
    doc.close()
    return path


def test_paragraph_lines_are_merged(split_paragraph_pdf):
    """Consecutive lines of one paragraph must become a single block, or each
    line is fitted separately and ends up at its own size and indent."""
    from app.core.extract import extract_pdf
    from app.core.merge import merge_paragraph_lines

    doc = extract_pdf(split_paragraph_pdf, QAReport())
    page = doc.pages[0]
    assert len(page.blocks) == 3, "the fixture must start as three blocks"

    merged = merge_paragraph_lines(page, "en2ar")
    assert merged == 1
    assert len(page.blocks) == 1
    assert len(page.blocks[0].lines) == 3


def test_distant_blocks_are_not_merged(tmp_path):
    """A paragraph must not swallow the next heading or list below it."""
    from app.core.extract import extract_pdf
    from app.core.merge import merge_paragraph_lines

    path = str(tmp_path / "spaced.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 300), "First paragraph line.", fontname="helv", fontsize=16)
    page.insert_text((72, 325), "Second line of it.", fontname="helv", fontsize=16)
    page.insert_text((72, 420), "A separate heading", fontname="helv", fontsize=16)
    doc.save(path)
    doc.close()

    extracted = extract_pdf(path, QAReport())
    merge_paragraph_lines(extracted.pages[0], "en2ar")

    texts = [b.text for b in extracted.pages[0].blocks]
    assert any("separate heading" in t and "First paragraph" not in t
               for t in texts), "a distant block was merged into the paragraph"


def test_merged_paragraph_renders_at_one_size(split_paragraph_pdf, tmp_path):
    """The whole point: one paragraph, one font size, one left edge."""
    from app.core import translate as T

    T.set_provider(T.EchoProvider())
    out = str(tmp_path / "out.pdf")
    run_pipeline(split_paragraph_pdf, out, TranslationOptions("en2ar", mirror=False, html_engine=False))

    with fitz.open(out) as pdf:
        lines = [l for b in pdf[0].get_text("dict")["blocks"]
                 if b.get("type") == 0 for l in b["lines"]
                 if "".join(s["text"] for s in l["spans"]).strip()]

    sizes = {round(l["spans"][0]["size"], 1) for l in lines}
    assert len(sizes) == 1, f"paragraph rendered at several sizes: {sizes}"

    lefts = {round(l["bbox"][0]) for l in lines}
    assert max(lefts) - min(lefts) <= 2, f"inconsistent indents: {lefts}"


def test_merged_paragraph_lines_are_evenly_spaced(split_paragraph_pdf, tmp_path):
    from app.core import translate as T

    T.set_provider(T.EchoProvider())
    out = str(tmp_path / "out.pdf")
    run_pipeline(split_paragraph_pdf, out, TranslationOptions("en2ar", mirror=False, html_engine=False))

    with fitz.open(out) as pdf:
        tops = sorted(l["bbox"][1] for b in pdf[0].get_text("dict")["blocks"]
                      if b.get("type") == 0 for l in b["lines"]
                      if "".join(s["text"] for s in l["spans"]).strip())

    gaps = [round(b - a, 1) for a, b in zip(tops, tops[1:])]
    assert gaps, "the paragraph must wrap onto several lines"
    assert max(gaps) - min(gaps) <= 1.5, f"uneven line spacing: {gaps}"

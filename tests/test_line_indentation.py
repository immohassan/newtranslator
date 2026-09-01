"""Left edge alignment: every line of a block starts at the same x.

Source PDFs indent lines with literal spaces rather than by moving the text
cursor. Those spaces survive extraction and translation, and insert_textbox
honours them, so the redrawn paragraph comes out with a ragged left edge.

The important detail is that `line["bbox"].x0` does NOT reveal this - the line
box includes the leading space, so every line reports the same x0 while the
visible glyphs sit several points apart. These tests measure the first
non-space glyph's origin instead.
"""
import fitz
import pytest

from app.core.merge import _combine
from app.core.models import BBox, Line, Span, TextBlock
from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.qa import QAReport
from app.core.shape_arabic import normalize_block_text

LATIN_FONT = "fonts/DejaVuSans-Bold.ttf"
# The heading from the reported screenshot, with its original indentation.
RAGGED = "Free English lessons\n for immigrants from New\nYork residents!"


def _first_glyph_x(path: str) -> list[float]:
    """x origin of the first visible glyph on each rendered line."""
    xs: list[float] = []
    with fitz.open(path) as pdf:
        for page in pdf:
            for block in page.get_text("rawdict")["blocks"]:
                for line in block.get("lines", []):
                    visible = [c for s in line["spans"] for c in s["chars"]
                               if not c["c"].isspace()]
                    if visible:
                        xs.append(round(visible[0]["origin"][0], 2))
    return xs


def _render(tmp_path, text: str, name: str = "out.pdf") -> str:
    out = str(tmp_path / name)
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_font(fontname="dj", fontfile=LATIN_FONT)
    page.insert_textbox(fitz.Rect(20, 20, 380, 150), text,
                        fontname="dj", fontfile=LATIN_FONT, fontsize=14)
    doc.save(out)
    doc.close()
    return out


# -- the normaliser --------------------------------------------------------

def test_leading_space_is_stripped_per_line():
    assert normalize_block_text(RAGGED) == (
        "Free English lessons\nfor immigrants from New\nYork residents!"
    )


def test_trailing_space_is_stripped_per_line():
    assert normalize_block_text("Line one \nLine two  ") == "Line one\nLine two"


def test_internal_runs_are_collapsed():
    """Two spaces between words render as a visible gap after a merge."""
    assert normalize_block_text("The Mayor's  Office") == "The Mayor's Office"


def test_blank_lines_are_preserved():
    """Paragraph breaks are structure, not stray whitespace."""
    assert normalize_block_text("one\n\ntwo") == "one\n\ntwo"


def test_normaliser_is_idempotent():
    once = normalize_block_text(RAGGED)
    assert normalize_block_text(once) == once


def test_normaliser_is_safe_on_clean_text():
    for text in ("", "already clean", "two\nlines"):
        assert normalize_block_text(text) == text


def test_nbsp_indent_is_also_stripped():
    """A no-break space indents just as visibly as an ordinary one."""
    assert normalize_block_text("first\n second") == "first\nsecond"


# -- rendered output -------------------------------------------------------

def test_ragged_text_renders_with_uneven_left_edge(tmp_path):
    """Pins the defect itself, so the fix below is testing something real."""
    xs = _first_glyph_x(_render(tmp_path, RAGGED, "ragged.pdf"))
    assert len(set(xs)) > 1, "expected a ragged left edge before normalising"


def test_normalised_text_renders_flush_left(tmp_path):
    xs = _first_glyph_x(
        _render(tmp_path, normalize_block_text(RAGGED), "flush.pdf")
    )
    assert len(set(xs)) == 1, f"left edge is still ragged: {xs}"


def test_line_bbox_hides_the_problem(tmp_path):
    """Why this is measured by glyph origin, not by the line box.

    The line's bbox starts at the text box edge whether or not the line begins
    with a space, so a bbox-based assertion would pass on ragged output.
    """
    path = _render(tmp_path, RAGGED, "bbox.pdf")
    with fitz.open(path) as pdf:
        box_xs = {round(line["bbox"][0], 2)
                  for block in pdf[0].get_text("dict")["blocks"]
                  for line in block.get("lines", [])}
    assert len(box_xs) == 1, "line boxes agree even when the glyphs do not"
    assert len(set(_first_glyph_x(path))) > 1, "glyphs actually disagree"


# -- merging must not introduce a double space -----------------------------

def _block(text: str, x0: float, x1: float) -> TextBlock:
    span = Span(text, "helv", 11.0, (0, 0, 0), False, False, False,
                BBox(x0, 50, x1, 62), origin=(x0, 62), mirror=False)
    return TextBlock(lines=[Line(spans=[span], bbox=BBox(x0, 50, x1, 62))],
                     bbox=BBox(x0, 50, x1, 62), mirror=False)


def test_merge_does_not_double_space_fragments():
    """The next fragment often carries its own leading space already."""
    merged = _combine([_block("The Mayor's", 100, 180),
                       _block(" Office", 185, 240)], False)
    assert "  " not in merged.text


def test_merge_still_separates_fragments():
    merged = _combine([_block("Email", 100, 150),
                       _block("us", 155, 190)], False)
    assert "Emailus" not in merged.text


# -- end to end ------------------------------------------------------------

def test_rebuilt_page_has_no_indented_lines(full_page_pdf, tmp_path):
    """No block in a real rebuilt page starts a line with whitespace."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(full_page_pdf, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), QAReport())
    with fitz.open(out) as pdf:
        for block in pdf[0].get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"])
                if text.strip():
                    assert text == text.lstrip(), \
                        f"line renders with a leading space: {text[:40]!r}"


def test_multiline_block_is_flush_left_after_rebuild(full_page_pdf, tmp_path):
    """Every wrapped line of a block shares one left edge."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(full_page_pdf, out,
                 TranslationOptions(direction="ar2en", mirror=True, html_engine=False), QAReport())
    with fitz.open(out) as pdf:
        for block in pdf[0].get_text("rawdict")["blocks"]:
            if block.get("type") != 0:
                continue
            lefts = []
            for line in block.get("lines", []):
                visible = [c for s in line["spans"] for c in s["chars"]
                           if not c["c"].isspace()]
                if visible:
                    lefts.append(round(visible[0]["origin"][0], 1))
            if len(lefts) < 2:
                continue
            # Every line but the last must share the block's left edge. The
            # final line of a paragraph is short and, in a right-aligned Arabic
            # block, legitimately starts further right - so it is excluded
            # rather than being allowed to slacken the check on the others.
            edge = min(lefts)
            for x in lefts[:-1]:
                assert x == pytest.approx(edge, abs=0.5), \
                    f"line starts {x - edge:.1f}pt in from the block edge: {lefts}"


# -- ragged fragments of one paragraph are merged --------------------------

def _line_block(text, x0, y0, x1, y1, size=20.0, bold=True):
    from app.core.models import BBox as B, Line as L, Span as S, TextBlock as TB
    span = S(text, "helv", size, (0, 0, 0), bold, False, False,
             B(x0, y0, x1, y1), origin=(x0, y1))
    return TB(lines=[L(spans=[span], bbox=B(x0, y0, x1, y1))],
              bbox=B(x0, y0, x1, y1))


def test_lines_sharing_no_edge_still_merge():
    """A line carved up by a gloss lines up with nothing, yet is a line.

    Where a parenthetical or a differently-sized number is lifted out of the
    middle of a line, the remainder keeps neither the left nor the right edge
    of its neighbours. Laid out separately each fragment is fitted on its own,
    which is what produced the ragged left edge in the reported output.
    """
    from app.core.merge import merge_paragraph_lines
    from app.core.models import Page

    page = Page(number=0, width=612, height=792, blocks=[
        _line_block("يمتلك مكتب العمدة لشؤون المهاجرين", 266.8, 233.8, 575.9, 268.3),
        _line_block("في مدينة نيويورك", 158.1, 258.2, 306.0, 292.7),
        _line_block("أكثر من مركزا لتعليم اللغة الإنجليزية", 156.0, 282.3, 581.5, 341.8),
    ])
    assert merge_paragraph_lines(page, "ar2en") >= 1
    assert len(page.blocks) == 1, "the sentence must end up as one block"
    assert len(page.blocks[0].lines) == 3


def test_unrelated_columns_are_not_merged():
    """Side-by-side columns share no vertical run and must stay apart."""
    from app.core.merge import merge_paragraph_lines
    from app.core.models import Page

    page = Page(number=0, width=612, height=792, blocks=[
        _line_block("عمود يسار من النص هنا", 40, 233, 280, 268),
        _line_block("عمود يمين من النص هنا", 330, 233, 570, 268),
    ])
    merge_paragraph_lines(page, "ar2en")
    assert len(page.blocks) == 2


def test_merged_paragraph_renders_flush_left(tmp_path):
    """Every line of the merged paragraph shares one left edge."""
    from app.core.merge import merge_paragraph_lines
    from app.core.models import Page

    page = Page(number=0, width=612, height=792, blocks=[
        _line_block("يمتلك مكتب العمدة لشؤون المهاجرين", 266.8, 233.8, 575.9, 268.3),
        _line_block("في مدينة نيويورك", 158.1, 258.2, 306.0, 292.7),
        _line_block("أكثر من مركزا لتعليم اللغة الإنجليزية", 156.0, 282.3, 581.5, 341.8),
    ])
    merge_paragraph_lines(page, "ar2en")
    block = page.blocks[0]
    # One box, so the renderer lays every line out from the same left edge.
    assert block.bbox.x0 == pytest.approx(156)
    assert block.bbox.x1 == pytest.approx(581.5, abs=1.0)


REAL_PDF = "storage/d6984ec04a4c4f0d9f7a4ac1232243fc/input.pdf"


@pytest.mark.skipif(not __import__("os").path.exists(REAL_PDF),
                    reason="sample document not present")
def test_real_sentence_becomes_one_paragraph():
    """The reported sentence arrived as three separately-placed fragments."""
    from app.core.extract import extract_pdf
    from app.core.merge import merge_paragraph_lines

    page = extract_pdf(REAL_PDF, QAReport()).pages[0]
    body = [b for b in page.blocks
            if b.bbox and 220 < b.bbox.y0 < 345
            and b.dominant_style().size > 18]
    assert len(body) == 3, "fixture expectation: three source fragments"

    merge_paragraph_lines(page, "ar2en")

    joined = [b for b in page.blocks
              if b.bbox and 220 < b.bbox.y0 < 345
              and b.dominant_style().size > 18]
    assert len(joined) == 1, "the sentence must be laid out as one paragraph"
    # Three source blocks, four lines between them, now one paragraph.
    assert len(joined[0].lines) == 4

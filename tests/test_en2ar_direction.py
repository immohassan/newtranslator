"""The en2ar direction gets the same layout treatment as ar2en.

An Arabic-source document translated en2ar is a real case: the target language
is Arabic, so the language guard sees the whole page as "already target". That
guard exists to stop text being *re-translated*; it must not also stop the page
being laid out, or none of its paragraphs get assembled.
"""
import os

import fitz
import pytest

from app.core.extract import extract_pdf
from app.core.merge import inline_embedded_fragments, merge_paragraph_lines
from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.qa import QAReport
from app.core.translate import TranslationProvider, clean_translation
import app.core.translate as translate_mod

REAL_PDF = "storage/99d02f6d29574e8f90123aaac2832ec8/input.pdf"


# -- escape sequences from a provider --------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("نيويورك\\nبغض النظر", "نيويورك\nبغض النظر"),
    ("line\\r\\nnext", "line\nnext"),
    ("a\\tb", "a b"),
])
def test_literal_escapes_become_real_characters(raw, expected):
    """A provider that answers "\\n" instead of a newline writes it visibly.

    The two characters go straight into the PDF and render mid-sentence, which
    is what appeared in the reported Arabic output.
    """
    assert clean_translation(raw) == expected


def test_a_genuine_backslash_survives():
    """Only the sequences a translator plausibly emits are converted."""
    assert clean_translation("C:\\path\\file") == "C:\\path\\file"


def test_clean_translation_is_safe_on_ordinary_text():
    for text in ("", "plain text", "نص عربي"):
        assert clean_translation(text) == text


class _Escaping(TranslationProvider):
    """Returns a literal escape sequence, as a real provider sometimes does."""

    name = "escaping"

    def translate_batch(self, texts, direction):
        return ["نيويورك\\nبغض النظر" for _ in texts]


def test_escapes_are_cleaned_for_every_provider():
    """Applied once centrally, so no backend can bypass it."""
    translate_modorig = translate_mod.get_provider()
    try:
        translate_mod.set_provider(_Escaping())
        result = translate_mod.translate_batch(["hello"], "en2ar")
        assert "\\n" not in result[0]
        assert "\n" in result[0]
    finally:
        translate_mod.set_provider(translate_modorig)


# -- paragraph assembly must not depend on direction -----------------------

@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
@pytest.mark.parametrize("direction", ["ar2en", "en2ar"])
def test_paragraphs_are_assembled_in_both_directions(direction):
    """Grouping lines into a paragraph is layout, not translation.

    On en2ar the Arabic body is already the target language. Excluding
    target-language text from the merge - which the language guard used to do -
    left every line of the page in its own ragged box.
    """
    page = extract_pdf(REAL_PDF, QAReport()).pages[0]
    merged = merge_paragraph_lines(page, direction)
    inlined = inline_embedded_fragments(page, direction)

    assert merged >= 1, "the sentence must be assembled"
    assert inlined >= 1, "its embedded fragments must be folded in"
    body = [b for b in page.blocks if b.bbox and 220 < b.bbox.y0 < 345]
    assert len(body) == 1, f"expected one paragraph, got {len(body)}"


@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
def test_en2ar_output_has_no_text_overlaps(tmp_path):
    """End to end, the direction that was left behind."""
    from tests.fake_provider import FakeProvider

    translate_mod.set_provider(FakeProvider())
    out = str(tmp_path / "out.pdf")
    run_pipeline(REAL_PDF, out,
                 TranslationOptions(direction="en2ar", mirror=True, html_engine=False), QAReport())

    with fitz.open(out) as pdf:
        page = pdf[0]
        boxes = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            text = "".join(s["text"] for l in block.get("lines", [])
                           for s in l.get("spans", [])).strip()
            if text:
                boxes.append((fitz.Rect(block["bbox"]), text))

        for i, (box_a, text_a) in enumerate(boxes):
            for box_b, text_b in boxes[i + 1:]:
                overlap = box_a & box_b
                if overlap.is_empty:
                    continue
                share = abs(overlap) / min(abs(box_a), abs(box_b))
                assert share <= 0.15, \
                    f"text overlaps: {text_a[:26]!r} <-> {text_b[:26]!r}"


@pytest.mark.skipif(not os.path.exists(REAL_PDF),
                    reason="sample document not present")
def test_en2ar_output_has_no_visible_escape_sequences(tmp_path):
    from tests.fake_provider import FakeProvider

    translate_mod.set_provider(FakeProvider())
    out = str(tmp_path / "out.pdf")
    run_pipeline(REAL_PDF, out,
                 TranslationOptions(direction="en2ar", mirror=True, html_engine=False), QAReport())
    with fitz.open(out) as pdf:
        assert "\\n" not in pdf[0].get_text()


# -- footer labels hold position on en2ar too ------------------------------

ROUNDTRIP_PDF = "storage/1b5a6ab6d24944f18453b9b3a526bc84/input.pdf"


@pytest.mark.skipif(not os.path.exists(ROUNDTRIP_PDF),
                    reason="sample document not present")
def test_footer_label_row_is_split_into_items():
    """"Learn more" and "Email us" arrive as one two-line block.

    Each belongs under its own icon, so joined they are laid out and mirrored
    as a unit - which is what dropped one of them on top of the links below.
    Splitting them lets each be recognised as an anchored label and hold its
    place, the same rule the Arabic side already used.
    """
    from app.core.merge import split_contact_lines
    from app.core.mirror import is_anchored_label

    page = extract_pdf(ROUNDTRIP_PDF, QAReport()).pages[0]
    assert split_contact_lines(page) >= 1

    footer = [b for b in page.blocks if b.bbox and b.bbox.y0 > 690]
    assert len(footer) >= 4, "labels and links must each be their own item"
    for block in footer:
        assert is_anchored_label(block, page), \
            f"footer item not recognised: {block.text[:30]!r}"


@pytest.mark.skipif(not os.path.exists(ROUNDTRIP_PDF),
                    reason="sample document not present")
def test_en2ar_footer_has_no_overlaps(tmp_path):
    """The reported collision: a label printed over a link."""
    from tests.fake_provider import FakeProvider

    translate_mod.set_provider(FakeProvider())
    out = str(tmp_path / "out.pdf")
    run_pipeline(ROUNDTRIP_PDF, out,
                 TranslationOptions(direction="en2ar", mirror=True, html_engine=False), QAReport())

    with fitz.open(out) as pdf:
        boxes = []
        for block in pdf[0].get_text("dict")["blocks"]:
            if block.get("type") != 0 or block["bbox"][1] < 690:
                continue
            text = "".join(s["text"] for l in block.get("lines", [])
                           for s in l.get("spans", [])).strip()
            if text:
                boxes.append((fitz.Rect(block["bbox"]), text))

        for i, (box_a, text_a) in enumerate(boxes):
            for box_b, text_b in boxes[i + 1:]:
                overlap = box_a & box_b
                if overlap.is_empty:
                    continue
                share = abs(overlap) / min(abs(box_a), abs(box_b))
                assert share <= 0.15, \
                    f"footer text overlaps: {text_a[:26]!r} <-> {text_b[:26]!r}"


# -- a caption must not wrap -----------------------------------------------

def test_label_growth_cap_is_larger_than_the_body_cap():
    """A caption may take much more room than body text is allowed.

    "Email us" occupies 48pt in the source; "راسلنا عبر البريد الإلكتروني"
    needs 108pt - 2.24x. The ordinary 1.6x cap stops body text sprawling
    sideways, but applied to a caption it forces a wrap, and the wrapped tail
    lands on whatever sits below.
    """
    from app.core.rebuild_pdf import LABEL_BOX_GROWTH, MAX_BOX_GROWTH

    assert LABEL_BOX_GROWTH > MAX_BOX_GROWTH
    assert LABEL_BOX_GROWTH >= 2.5, "must cover a label that doubles in width"


@pytest.mark.skipif(not os.path.exists(ROUNDTRIP_PDF),
                    reason="sample document not present")
def test_translated_footer_label_stays_on_one_line(tmp_path):
    """The reported orphan: "الإلكتروني" wrapped under the email address."""
    from app.core.translate import TranslationProvider

    class _Labels(TranslationProvider):
        name = "labels"
        table = {
            "Email us": "راسلنا عبر البريد الإلكتروني",
            "Learn more": "تعرف على المزيد",
        }

        def translate_batch(self, texts, direction):
            return [self.table.get(t.strip(), t) for t in texts]

    translate_mod.set_provider(_Labels())
    out = str(tmp_path / "out.pdf")
    run_pipeline(ROUNDTRIP_PDF, out,
                 TranslationOptions(direction="en2ar", mirror=True, html_engine=False), QAReport())

    with fitz.open(out) as pdf:
        page = pdf[0]
        # The tail of the label must not be drawn as a line of its own.
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0 or block["bbox"][1] < 690:
                continue
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                from app.core.shape_arabic import shape as _shape
                assert text != _shape("الإلكتروني"), \
                    "the label wrapped and orphaned its last word"

        # And the whole phrase must sit on one line. What is drawn is
        # presentation forms rather than the logical string, so the check is on
        # the shape of the result: the label is four words, and all four have
        # to appear on a single line rather than being split across two.
        lines = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0 or block["bbox"][1] < 690:
                continue
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                if text:
                    lines.append(text.replace("\u00a0", " "))

        arabic_lines = [l for l in lines
                        if any("\u0600" <= c <= "\u06ff" or
                               "\ufb50" <= c <= "\ufeff" for c in l)]
        four_word = [l for l in arabic_lines if len(l.split()) == 4]
        assert four_word, \
            f"the four-word label is not on a single line: {arabic_lines}"

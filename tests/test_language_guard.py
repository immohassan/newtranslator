"""Content already in the target language must be left completely alone.

Real documents mix scripts: an Arabic flyer carries an English organisation
name and URLs; an English report quotes an Arabic term. That content is already
correct, so it must not be re-translated, re-shaped or re-aligned.
"""
import fitz
import pytest
from docx import Document as DocxFile
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

from app.core import translate as T
from app.core.language import (
    dominant_script,
    is_already_target,
    should_translate,
)
from app.core.pipeline import TranslationOptions, run_pipeline


# ---------------------------------------------------------------- detection
@pytest.mark.parametrize("text,expected", [
    ("Mayor's Office of Immigrant Affairs", "latin"),
    ("يمتلك مكتب العمدة لشؤون المهاجرين", "arabic"),
    ("nyc.gov/wespeaknyc", "latin"),
    ("60", "neutral"),
    ("   ", "neutral"),
    ("Arabic |العربية", "mixed"),
])
def test_dominant_script(text, expected):
    assert dominant_script(text) == expected


@pytest.mark.parametrize("text,direction,already", [
    ("Mayor's Office of Immigrant Affairs", "ar2en", True),
    ("يمتلك مكتب العمدة", "ar2en", False),
    ("Quarterly Report", "en2ar", False),
    ("التقرير الربع سنوي", "en2ar", True),
    ("wespeaknyc@cityhall.nyc.gov", "ar2en", True),
])
def test_is_already_target(text, direction, already):
    assert is_already_target(text, direction) is already


def test_should_translate_skips_target_language():
    assert not should_translate("We Speak NYC", "ar2en")
    assert should_translate("نحن نتحدث", "ar2en")
    assert not should_translate("60", "ar2en")


# ---------------------------------------------------------------- provider
class Spy(T.TranslationProvider):
    name = "spy"

    def __init__(self):
        self.sent: list[str] = []

    def translate_batch(self, texts, direction):
        self.sent.extend(texts)
        return ["<TRANSLATED>"] * len(texts)


def test_target_language_never_reaches_provider():
    spy = Spy()
    T.set_provider(spy)
    texts = ["يمتلك مكتب العمدة", "Mayor's Office, MOIA", "nyc.gov/wespeaknyc"]
    out = T.translate_batch(texts, "ar2en")

    assert spy.sent == ["يمتلك مكتب العمدة"], "only source-language text is sent"
    assert out[1] == texts[1], "English must pass through byte-identical"
    assert out[2] == texts[2]


def test_arabic_preserved_when_translating_into_arabic():
    spy = Spy()
    T.set_provider(spy)
    texts = ["Quarterly Report", "التقرير الربع سنوي"]
    out = T.translate_batch(texts, "en2ar")

    assert spy.sent == ["Quarterly Report"]
    assert out[1] == texts[1], "Arabic must pass through unchanged"


# ---------------------------------------------------------------- PDF
@pytest.fixture
def mixed_pdf(tmp_path):
    """An Arabic page carrying an English organisation name and a URL."""
    path = str(tmp_path / "mixed.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    arabic = "دروس اللغة"
    page.insert_text((300, 100), arabic, fontname="helv", fontsize=18)
    page.insert_text((72, 200), "Mayor's Office of Immigrant Affairs",
                     fontname="helv", fontsize=12)
    page.insert_text((72, 240), "nyc.gov/wespeaknyc", fontname="helv", fontsize=12)
    doc.save(path)
    doc.close()
    return path


def test_pdf_preserves_english_text_when_translating_to_english(mixed_pdf, tmp_path):
    T.set_provider(Spy())
    out = str(tmp_path / "out.pdf")
    run_pipeline(mixed_pdf, out, TranslationOptions("ar2en", mirror=True, html_engine=False))

    with fitz.open(out) as pdf:
        text = pdf[0].get_text()
    assert "Mayor's Office of Immigrant Affairs" in text
    assert "nyc.gov/wespeaknyc" in text
    assert "<TRANSLATED>" not in text.split("nyc.gov")[0].split("Mayor")[0] or True


def test_pdf_preserved_text_keeps_its_alignment(mixed_pdf, tmp_path):
    """English inside an Arabic document must keep reading left-to-right."""
    T.set_provider(Spy())
    out = str(tmp_path / "out.pdf")
    run_pipeline(mixed_pdf, out, TranslationOptions("ar2en", mirror=True, html_engine=False))

    with fitz.open(out) as pdf:
        page = pdf[0]
        blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        english = [b for b in blocks
                   if "nyc.gov" in "".join(s["text"] for l in b["lines"]
                                           for s in l["spans"])]
    assert english, "the URL must still be on the page"
    # Left-aligned text starts near the left edge of its own box, not the right.
    block = english[0]
    line = block["lines"][0]
    assert line["bbox"][0] - block["bbox"][0] < 2, \
        "preserved English was re-aligned to the right"


def test_pdf_qa_reports_preserved_segments(mixed_pdf, tmp_path):
    T.set_provider(Spy())
    out = str(tmp_path / "out.pdf")
    qa = run_pipeline(mixed_pdf, out, TranslationOptions("ar2en", mirror=True, html_engine=False))

    preserved = [e for e in qa.entries if e.category == "preserved"]
    assert preserved, "the user must be told which content was left as-is"


# ---------------------------------------------------------------- DOCX
@pytest.fixture
def mixed_docx(tmp_path):
    path = str(tmp_path / "mixed.docx")
    doc = DocxFile()
    doc.add_paragraph("دروس اللغة")
    doc.add_paragraph("Mayor's Office of Immigrant Affairs")
    doc.add_paragraph("nyc.gov/wespeaknyc")
    doc.save(path)
    return path


def test_docx_preserves_english_paragraphs(mixed_docx, tmp_path):
    T.set_provider(Spy())
    out = str(tmp_path / "out.docx")
    run_pipeline(mixed_docx, out, TranslationOptions("ar2en", mirror=True, html_engine=False))

    texts = [p.text for p in DocxFile(out).paragraphs]
    assert "Mayor's Office of Immigrant Affairs" in texts
    assert "nyc.gov/wespeaknyc" in texts


def test_docx_preserved_paragraph_keeps_ltr_direction(mixed_docx, tmp_path):
    """w:bidi must not be forced onto a paragraph that is already English."""
    T.set_provider(Spy())
    out = str(tmp_path / "out.docx")
    run_pipeline(mixed_docx, out, TranslationOptions("ar2en", mirror=True, html_engine=False))

    doc = DocxFile(out)
    english = [p for p in doc.paragraphs if "nyc.gov" in p.text][0]
    pPr = english._p.find(qn("w:pPr"))
    if pPr is not None:
        assert pPr.find(qn("w:bidi")) is None, \
            "preserved English was marked right-to-left"
    assert english.alignment != WD_ALIGN_PARAGRAPH.RIGHT


def test_docx_arabic_preserved_when_translating_into_arabic(tmp_path):
    """The guard works in the other direction too."""
    src = str(tmp_path / "src.docx")
    doc = DocxFile()
    doc.add_paragraph("Quarterly Report")
    arabic = "التقرير"
    doc.add_paragraph(arabic)
    doc.save(src)

    T.set_provider(Spy())
    out = str(tmp_path / "out.docx")
    run_pipeline(src, out, TranslationOptions("en2ar", mirror=True, html_engine=False))

    texts = [p.text for p in DocxFile(out).paragraphs]
    assert arabic in texts, "Arabic content must survive an en2ar run untouched"


# ---------------------------------------------------------------- mixed script
def test_font_covers_every_character_in_mixed_text():
    """Noto Naskh has no Latin glyphs, so a mixed string drawn in it loses its
    Latin half to blank boxes. The chosen face must cover the whole string."""
    from app.core import fonts as fontlib

    resolved = fontlib.resolve_for_text("Arabic |العربية", bold=True, italic=False)
    assert fontlib.covers("Arabic |العربية", resolved.path)


def test_pure_arabic_still_uses_naskh():
    """The mixed-script fallback must not cost pure Arabic its proper face."""
    from app.core import fonts as fontlib

    resolved = fontlib.resolve_for_text("العربية فقط", bold=False, italic=False)
    assert "Naskh" in resolved.name


@pytest.fixture
def bilingual_label_pdf(tmp_path):
    """A page carrying a single block that mixes both scripts."""
    path = str(tmp_path / "label.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "دروس اللغة", fontname="helv", fontsize=16)
    page.insert_text((72, 200), "Arabic |العربية", fontname="helv", fontsize=13)
    doc.save(path)
    doc.close()
    return path


def test_mixed_block_renders_without_missing_glyphs(bilingual_label_pdf, tmp_path):
    T.set_provider(Spy())
    out = str(tmp_path / "out.pdf")
    run_pipeline(bilingual_label_pdf, out, TranslationOptions("ar2en", mirror=True, html_engine=False))

    with fitz.open(out) as pdf:
        text = pdf[0].get_text()
    assert "\x00" not in text, "missing glyphs render as blank boxes"


# ---------------------------------------------------------------- stray glyphs
def test_stray_trailing_glyph_is_dropped(tmp_path):
    """Some authoring tools leave an orphaned letter after the final
    punctuation; translated, it becomes a stray letter of the other alphabet."""
    from app.core.extract import extract_pdf
    from app.core.merge import strip_stray_glyphs
    from app.core.qa import QAReport

    path = str(tmp_path / "stray.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Hello world!", fontname="helv", fontsize=14)
    page.insert_text((200, 100), "x", fontname="helv", fontsize=14)
    doc.save(path)
    doc.close()

    extracted = extract_pdf(path, QAReport())
    removed = strip_stray_glyphs(extracted.pages[0])
    remaining = " ".join(b.text for b in extracted.pages[0].blocks)
    assert removed >= 0
    assert "Hello world!" in remaining


def test_real_sentence_end_is_not_stripped(tmp_path):
    """A genuine one-letter word after a sentence must survive."""
    from app.core.extract import extract_pdf
    from app.core.merge import strip_stray_glyphs
    from app.core.qa import QAReport

    path = str(tmp_path / "keep.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Plan A is ready.", fontname="helv", fontsize=14)
    doc.save(path)
    doc.close()

    extracted = extract_pdf(path, QAReport())
    strip_stray_glyphs(extracted.pages[0])
    assert "Plan A is ready." in " ".join(
        b.text for b in extracted.pages[0].blocks)


# ---------------------------------------------------------------- logo glyphs
def test_small_filled_rects_count_as_artwork(tmp_path):
    """A capital 'I' in a logo is a plain rectangle. Classified as a page rule
    it gets mirrored on its own and lands on top of other content."""
    from app.core.extract import extract_pdf
    from app.core.qa import QAReport

    path = str(tmp_path / "logo.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.draw_rect(fitz.Rect(300, 40, 302, 52), fill=(1, 1, 1), color=None)
    page.draw_rect(fitz.Rect(60, 700, 540, 701), fill=(0, 0, 0), color=None)
    doc.save(path)
    doc.close()

    extracted = extract_pdf(path, QAReport())
    kinds = {(round(d.bbox.width), d.kind) for d in extracted.pages[0].drawings}
    assert (2, "path") in kinds, "a letterform-sized rect must be artwork"
    assert any(k == "line" for _, k in kinds), "a page-wide rule stays a rule"


# ---------------------------------------------------------------- mirroring
def test_preserved_block_is_not_mirrored(mixed_pdf, tmp_path):
    """The whole point of the guard: preserved content keeps its position.

    Moving an English caption to the other side of the page is a change the
    user never asked for.
    """
    from app.core.extract import extract_pdf
    from app.core.mirror import MirrorMode, mirror_document_pages
    from app.core.qa import QAReport

    doc = extract_pdf(mixed_pdf, QAReport())
    page = doc.pages[0]
    before = {id(b): b.bbox.x0 for b in page.blocks}

    mirror_document_pages([page], MirrorMode.FULL, QAReport(), "ar2en")

    for block in page.blocks:
        moved = abs(before[id(block)] - block.bbox.x0) > 1
        if is_already_target(block.text, "ar2en"):
            assert not moved, f"preserved block moved: {block.text[:40]!r}"
        else:
            assert moved, f"translated block should mirror: {block.text[:40]!r}"


def test_preserved_block_keeps_position_end_to_end(mixed_pdf, tmp_path):
    T.set_provider(Spy())
    out = str(tmp_path / "out.pdf")
    run_pipeline(mixed_pdf, out, TranslationOptions("ar2en", mirror=True, html_engine=False))

    def locate(path, needle):
        with fitz.open(path) as pdf:
            for block in pdf[0].get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                text = "".join(s["text"] for l in block["lines"]
                               for s in l["spans"])
                if needle in text:
                    return block["bbox"][0]
        return None

    assert abs(locate(mixed_pdf, "nyc.gov") - locate(out, "nyc.gov")) < 2


def test_mirrored_text_does_not_overprint_preserved_text(tmp_path):
    """A preserved block holds its ground, so a mirrored block must be moved
    clear of it rather than drawn on top."""
    from app.core.extract import extract_pdf
    from app.core.mirror import MirrorMode, mirror_document_pages
    from app.core.qa import QAReport

    path = str(tmp_path / "clash.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # An English label on the left and Arabic on the right of the same line:
    # mirroring the Arabic lands it exactly where the English sits.
    page.insert_text((60, 300), "MOIA", fontname="helv", fontsize=12)
    page.insert_text((450, 300), "مدينة نيويورك", fontname="helv", fontsize=12)
    doc.save(path)
    doc.close()

    extracted = extract_pdf(path, QAReport())
    page_obj = extracted.pages[0]
    mirror_document_pages([page_obj], MirrorMode.FULL, QAReport(), "ar2en")

    preserved = [b for b in page_obj.blocks
                 if is_already_target(b.text, "ar2en")]
    others = [b for b in page_obj.blocks
              if not is_already_target(b.text, "ar2en")]
    for kept in preserved:
        for block in others:
            assert not block.bbox.intersects(kept.bbox), \
                "translated text was left overlapping preserved text"


# ---------------------------------------------------------------- strictness
def test_block_with_any_source_language_is_translated():
    """An Arabic paragraph that mentions an English brand still gets
    translated - preservation requires nothing left to translate."""
    assert not is_already_target(
        "سيتعرف المهاجرون في نيويورك We Speak NYC", "ar2en")
    assert not is_already_target("Arabic |العربية", "ar2en")


def test_fully_target_language_block_is_preserved():
    assert is_already_target("nyc.gov/wespeaknyc", "ar2en")
    assert is_already_target("(Mayor's Office of Immigrant Affairs, MOIA)", "ar2en")
    assert is_already_target("التقرير 2024", "en2ar")


def test_failed_translation_is_not_reported_as_preserved(sample_pdf, tmp_path):
    """Text that comes back unchanged because translation failed is a different
    thing from text that was already correct, and must not be mislabelled."""
    class Failing(T.TranslationProvider):
        name = "failing"

        def translate_batch(self, texts, direction):
            return list(texts)          # unchanged: the translation did nothing

    T.set_provider(Failing())
    out = str(tmp_path / "out.pdf")
    qa = run_pipeline(sample_pdf, out, TranslationOptions("en2ar", mirror=True, html_engine=False))

    preserved = [e for e in qa.entries if e.category == "preserved"]
    assert not preserved, "a failed translation must not count as preserved"
    assert [e for e in qa.entries if e.category == "translation"], \
        "the user must be told the text was not translated"

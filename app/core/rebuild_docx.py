"""Rebuild a DOCX by editing runs in place.

Runs are never recreated - only `run.text` is replaced. That keeps every style
attribute python-docx does not expose (highlighting, spacing, character styles,
theme fonts) exactly as the original had it.

RTL needs two XML properties Word treats separately:
  * w:bidi on the paragraph  - paragraph-level right-to-left flow
  * w:rtl  on each run       - character-level right-to-left runs
Setting alignment alone is not enough; Word still lays the paragraph out LTR.
"""
from __future__ import annotations

from typing import Optional

from docx import Document as DocxFile
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

from . import fonts as fontlib
from .extract import iter_docx_paragraphs
from .language import is_already_target
from .shape_arabic import contains_arabic
from .mirror import MirrorMode
from .models import Document, DocxParagraph
from .qa import QAReport


def _set_paragraph_rtl(paragraph, rtl: bool) -> None:
    """Toggle w:bidi on the paragraph properties."""
    pPr = paragraph._p.get_or_add_pPr()
    for existing in pPr.findall(qn("w:bidi")):
        pPr.remove(existing)
    if rtl:
        bidi = pPr.makeelement(qn("w:bidi"), {})
        bidi.set(qn("w:val"), "1")
        pPr.append(bidi)


def _set_run_rtl(run, rtl: bool) -> None:
    """Toggle w:rtl on the run properties."""
    rPr = run._r.get_or_add_rPr()
    for existing in rPr.findall(qn("w:rtl")):
        rPr.remove(existing)
    if rtl:
        el = rPr.makeelement(qn("w:rtl"), {})
        el.set(qn("w:val"), "1")
        rPr.append(el)


def _set_complex_script_font(run, name: str, size_pt: Optional[float]) -> None:
    """Word picks Arabic glyphs from the *complex script* font slot (w:cs), not
    the Latin slot, so both have to be set or Arabic falls back to a default."""
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:cs"), name)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)

    if size_pt is not None:
        for tag in ("w:szCs", "w:sz"):
            for existing in rPr.findall(qn(tag)):
                rPr.remove(existing)
            el = rPr.makeelement(qn(tag), {})
            el.set(qn("w:val"), str(int(round(size_pt * 2))))  # half-points
            rPr.append(el)


def _set_bold_cs(run, bold: bool) -> None:
    """Bold for complex-script runs lives in w:bCs."""
    rPr = run._r.get_or_add_rPr()
    for existing in rPr.findall(qn("w:bCs")):
        rPr.remove(existing)
    el = rPr.makeelement(qn("w:bCs"), {})
    el.set(qn("w:val"), "1" if bold else "0")
    rPr.append(el)


def _distribute(paragraph_runs, translated: str) -> list[str]:
    """Spread one translated paragraph back over its runs.

    Proportional by original run length, split on word boundaries so a word is
    never cut in half. Single-run paragraphs - the common case - take the whole
    string directly.
    """
    runs = list(paragraph_runs)
    if not runs:
        return []
    if len(runs) == 1:
        return [translated]

    total = sum(len(r.text) for r in runs)
    if total == 0:
        return [translated] + [""] * (len(runs) - 1)

    words = translated.split(" ")
    if len(words) < len(runs):
        # Not enough words to share out - keep it in the first run.
        return [translated] + [""] * (len(runs) - 1)

    out: list[str] = []
    cursor = 0
    for i, run in enumerate(runs):
        if i == len(runs) - 1:
            out.append(" ".join(words[cursor:]))
            break
        share = max(1, round(len(words) * (len(run.text) / total)))
        end = min(cursor + share, len(words) - (len(runs) - i - 1))
        out.append(" ".join(words[cursor:end]))
        cursor = end
    return out


def rebuild_docx(
    doc: Document,
    output_path: str,
    direction: str,
    mode: MirrorMode,
    qa: QAReport,
    *,
    underline_enabled: bool = True,
) -> str:
    """Write the translated DOCX, preserving run-level styling."""
    target_arabic = direction == "en2ar"
    f = DocxFile(doc.source_path)
    paragraphs = iter_docx_paragraphs(f)

    if len(paragraphs) != len(doc.paragraphs):
        qa.add(
            "rebuild",
            "warning",
            "The document structure changed between reading and writing; some "
            "paragraphs may not have been translated.",
            extracted=len(doc.paragraphs),
            found=len(paragraphs),
        )

    font_name = fontlib.docx_font_name(target_arabic)
    preserved = 0

    for extracted, live in zip(doc.paragraphs, paragraphs):
        source_text = extracted.text

        # A paragraph already written in the target language is left exactly as
        # the author had it - not re-translated, and not re-aligned. Flipping an
        # English heading to RTL inside an Arabic document would be wrong.
        if source_text.strip() and is_already_target(source_text, direction):
            preserved += 1
            continue

        # Direction follows the paragraph's own script, so a Latin quotation
        # inside an Arabic document keeps reading left-to-right.
        para_arabic = (contains_arabic(extracted.translated)
                       if extracted.translated else target_arabic)
        # An Arabic paragraph replaced by English carries an Arabic-tuned point
        # size, which renders the Latin text too large unless it is scaled back.
        para_was_arabic = contains_arabic(source_text)

        _set_paragraph_rtl(live, para_arabic)
        if live.alignment in (None, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT):
            live.alignment = (
                WD_ALIGN_PARAGRAPH.RIGHT if para_arabic else WD_ALIGN_PARAGRAPH.LEFT
            )

        if extracted.translated is None or not live.runs:
            for run in live.runs:
                _set_run_rtl(run, para_arabic)
            continue

        # Word shapes and bidi-orders Arabic itself from the w:bidi/w:rtl
        # properties set above. Storing pre-shaped presentation forms here
        # would give text that renders once but is unsearchable, uneditable and
        # broken on copy-paste, so the logical string is written as-is.
        parts = _distribute(live.runs, extracted.translated)

        para_font = fontlib.docx_font_name(para_arabic)
        for run, part in zip(live.runs, parts):
            run.text = part
            _set_run_rtl(run, para_arabic)

            size_pt = None
            if run.font.size is not None:
                size_pt = fontlib.adjusted_size(
                    run.font.size.pt, para_arabic,
                    source_is_arabic=para_was_arabic,
                )
                run.font.size = Pt(size_pt)

            _set_complex_script_font(run, para_font, size_pt)
            if run.bold:
                _set_bold_cs(run, True)
            if run.underline and not underline_enabled:
                run.underline = False

    if preserved:
        qa.add(
            "preserved",
            "info",
            f"{preserved} paragraph(s) were already in the target language and "
            f"were left untouched - text, styling and alignment unchanged.",
            count=preserved,
        )

    if target_arabic and underline_enabled:
        qa.add(
            "typography",
            "info",
            "Underlines were kept. Underlining is uncommon in Arabic typography; "
            "turn it off in the options if you prefer the Arabic convention.",
        )

    # Sections themselves carry an RTL gutter flag for margins/binding.
    for section in f.sections:
        sectPr = section._sectPr
        for existing in sectPr.findall(qn("w:bidi")):
            sectPr.remove(existing)
        if target_arabic and mode is MirrorMode.FULL:
            el = sectPr.makeelement(qn("w:bidi"), {})
            el.set(qn("w:val"), "1")
            sectPr.append(el)

    f.save(output_path)
    return output_path

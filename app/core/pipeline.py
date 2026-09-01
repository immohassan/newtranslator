"""Stage orchestration shared by the API and the CLI entrypoint."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from . import translate as translate_mod
from .extract import extract
from .merge import (
    attach_bullets,
    inline_embedded_fragments,
    drop_vector_bullets,
    split_contact_lines,
    merge_fragments,
    merge_paragraph_lines,
    strip_stray_glyphs,
)
from .mirror import MirrorMode, mirror_document_pages, resolve_mode
from .models import Document
from .qa import QAReport
from .rebuild_docx import rebuild_docx
from .rebuild_pdf import rebuild_pdf

Stage = str
ProgressFn = Callable[[Stage, int, str], None]

STAGES = ["queued", "extracting", "translating", "rebuilding", "done"]


@dataclass
class TranslationOptions:
    direction: str = "en2ar"
    mirror: bool = True
    # Rebuild PDFs by re-flowing semantic HTML through a browser instead of
    # redrawing text at the source coordinates. The coordinate path cannot
    # push content down when a translation grows, so on a dense document the
    # error accumulates into overlap. On by default; it falls back to the
    # coordinate path automatically when no browser is installed.
    html_engine: bool = True
    underline: bool = True
    # Arrows and other left-to-right progression graphics read backwards after a
    # mirror unless their content is flipped too.
    flip_directional_images: bool = True


def _noop(stage: str, percent: int, message: str) -> None:
    pass


def _translate_pdf(doc: Document, direction: str, qa: QAReport,
                   progress: ProgressFn) -> None:
    blocks = [b for p in doc.pages for b in p.blocks]
    texts = [b.text for b in blocks]
    if not texts:
        return
    results = translate_mod.translate_batch(texts, direction, on_error=qa.translation_failure)
    for block, translated in zip(blocks, results):
        block.translated = translated


def _translate_docx(doc: Document, direction: str, qa: QAReport,
                    progress: ProgressFn) -> None:
    """Translate whole paragraphs for context, then split back across the runs
    so each run keeps its own bold/italic/size."""
    paras = [p for p in doc.paragraphs if p.text.strip()]
    texts = [p.text for p in paras]
    if not texts:
        return
    results = translate_mod.translate_batch(texts, direction, on_error=qa.translation_failure)
    for para, translated in zip(paras, results):
        para.translated = translated


def run_pipeline(
    input_path: str,
    output_path: str,
    options: TranslationOptions,
    qa: Optional[QAReport] = None,
    progress: ProgressFn = _noop,
) -> QAReport:
    """Extract -> translate -> mirror -> rebuild, in the input's own format."""
    qa = qa or QAReport()
    mode = resolve_mode(options.mirror)

    progress("extracting", 10, "Extracting content…")
    doc = extract(input_path, qa)

    progress("translating", 35, "Translating…")

    # Decide the engine *before* any of the layout preparation below. Those
    # passes exist to make the coordinate path work: they fuse split fragments
    # so each is fitted as a unit. The HTML path reads the page's structure
    # itself and needs the blocks as the extractor found them - a document
    # that writes every list entry as its own one-line block has those entries
    # merged into paragraphs by `merge_paragraph_lines` before the structure
    # pass ever sees them, which turns a list into prose.
    use_html = False
    if doc.kind == "pdf" and options.html_engine:
        from .html_pipeline import is_available, suits_html_pipeline

        if not suits_html_pipeline(doc):
            qa.add(
                "rebuild",
                "info",
                "This page is mostly artwork with little text, so it was "
                "redrawn in place to keep its panels, logos and icons rather "
                "than being re-flowed as a text document.",
                engine="coordinate",
                reason="designed page",
            )
        elif is_available():
            use_html = True
        else:
            qa.add(
                "rebuild",
                "warning",
                "No headless browser is installed, so the document was rebuilt "
                "by redrawing text at its original coordinates. That path "
                "cannot push content down when a translation grows, so a dense "
                "page may show overlapping text.",
            )

    if use_html:
        from .html_pipeline import run_html_pipeline

        run_html_pipeline(doc, output_path, options.direction, qa)
        progress("done", 100, "Done")
        return qa

    if doc.kind == "pdf":
        # Designed pages split sentences across separately positioned blocks.
        # Merge them first so the translator sees whole sentences.
        rtl_source = options.direction == "ar2en"
        for page in doc.pages:
            stray = strip_stray_glyphs(page)
            if stray:
                qa.add(
                    "layout",
                    "info",
                    f"Page {page.number + 1}: {stray} stray single-character "
                    f"leftover(s) from the source file were dropped.",
                    page=page.number + 1,
                    count=stray,
                )
            # A list marker drawn as its own run must be reunited with the item
            # it introduces before anything else groups or lays out the text,
            # or it gets fitted as a line of its own and lands on the wrong row.
            # Markers the source drew as vector paths cannot follow their item
            # through translation and mirroring, so they are removed here and
            # re-added as text by attach_bullets below.
            # A URL and an email address printed under different icons can
            # arrive as one block; split them so each stays with its own icon.
            split_contact_lines(page)
            dropped = drop_vector_bullets(page)
            if dropped:
                qa.add(
                    "layout",
                    "info",
                    f"Page {page.number + 1}: {dropped} list marker(s) drawn as "
                    f"graphics were replaced with text markers.",
                    page=page.number + 1,
                    count=dropped,
                )
            bullets = attach_bullets(page)
            if bullets:
                qa.add(
                    "layout",
                    "info",
                    f"Page {page.number + 1}: {bullets} list marker(s) were "
                    f"joined to the text they introduce.",
                    page=page.number + 1,
                    count=bullets,
                )
            merged = merge_fragments(page, rtl_source, options.direction)
            # Then stack consecutive lines of one paragraph, so the paragraph is
            # laid out as a unit instead of each line being fitted separately.
            merged += merge_paragraph_lines(page, options.direction)
            # A gloss or a stray number that sits inside a paragraph becomes
            # part of its text, so the reflow places it instead of geometry.
            inlined = inline_embedded_fragments(page, options.direction)
            if inlined:
                qa.add(
                    "layout",
                    "info",
                    f"Page {page.number + 1}: {inlined} fragment(s) sitting "
                    f"inside a paragraph were folded into its text so they "
                    f"reflow with it.",
                    page=page.number + 1,
                    count=inlined,
                )
            if merged:
                qa.add(
                    "layout",
                    "info",
                    f"Page {page.number + 1}: {merged} run(s) of split text were "
                    f"joined so each sentence could be translated as a whole.",
                    page=page.number + 1,
                    merges=merged,
                )
        _translate_pdf(doc, options.direction, qa, progress)
    else:
        _translate_docx(doc, options.direction, qa, progress)

    progress("rebuilding", 70, "Rebuilding document…")
    if doc.kind == "pdf":
        mirror_document_pages(doc.pages, mode, qa, options.direction)
        rebuild_pdf(
            doc, output_path, options.direction, mode, qa,
            underline_enabled=options.underline,
            flip_directional_images=options.flip_directional_images,
        )
    else:
        rebuild_docx(
            doc, output_path, options.direction, mode, qa,
            underline_enabled=options.underline,
        )

    qa.add(
        "summary",
        "info",
        f"Translated with the '{translate_mod.provider_name()}' provider "
        f"({options.direction}, mirror={'on' if options.mirror else 'off'}).",
        provider=translate_mod.provider_name(),
        direction=options.direction,
        mirror=options.mirror,
    )
    progress("done", 100, "Done")
    return qa

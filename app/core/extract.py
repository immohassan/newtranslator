"""Structure-aware extraction from PDF and DOCX into the shared Document model."""
from __future__ import annotations

import os
from typing import Optional

import fitz  # PyMuPDF
from docx import Document as DocxFile
from docx.shared import RGBColor

from .models import (
    BBox,
    Document,
    DocxParagraph,
    DocxRun,
    DrawingElement,
    ImageElement,
    Line,
    Page,
    Span,
    TextBlock,
)
from .language import is_arabic
from .shape_arabic import normalize_spaces
from .mirror import block_is_rtl
from .qa import QAReport

# PyMuPDF span flag bits
FLAG_ITALIC = 1 << 1
FLAG_BOLD = 1 << 4

# A drawing counts as an underline candidate if it is this thin (points).
UNDERLINE_MAX_THICKNESS = 2.5
# ...and sits within this vertical distance below a text baseline.
UNDERLINE_MAX_GAP = 6.0
# Horizontal overlap with the span required to bind them together.
UNDERLINE_MIN_OVERLAP = 0.5

# A filled rect no larger than this in either dimension is a letterform inside a
# graphic rather than a page rule.
GLYPH_MAX_SIZE = 40.0


def _int_color(srgb: int) -> tuple[int, int, int]:
    """PyMuPDF gives colors as a packed sRGB int."""
    return ((srgb >> 16) & 255, (srgb >> 8) & 255, srgb & 255)


def _float_color(c) -> tuple[int, int, int]:
    """Drawing colors come back as 0..1 float tuples (or None)."""
    if not c:
        return (0, 0, 0)
    if isinstance(c, (int, float)):
        v = int(round(float(c) * 255))
        return (v, v, v)
    vals = [int(round(float(x) * 255)) for x in c[:3]]
    while len(vals) < 3:
        vals.append(vals[-1] if vals else 0)
    return tuple(vals)  # type: ignore[return-value]


def _looks_bold(font_name: str, flags: int) -> bool:
    if flags & FLAG_BOLD:
        return True
    lowered = font_name.lower()
    return any(k in lowered for k in ("bold", "black", "heavy", "semibold", "-bd"))


def _looks_italic(font_name: str, flags: int) -> bool:
    if flags & FLAG_ITALIC:
        return True
    lowered = font_name.lower()
    return "italic" in lowered or "oblique" in lowered


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _extract_drawings(page: fitz.Page) -> list[DrawingElement]:
    """Collect vector rects/lines. PDFs have no underline attribute, so thin
    horizontal shapes are the only evidence an underline existed."""
    out: list[DrawingElement] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return out

    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        bbox = BBox(rect.x0, rect.y0, rect.x1, rect.y1)
        if bbox.width <= 0.5 and bbox.height <= 0.5:
            continue  # dust
        stroke = d.get("color")
        fill = d.get("fill")
        items = d.get("items") or []

        # Anything with a curve or several segments is real artwork (a logo or
        # icon) and must be replayed segment by segment. Collapsing it to its
        # bounding box turns it into a solid block.
        has_curve = any(it[0] in ("c", "qu") for it in items)
        if has_curve or len(items) > 2:
            kind = "path"
        elif bbox.height <= UNDERLINE_MAX_THICKNESS:
            kind = "line"
        elif bbox.width <= GLYPH_MAX_SIZE and bbox.height <= GLYPH_MAX_SIZE:
            # A small filled rect inside a graphic is a letterform - the stem of
            # an "I", the bar of an "F" - not a page rule. Treating it as artwork
            # keeps it with the rest of the logo when the page mirrors; left as a
            # rule it is mirrored alone and lands on top of other content.
            kind = "path"
        else:
            kind = "rect"

        out.append(
            DrawingElement(
                bbox=bbox,
                kind=kind,
                color=_float_color(stroke) if stroke else None,
                width=float(d.get("width") or 0.0),
                fill=_float_color(fill) if fill else None,
                items=items,
                close_path=bool(d.get("closePath")),
                even_odd=bool(d.get("even_odd", True)),
                fill_opacity=float(d.get("fill_opacity", 1.0) or 1.0),
                stroke_opacity=float(d.get("stroke_opacity", 1.0) or 1.0),
                line_cap=(d.get("lineCap") or [0])[0] if d.get("lineCap") else 0,
                line_join=float(d.get("lineJoin") or 0.0),
                dashes=d.get("dashes"),
                seqno=int(d.get("seqno") or 0),
            )
        )
    return out


def _bind_underlines(page_obj: Page) -> None:
    """Mark thin horizontal drawings that sit just under text as underlines and
    set the matching spans' underline flag."""
    for drawing in page_obj.drawings:
        if drawing.kind != "line" or drawing.bbox.height > UNDERLINE_MAX_THICKNESS:
            continue
        if drawing.bbox.width < 2:
            continue
        best: Optional[tuple[float, int, Span]] = None
        for bi, block in enumerate(page_obj.blocks):
            for span in block.spans:
                if not span.text.strip():
                    continue
                gap = drawing.bbox.y0 - span.bbox.y1
                if not (-2.0 <= gap <= UNDERLINE_MAX_GAP):
                    continue
                overlap = min(span.bbox.x1, drawing.bbox.x1) - max(
                    span.bbox.x0, drawing.bbox.x0
                )
                if overlap <= 0:
                    continue
                ratio = overlap / max(span.bbox.width, 1e-6)
                if ratio < UNDERLINE_MIN_OVERLAP:
                    continue
                score = abs(gap) - ratio  # prefer closest + best covered
                if best is None or score < best[0]:
                    best = (score, bi, span)
        if best is not None:
            _, block_index, span = best
            drawing.kind = "underline"
            drawing.underline_for = block_index
            span.underline = True


def _is_directional_image(bbox: BBox, data: bytes) -> bool:
    """Wide, short images are usually arrows / progression graphics, so they are
    the ones worth offering a horizontal flip for."""
    if bbox.height <= 0:
        return False
    aspect = bbox.width / bbox.height
    return aspect >= 2.5 and bbox.height < 120


def _extract_images(doc: fitz.Document, page: fitz.Page, qa: QAReport) -> list[ImageElement]:
    images: list[ImageElement] = []
    seen: set[tuple[int, tuple]] = set()
    for info in page.get_images(full=True):
        xref = info[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        if not rects:
            continue
        try:
            raw = doc.extract_image(xref)
            data, ext = raw["image"], raw["ext"]
        except Exception as exc:
            qa.add(
                "image",
                "warning",
                f"Image xref {xref} on page {page.number + 1} could not be extracted "
                f"and was left in place.",
                page=page.number + 1,
                error=str(exc),
            )
            continue
        for rect in rects:
            key = (xref, (round(rect.x0, 2), round(rect.y0, 2)))
            if key in seen:
                continue
            seen.add(key)
            bbox = BBox(rect.x0, rect.y0, rect.x1, rect.y1)
            images.append(
                ImageElement(
                    bbox=bbox,
                    data=data,
                    ext=ext,
                    xref=xref,
                    directional=_is_directional_image(bbox, data),
                )
            )
    return images


def extract_pdf(path: str, qa: QAReport) -> Document:
    """Read every text span with its style + position, plus images and vectors."""
    doc = Document(source_path=path, kind="pdf")
    with fitz.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("This PDF is password protected and cannot be read.")
        for page in pdf:
            page_obj = Page(
                number=page.number,
                width=page.rect.width,
                height=page.rect.height,
                rotation=page.rotation,
            )
            raw = page.get_text("dict")
            for block in raw.get("blocks", []):
                if block.get("type") != 0:  # 0 = text
                    continue
                tb = TextBlock(
                    bbox=BBox.from_tuple(block["bbox"]),
                    block_no=block.get("number", 0),
                )
                for line in block.get("lines", []):
                    ln = Line(bbox=BBox.from_tuple(line["bbox"]))
                    for span in line.get("spans", []):
                        # PyMuPDF hands back U+00A0 wherever an Arabic face maps
                        # its space glyph there. Normalising at the point of
                        # entry keeps every later stage - translation, wrapping,
                        # measurement - working with real word boundaries.
                        text = normalize_spaces(span.get("text", ""))
                        if not text:
                            continue
                        font = span.get("font", "")
                        flags = int(span.get("flags", 0))
                        ln.spans.append(
                            Span(
                                text=text,
                                font=font,
                                size=float(span.get("size", 11.0)),
                                color=_int_color(int(span.get("color", 0))),
                                bold=_looks_bold(font, flags),
                                italic=_looks_italic(font, flags),
                                underline=False,  # resolved by _bind_underlines
                                bbox=BBox.from_tuple(span["bbox"]),
                                origin=tuple(span.get("origin", (0.0, 0.0))),
                                # Decided by this span's own script, so an
                                # English label inside an Arabic page is not
                                # dragged across the page with the body text.
                                mirror=is_arabic(text),
                            )
                        )
                    if ln.spans:
                        tb.lines.append(ln)
                if tb.lines:
                    tb.mirror = block_is_rtl(tb)
                    page_obj.blocks.append(tb)

            page_obj.images = _extract_images(pdf, page, qa)
            page_obj.drawings = _extract_drawings(page)
            _bind_underlines(page_obj)
            doc.pages.append(page_obj)

    doc.image_count = sum(len(p.images) for p in doc.pages)
    if not any(p.blocks for p in doc.pages):
        qa.add(
            "extraction",
            "error",
            "No selectable text was found in this PDF. It is probably a scan; "
            "run OCR on it before translating.",
        )
    return doc


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
def _run_color(run) -> Optional[tuple[int, int, int]]:
    try:
        col = run.font.color
        if col is None or col.rgb is None:
            return None
        rgb: RGBColor = col.rgb
        return (rgb[0], rgb[1], rgb[2])
    except Exception:
        return None


def _collect_paragraphs(paragraphs, start_index: int, location: str) -> list[DocxParagraph]:
    out: list[DocxParagraph] = []
    for offset, para in enumerate(paragraphs):
        dp = DocxParagraph(
            index=start_index + offset,
            style_name=para.style.name if para.style is not None else "Normal",
            alignment=para.alignment,
            location=location,
        )
        for ri, run in enumerate(para.runs):
            size = run.font.size.pt if run.font.size is not None else None
            dp.runs.append(
                DocxRun(
                    para_index=dp.index,
                    run_index=ri,
                    text=run.text,
                    bold=run.bold,
                    italic=run.italic,
                    underline=bool(run.underline) if run.underline is not None else None,
                    font_name=run.font.name,
                    font_size=size,
                    color=_run_color(run),
                    location=location,
                )
            )
        out.append(dp)
    return out


def iter_docx_paragraphs(docx_file) -> list:
    """Body paragraphs, then every table cell, then headers/footers - the same
    order used when writing translations back, so indices line up."""
    paras = list(docx_file.paragraphs)
    for table in docx_file.tables:
        for row in table.rows:
            for cell in row.cells:
                paras.extend(cell.paragraphs)
    for section in docx_file.sections:
        for container in (section.header, section.footer):
            if container is not None:
                paras.extend(container.paragraphs)
    return paras


def extract_docx(path: str, qa: QAReport) -> Document:
    """DOCX already separates structure from styling, so this is a direct read."""
    doc = Document(source_path=path, kind="docx")
    f = DocxFile(path)

    body = _collect_paragraphs(f.paragraphs, 0, "body")
    doc.paragraphs.extend(body)
    idx = len(body)

    for table in f.tables:
        for row in table.rows:
            for cell in row.cells:
                cell_paras = _collect_paragraphs(cell.paragraphs, idx, "table")
                doc.paragraphs.extend(cell_paras)
                idx += len(cell_paras)

    for section in f.sections:
        for container, name in ((section.header, "header"), (section.footer, "footer")):
            if container is None:
                continue
            hp = _collect_paragraphs(container.paragraphs, idx, name)
            doc.paragraphs.extend(hp)
            idx += len(hp)

    try:
        doc.image_count = sum(
            1 for r in f.part.package.iter_parts()
            if "image" in getattr(r, "content_type", "")
        )
    except Exception:
        doc.image_count = 0

    if not any(p.text.strip() for p in doc.paragraphs):
        qa.add("extraction", "error", "No text was found in this DOCX file.")
    return doc


def extract(path: str, qa: QAReport) -> Document:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path, qa)
    if ext == ".docx":
        return extract_docx(path, qa)
    raise ValueError(f"Unsupported file type '{ext}'. Only PDF and DOCX are supported.")

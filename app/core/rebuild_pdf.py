"""Rebuild a PDF in place: erase original content, redraw translated content.

The file never leaves PDF format. Going PDF -> DOCX -> PDF reflows everything
and destroys the original layout, so instead each page is edited natively:

  1. redact the original text and images away (removes the real content, not
     just a white box over it),
  2. redraw the translated text, images and rules at their new coordinates.
"""
from __future__ import annotations

import io
from typing import Optional

import fitz

from . import fonts as fontlib
from .mirror import (
    MirrorMode,
    flip_image_bytes,
    is_anchored_label,
    is_contact_detail,
    mirror_bbox,
)
from .models import BBox, Document, DrawingElement, ImageElement, Page, Span, TextBlock
from .qa import QAReport
from .language import is_already_target, should_preserve
from .shape_arabic import (
    contains_arabic,
    normalize_block_text,
    normalize_spaces,
    shape,
    shape_lines,
    shape_for_render,
)

MIN_SIZE_RATIO = 0.80   # never shrink below 80% of the original size
MAX_BOX_GROWTH = 1.6    # a box may widen up to 60% to avoid an ugly wrap
# A short caption may grow much further: it is anchored beside its icon and a
# wrapped caption drops its tail onto the content below. Widening is still
# bounded by what the neighbouring elements leave free, so this is a ceiling
# rather than a licence to overlap.
LABEL_BOX_GROWTH = 6.0
LINE_HEIGHT = 1.35      # PyMuPDF's default line advance factor
SIZE_STEP = 0.25
BOX_PADDING = 1.0
# A block that ends up smaller than this share of its original size is flagged
# for manual review. It sits inside MIN_SIZE_RATIO on purpose: shrinking stops
# at MIN_SIZE_RATIO, so review has to trigger before that floor or it never
# triggers at all.
SHRINK_REVIEW_RATIO = 0.90
# Ignore paper-thin contacts; only a real collision is worth reporting.
OVERLAP_REPORT_RATIO = 0.15
# Breathing room kept between a table cell's ruling line and the text inside
# it, so a clamped translation never touches the border it must not cross.
CELL_INSET = 1.5
# A block counts as belonging to a cell when this much of its area lies inside
# it. Source text is often drawn a hair over its own ruling line, so the test
# cannot demand full containment.
CELL_CONTAINMENT = 0.6


def _norm(c: tuple[int, int, int]) -> tuple[float, float, float]:
    return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)


def _register_font(page: fitz.Page, resolved: fontlib.ResolvedFont) -> str:
    """Embed the font once per page and return the handle used by insert_textbox."""
    try:
        page.insert_font(fontname=resolved.name, fontfile=resolved.path)
    except Exception:
        pass
    return resolved.name


def _wrap_lines(
    text: str, font_path: str, size: float, width: float, shaped: bool = False
) -> list[str]:
    """Greedy word wrap on logical text, returning logical lines.

    `shaped=True` measures each candidate line in its *rendered* form. Arabic
    presentation forms are roughly a third narrower than the unjoined letters,
    so measuring the logical string would break lines far too early.
    """
    measure_of = (lambda t: fontlib.measure(shape(t), font_path, size)) if shaped \
        else (lambda t: fontlib.measure(t, font_path, size))

    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        for word in paragraph.split(" "):
            candidate = f"{current} {word}".strip()
            if current and measure_of(candidate) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def _try_draw(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontname: str,
    font_path: str,
    size: float,
    color: tuple[float, float, float],
    align: int,
) -> float:
    """Draw `text` into `rect`, returning PyMuPDF's leftover vertical space.

    A negative result means the text did not fit *and nothing was written* -
    PyMuPDF drops the whole block rather than clipping it. Callers must check
    the sign and retry smaller, or the content silently disappears.
    """
    return page.insert_textbox(
        rect,
        text,
        fontname=fontname,
        fontfile=font_path,
        fontsize=size,
        color=color,
        align=align,
    )


def _find_cells(page_obj: Page, source: Optional["fitz.Page"]) -> list[BBox]:
    """The table cells on this page, in the coordinates the blocks now use.

    A cell is the one box on the page that text is genuinely forbidden to
    leave: its meaning comes from the row and column it sits in, so a
    translation that spills over a ruling line does not just look wrong, it
    says something the source did not. Everywhere else on a page, growing into
    neighbouring whitespace is the lesser evil - inside a grid it is the worst
    one.

    PyMuPDF finds the cells from the ruling lines. A file that cannot be read,
    or a page with no grid, simply yields no cells and the ordinary layout
    rules apply unchanged.
    """
    if source is None:
        return []
    try:
        tables = source.find_tables()
    except Exception:
        return []
    cells: list[BBox] = []
    for table in tables.tables:
        for cell in getattr(table, "cells", []) or []:
            if cell is None:
                continue
            box = BBox(*cell)
            if box.width > 2 and box.height > 2:
                cells.append(box)
    return cells


def _cell_for(box: Optional[BBox], cells: list[BBox]) -> Optional[BBox]:
    """The cell this block belongs to, if any.

    The smallest cell holding most of the block wins: cells nest (a table's own
    bbox can be reported alongside its cells), and the innermost one is the
    boundary that actually matters.
    """
    if box is None or not cells:
        return None
    area = max(box.width * box.height, 0.01)
    best: Optional[BBox] = None
    for cell in cells:
        overlap_w = min(box.x1, cell.x1) - max(box.x0, cell.x0)
        overlap_h = min(box.y1, cell.y1) - max(box.y0, cell.y0)
        if overlap_w <= 0 or overlap_h <= 0:
            continue
        if (overlap_w * overlap_h) / area < CELL_CONTAINMENT:
            continue
        if best is None or cell.width * cell.height < best.width * best.height:
            best = cell
    return best


def _clamp_to_cell(rect: fitz.Rect, cell: Optional[BBox]) -> fitz.Rect:
    """Trim a candidate box back inside its cell's ruling lines."""
    if cell is None:
        return rect
    x0 = max(rect.x0, cell.x0 + CELL_INSET)
    y0 = max(rect.y0, cell.y0 + CELL_INSET)
    x1 = min(rect.x1, cell.x1 - CELL_INSET)
    y1 = min(rect.y1, cell.y1 - CELL_INSET)
    if x1 - x0 < 2 or y1 - y0 < 2:
        # The inset is wider than the cell itself - a very tight grid. Use the
        # cell as it stands rather than handing back an empty box.
        return fitz.Rect(max(rect.x0, cell.x0), max(rect.y0, cell.y0),
                         min(rect.x1, cell.x1), min(rect.y1, cell.y1))
    return fitz.Rect(x0, y0, x1, y1)


def _widen(
    rect: fitz.Rect, needed: float, max_x0: float, max_x1: float, align: int
) -> Optional[fitz.Rect]:
    """Grow a box horizontally to `needed` points, staying inside the space
    that neighbouring elements leave free. Right-aligned text grows leftwards,
    left-aligned text grows rightwards."""
    if align == fitz.TEXT_ALIGN_RIGHT:
        x0 = max(rect.x1 - needed, max_x0)
        widened = fitz.Rect(x0, rect.y0, rect.x1, rect.y1)
    else:
        x1 = min(rect.x0 + needed, max_x1)
        widened = fitz.Rect(rect.x0, rect.y0, x1, rect.y1)
    return widened if widened.width >= needed - 0.5 else None


def _draw_fitted(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontname: str,
    font_path: str,
    start_size: float,
    color: tuple[float, float, float],
    align: int,
    max_y: float,
    max_x0: float,
    max_x1: float,
    page_bottom: float,
    arabic: bool = False,
    label: bool = False,
    cell: Optional[BBox] = None,
) -> tuple[float, bool, fitz.Rect]:
    """Draw `text` at the largest size that actually fits.

    `text` is always the *logical* string. For Arabic the renderable form is
    derived here - wrapped for the box being tried, then shaped line by line -
    because both the wrap points and the shaped width change with every size
    and box the search considers.

    Returns (size_used, fitted, rect_used). Shrinks to MIN_SIZE_RATIO of the
    original, then as a last resort grows the box downwards so long
    translations spill instead of vanishing.

    `cell` closes off that last resort. Text inside a table cell may not leave
    it in any direction: the cell it lands in *is* its meaning, so a spill into
    the neighbour reads as a different row or column. Inside a cell the search
    keeps shrinking instead, down to a hard floor, and every candidate box is
    trimmed back to the ruling lines before it is tried.
    """
    in_cell = cell is not None
    # A cell may shrink harder than the page floor allows: a cramped cell is
    # readable, a cell whose text sits on top of the next one is not.
    floor = max(start_size * (0.5 if in_cell else MIN_SIZE_RATIO), 4.0)
    rect = _clamp_to_cell(rect, cell)

    def render(box: fitz.Rect, size: float) -> str:
        if not arabic:
            return text
        return shape_lines(
            _wrap_lines(text, font_path, size, max(box.width - 2, 10), shaped=True)
        )

    def attempt(box: fitz.Rect, size: float) -> bool:
        return _try_draw(
            page, box, render(box, size), fontname, font_path, size, color, align
        ) >= 0

    # A single line that is only slightly too wide should widen its box rather
    # than wrap. PyMuPDF lays wrapped lines out top-to-bottom in logical order,
    # which for already-bidi'd right-to-left text puts the phrase in the wrong
    # visual order - so avoiding an unnecessary wrap also avoids scrambling it.
    # Arabic faces have far taller line metrics than the Latin ones the source
    # box was sized for, so give the box the height one line actually needs
    # before considering any shrinking.
    needed_height = fontlib.line_height(font_path, start_size)
    if rect.height < needed_height:
        # max_y is the top of the next element, so growing never pushes this
        # block over its neighbour - documents whose source lines already sit
        # closer together than the Arabic line height simply shrink instead.
        taller = _clamp_to_cell(fitz.Rect(
            rect.x0, rect.y0, rect.x1,
            min(rect.y0 + needed_height, max_y),
        ), cell)
        if taller.height > rect.height:
            rect = taller

    # Widening is how a caption avoids an ugly wrap - it takes room from its
    # neighbours. A cell has no room to take: everything beside it is another
    # cell, so a widened box would print straight over the ruling line.
    if "\n" not in text and not in_cell:
        one_line = shape(text) if arabic else text
        needed = fontlib.measure(one_line, font_path, start_size)
        # A short caption is sized to the words it held in the source, and a
        # translation of it is routinely twice as wide - "Email us" becomes
        # "راسلنا عبر البريد الإلكتروني", 48pt of box for 108pt of text. The
        # ordinary growth cap is meant to stop body text sprawling sideways,
        # but applied to a caption it forces a wrap, and the wrapped tail lands
        # under whatever sits below. A label wraps to nothing: it either fits
        # on one line or it is not a label any more, so it may take whatever
        # room its neighbours actually leave free.
        cap = LABEL_BOX_GROWTH if label else MAX_BOX_GROWTH
        if rect.width < needed <= rect.width * cap:
            widened = _widen(rect, needed + 4, max_x0, max_x1, align)
            if widened is not None:
                # Give the widened box the height one line of this face needs
                # before testing it. Arabic line metrics are far taller than
                # the Latin ones the source box was cut for, so a box that is
                # wide enough can still reject the text for being too short -
                # and it then wraps, which is exactly what widening was meant
                # to avoid.
                # Give the widened box every point of vertical room its
                # neighbours leave free. `line_height` reports the font's own
                # metric, but insert_textbox wants more slack than that before
                # it will place a line - and one point short is indistinguish-
                # able from far too short: it refuses, and the caption wraps.
                if widened.height < max_y - widened.y0:
                    widened = fitz.Rect(widened.x0, widened.y0, widened.x1,
                                        max(max_y, widened.y1))
                if attempt(widened, start_size):
                    return start_size, True, widened

    size = start_size
    while size >= floor:
        if attempt(rect, size):
            return size, size >= start_size, rect
        size -= SIZE_STEP

    if in_cell:
        # Out of room inside the grid. The text is drawn at the floor size and
        # clipped by the cell rather than allowed to escape it; QA reports the
        # shortfall to the user, who can widen the column in the source.
        _try_draw(page, rect, render(rect, floor), fontname, font_path,
                  floor, color, align)
        return floor, False, rect

    grown = fitz.Rect(
        rect.x0, rect.y0, rect.x1,
        min(rect.y1 + rect.height * 1.5 + 24, max_y),
    )
    if grown.y1 <= rect.y1:
        grown = rect
    if grown.height > rect.height:
        size = start_size
        while size >= floor:
            if attempt(grown, size):
                return size, False, grown
            size -= SIZE_STEP

    # Nothing fits inside the space the neighbours allow. Losing a heading is
    # worse than a cramped one, so as a last resort the box is allowed to run
    # past its ceiling - down the page if there is room, else back up into its
    # own margin - and the shortfall is reported to the user by QA.
    for box in (
        fitz.Rect(rect.x0, rect.y0, rect.x1,
                  min(rect.y0 + rect.height * 4 + 48, page_bottom)),
        fitz.Rect(max_x0, rect.y0, max_x1,
                  min(rect.y0 + rect.height * 4 + 48, page_bottom)),
    ):
        if box.height <= rect.height and box.width <= rect.width:
            continue
        for size in (start_size, floor):
            if _try_draw(page, box, render(box, size), fontname, font_path,
                         size, color, align) >= 0:
                return size, False, box

    _try_draw(page, grown, render(grown, floor), fontname, font_path,
              floor, color, align)
    return floor, False, grown


def _draw_block(
    page: fitz.Page,
    page_obj: Page,
    block: TextBlock,
    direction: str,
    mode: MirrorMode,
    qa: QAReport,
    underline_enabled: bool,
    max_y: float,
    max_x0: float,
    max_x1: float,
    cell: Optional[BBox] = None,
) -> Optional[BBox]:
    """Draw one block. Returns the box it actually occupied, so the caller can
    check the finished layout for collisions.

    `cell` is the table cell this block sits in, when it sits in one; it is a
    hard boundary the drawn text may not cross."""
    text = block.translated if block.translated is not None else block.text
    # Belt and braces: a translation provider can return no-break spaces of its
    # own, and _wrap_lines splits on U+0020 only. Per-line leading and trailing
    # spaces are stripped at the same time - insert_textbox honours them, so a
    # source line indented with a space redraws with a ragged left edge.
    text = normalize_block_text(text)
    if not text or not text.strip():
        return None

    style = block.dominant_style()
    target_arabic = direction == "en2ar"

    # Font, shaping and alignment all follow the text actually being drawn -
    # never the requested direction. Content already in the target language is
    # left exactly as it was: same script, same alignment, no reshaping.
    use_arabic_font = contains_arabic(text)

    resolved = fontlib.resolve_for_text(
        text,
        bold=style.bold,
        italic=style.italic,
        original_font=style.font,
        qa=qa,
    )
    # Judge against the *source* text: a block whose original was already in
    # the target language was preserved deliberately, while a block that stayed
    # in the source language means the translation did not happen.
    source_text = block.text
    if source_text.strip():
        # Only text that was *already* in the target language counts as
        # preserved. Text that came back unchanged because the translation
        # failed is a different thing entirely and is reported as such below.
        if is_already_target(source_text, direction):
            qa.add(
                "preserved",
                "info",
                f"A segment on page {page_obj.number + 1} was already in the "
                f"target language and was left untouched.",
                page=page_obj.number + 1,
                excerpt=source_text[:80],
            )
        elif (direction == "en2ar") != use_arabic_font:
            qa.add(
                "translation",
                "warning",
                f"A segment on page {page_obj.number + 1} is still in the source "
                f"language and was left in its original script.",
                page=page_obj.number + 1,
                excerpt=source_text[:80],
            )
    fontname = _register_font(page, resolved)

    box = block.bbox or BBox(0, 0, page_obj.width, page_obj.height)
    # Give the box a little slack - translated text rarely matches source length
    # - while keeping it inside the page so nothing is drawn off the edge.
    rect = fitz.Rect(
        max(box.x0 - BOX_PADDING, 0),
        max(box.y0 - BOX_PADDING, 0),
        min(box.x1 + BOX_PADDING, page_obj.width - 1),
        min(box.y1 + BOX_PADDING, page_obj.height - 1),
    )
    rect = _clamp_to_cell(rect, cell)
    if rect.width <= 2 or rect.height <= 2:
        return None

    # Arabic is right-aligned, English left-aligned - decided by the script of
    # this block's own text, so a Latin caption inside an Arabic document keeps
    # reading left-to-right (and vice versa).
    align = fitz.TEXT_ALIGN_RIGHT if use_arabic_font else fitz.TEXT_ALIGN_LEFT

    # The size on the source span was chosen for the script that span was
    # written in. When an Arabic block is replaced by English, that size is
    # Arabic-tuned and renders the Latin text too large, so it is scaled back.
    # Judged from the source text's own script rather than from `direction`,
    # so an English caption that was always English keeps its author's size.
    source_is_arabic = contains_arabic(block.text)
    start_size = fontlib.adjusted_size(
        style.size, use_arabic_font, source_is_arabic=source_is_arabic
    )
    # The logical string is handed to _draw_fitted, which wraps and shapes it
    # for whichever box and size it settles on.
    # A caption anchored beside an icon must not wrap: its tail would land on
    # whatever sits below it. Judged on the *source* text, so a label stays a
    # label however long its translation turns out to be.
    is_label = is_anchored_label(block, page_obj) or is_contact_detail(block.text)
    size, fitted, rect = _draw_fitted(
        page, rect, text, fontname, resolved.path, start_size,
        _norm(style.color), align, max_y, max_x0, max_x1,
        page_obj.height - 2, arabic=use_arabic_font,
        label=is_label and cell is None, cell=cell,
    )
    display = shape(text) if use_arabic_font else shape_for_render(text, direction)

    if size < start_size:
        qa.overflow(page_obj.number + 1, text, start_size, size)
        # Shrinking past the review threshold means the box can no longer hold
        # the translation at a size close to the original - a layout call the
        # pipeline surfaces rather than makes on its own.
        if start_size > 0 and size < start_size * SHRINK_REVIEW_RATIO:
            qa.aggressive_shrink(page_obj.number + 1, text, start_size, size,
                                 1.0 - SHRINK_REVIEW_RATIO)
    if not fitted and size <= max(start_size * MIN_SIZE_RATIO, 4.0):
        qa.clipped(page_obj.number + 1, text)

    # Shrinking bottoms out at MIN_SIZE_RATIO; past that the box is grown
    # instead so the text is never dropped. A box that had to grow beyond its
    # designed area is the honest signal that the translation did not fit, so
    # it is reported even when the font size itself barely moved.
    if cell is not None and size < start_size * SHRINK_REVIEW_RATIO:
        qa.add(
            "layout_review",
            "warning",
            f"Translated text in a table cell on page {page_obj.number + 1} was "
            f"shrunk to fit its cell rather than allowed to overflow into the "
            f"cells beside it - widen that column in the source if it reads "
            f"too small.",
            page=page_obj.number + 1,
            excerpt=text[:80],
            size=round(size, 2),
            original_size=round(start_size, 2),
        )

    original_area = max(box.width * box.height, 1.0)
    grown_share = (rect.get_area() / original_area) if original_area else 1.0
    if grown_share > 1.0 + (1.0 - SHRINK_REVIEW_RATIO):
        qa.add(
            "layout_review",
            "warning",
            f"Translated text on page {page_obj.number + 1} needed "
            f"{(grown_share - 1) * 100:.0f}% more space than the original box "
            f"and was given room to spill - check this block by hand.",
            page=page_obj.number + 1,
            excerpt=text[:80],
            growth=round(grown_share, 3),
        )

    if underline_enabled and any(s.underline for s in block.spans):
        _draw_fitted_underline(page, rect, display, resolved, size, style, align)

    return BBox(rect.x0, rect.y0, rect.x1, rect.y1)


def _draw_fitted_underline(
    page: fitz.Page,
    rect: fitz.Rect,
    display: str,
    resolved: fontlib.ResolvedFont,
    size: float,
    style: Span,
    align: int,
) -> None:
    """Underline stretched to the *translated* text width, not the original's."""
    first_line = display.split("\n")[0]
    width = min(fontlib.measure(first_line, resolved.path, size), rect.width)
    y = rect.y0 + size * 1.15
    if y > rect.y1:
        y = rect.y1
    if align == fitz.TEXT_ALIGN_RIGHT:
        x1 = rect.x1
        x0 = max(rect.x1 - width, rect.x0)
    else:
        x0 = rect.x0
        x1 = min(rect.x0 + width, rect.x1)
    page.draw_line(
        fitz.Point(x0, y), fitz.Point(x1, y),
        color=_norm(style.color), width=max(size * 0.05, 0.4),
    )


def _draw_image(page: fitz.Page, image: ImageElement, flip_directional: bool,
                qa: QAReport, page_number: int) -> None:
    data = image.data
    if not data:
        return
    if flip_directional and image.directional:
        data = flip_image_bytes(data)
        qa.add(
            "image",
            "info",
            f"A directional graphic on page {page_number} was flipped "
            f"horizontally to match the new reading direction.",
            page=page_number,
            bbox=image.bbox.as_tuple(),
        )
    rect = fitz.Rect(*image.bbox.as_tuple())
    if rect.width <= 0 or rect.height <= 0:
        return
    try:
        page.insert_image(rect, stream=data, keep_proportion=False)
    except Exception as exc:
        qa.add(
            "image",
            "error",
            f"An image on page {page_number} could not be redrawn after mirroring.",
            page=page_number,
            error=str(exc),
        )


def _path_offset(drawing: DrawingElement, mode: MirrorMode,
                 page_width: float) -> float:
    """Horizontal shift recorded when the layout was mirrored.

    The artwork itself is never reflected - a mirrored logo reads backwards -
    and a multi-path graphic carries one shared shift so its parts keep their
    arrangement.
    """
    return drawing.shift_x


def _draw_drawing(page: fitz.Page, drawing: DrawingElement, mode: MirrorMode,
                  page_width: float, qa: Optional[QAReport] = None,
                  page_number: int = 1) -> None:
    """Replay a vector path.

    Logos and icons are Bezier artwork, so each segment is redrawn in turn.
    Approximating them by their bounding box - the old behaviour - turned every
    logo on the page into a solid coloured rectangle.
    """
    if drawing.kind == "underline":
        return  # redrawn with its text, sized to the translation

    fill = _norm(drawing.fill) if drawing.fill else None
    color = _norm(drawing.color) if drawing.color else None
    width = drawing.width if color else 0

    if drawing.items:
        dx = _path_offset(drawing, mode, page_width)

        def pt(p) -> fitz.Point:
            return fitz.Point(p.x + dx, p.y)

        shape = page.new_shape()
        try:
            for item in drawing.items:
                op = item[0]
                if op == "l":
                    shape.draw_line(pt(item[1]), pt(item[2]))
                elif op == "c":
                    shape.draw_bezier(pt(item[1]), pt(item[2]),
                                      pt(item[3]), pt(item[4]))
                elif op == "re":
                    rect = item[1]
                    shape.draw_rect(fitz.Rect(rect.x0 + dx, rect.y0,
                                              rect.x1 + dx, rect.y1))
                elif op == "qu":
                    quad = item[1]
                    shape.draw_quad(fitz.Quad(
                        pt(quad.ul), pt(quad.ur), pt(quad.lr), pt(quad.ll)))
            shape.finish(
                color=color,
                fill=fill,
                width=width,
                even_odd=drawing.even_odd,
                closePath=drawing.close_path,
                fill_opacity=drawing.fill_opacity,
                stroke_opacity=drawing.stroke_opacity,
                lineJoin=drawing.line_join,
                dashes=drawing.dashes,
            )
            shape.commit()
        except Exception as exc:
            # A path that cannot be replayed is skipped rather than drawn as a
            # block over the content beneath it - but the user is told, because
            # a missing logo is not something they should have to spot.
            if qa is not None:
                qa.add(
                    "artwork",
                    "warning",
                    f"A graphic on page {page_number} could not be redrawn and "
                    f"is missing from the translated file.",
                    page=page_number,
                    bbox=drawing.bbox.as_tuple(),
                    error=str(exc),
                )
        return

    # No path data (rare): fall back to the bounding box.
    rect = fitz.Rect(*drawing.bbox.as_tuple())
    if rect.width <= 0 and rect.height <= 0:
        return
    try:
        if drawing.kind == "line" or rect.height <= 2.5:
            y = (rect.y0 + rect.y1) / 2
            page.draw_line(
                fitz.Point(rect.x0, y), fitz.Point(rect.x1, y),
                color=color or (0, 0, 0), width=max(drawing.width, 0.3),
            )
        else:
            page.draw_rect(rect, color=color, fill=fill,
                           width=max(drawing.width, 0.3))
    except Exception:
        pass


def _overlap_share(a: BBox, b: BBox) -> float:
    """Intersection area as a share of the smaller box."""
    if not a.intersects(b):
        return 0.0
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    smaller = min(a.width * a.height, b.width * b.height)
    return (w * h) / smaller if smaller > 0 else 0.0


def _report_layout_collisions(
    page_obj: Page, drawn: list[tuple[BBox, TextBlock]], qa: QAReport
) -> None:
    """Flag translated blocks that ended up covering a neighbour or an image."""
    for i, (box, block) in enumerate(drawn):
        text = block.translated if block.translated is not None else block.text
        for other_box, _ in drawn[i + 1:]:
            if _overlap_share(box, other_box) > OVERLAP_REPORT_RATIO:
                qa.block_overlap(page_obj.number + 1, text, "text block",
                                 box.as_tuple(), other_box.as_tuple())
        for image in page_obj.images:
            if _overlap_share(box, image.bbox) > OVERLAP_REPORT_RATIO:
                qa.block_overlap(page_obj.number + 1, text, "image",
                                 box.as_tuple(), image.bbox.as_tuple())


# A filled shape covering this much of the page is its background, not artwork.
BACKGROUND_AREA_RATIO = 0.9


def _split_backgrounds(page_obj: Page) -> tuple[list, list]:
    """Separate full-page background fills from the artwork drawn on top.

    Order matters only for shapes big enough to hide something. A background
    is identified by area rather than by colour, so a coloured panel behind a
    section is handled the same way as a white sheet.
    """
    page_area = max(page_obj.width * page_obj.height, 1.0)
    backgrounds, foreground = [], []
    for drawing in page_obj.drawings:
        area = drawing.bbox.width * drawing.bbox.height
        if drawing.fill is not None and area >= page_area * BACKGROUND_AREA_RATIO:
            backgrounds.append(drawing)
        else:
            foreground.append(drawing)
    return backgrounds, foreground


def rebuild_pdf(
    doc: Document,
    output_path: str,
    direction: str,
    mode: MirrorMode,
    qa: QAReport,
    *,
    underline_enabled: bool = True,
    flip_directional_images: bool = False,
) -> str:
    """Write the translated PDF. `doc` must already be translated and mirrored."""
    pdf = fitz.open(doc.source_path)
    try:
        # Cells are read from the untouched source, before any redaction
        # strips the ruling lines they are inferred from.
        cells_by_page = {
            page_obj.number: _find_cells(page_obj, pdf[page_obj.number])
            for page_obj in doc.pages
            if page_obj.number < len(pdf)
        }
        for page_obj in doc.pages:
            page = pdf[page_obj.number]

            # 1. Erase the originals. Redaction removes the underlying content
            #    rather than painting over it, so no stale text is recoverable.
            for block in page_obj.blocks:
                src_bbox = block.bbox
                if src_bbox is None:
                    continue
                # Recover where this block was drawn in the source. A block that
                # was preserved never moved, so its box is already in source
                # coordinates and must not be un-mirrored.
                # An LTR block was re-anchored by the same whole-box reflection
                # as an RTL one, so both are un-mirrored the same way here. Only
                # a block frozen by the language guard never moved.
                if mode is MirrorMode.FULL and not is_already_target(
                    block.text, direction
                ):
                    src_bbox = mirror_bbox(src_bbox, page_obj.width)
                page.add_redact_annot(fitz.Rect(*src_bbox.as_tuple()))

            for image in page_obj.images:
                src_bbox = image.bbox
                if mode is MirrorMode.FULL:
                    src_bbox = mirror_bbox(src_bbox, page_obj.width)
                page.add_redact_annot(fitz.Rect(*src_bbox.as_tuple()))

            # The original vector art is removed by the redaction pass below
            # (graphics=REMOVE_IF_TOUCHED) rather than painted over. Covering it
            # with white rectangles left those rectangles visible as stray boxes
            # and tripled the number of paths in the file.
            if mode is MirrorMode.FULL:
                page.add_redact_annot(page.rect)
            else:
                for drawing in page_obj.drawings:
                    if drawing.kind == "underline":
                        rect = fitz.Rect(*drawing.bbox.as_tuple())
                        rect.y0 -= 1.0
                        rect.y1 += 1.0
                        page.add_redact_annot(rect)

            try:
                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_REMOVE,
                    graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                )
            except Exception as exc:
                qa.add(
                    "rebuild",
                    "warning",
                    f"Original content on page {page_obj.number + 1} could not be "
                    f"fully cleared; the translated text may overlap it.",
                    page=page_obj.number + 1,
                    error=str(exc),
                )

            artwork = [d for d in page_obj.drawings if d.kind == "path"]
            if artwork and mode is MirrorMode.FULL:
                qa.add(
                    "artwork",
                    "info",
                    f"Page {page_obj.number + 1} contains {len(artwork)} vector "
                    f"graphics (logos, icons). They were moved to their mirrored "
                    f"positions without being flipped - worth a visual check.",
                    page=page_obj.number + 1,
                    count=len(artwork),
                )

            # 2. Redraw at the new coordinates, back to front.
            #
            # Background panels first, then images, then the rest of the
            # artwork. A page whose background is drawn as a filled rectangle
            # covering the whole sheet would otherwise be replayed *after* the
            # photographs and screenshots that sit on it, painting them out -
            # the images survive in the file and report as placed, but nothing
            # of them is visible.
            backgrounds, foreground = _split_backgrounds(page_obj)
            for drawing in backgrounds:
                _draw_drawing(page, drawing, mode, page_obj.width,
                              qa, page_obj.number + 1)
            for image in page_obj.images:
                _draw_image(page, image, flip_directional_images, qa,
                            page_obj.number + 1)
            for drawing in foreground:
                _draw_drawing(page, drawing, mode, page_obj.width,
                              qa, page_obj.number + 1)
            # A block may grow downwards only into empty space, never into the
            # block below it.
            occupied = [b.bbox for b in page_obj.blocks if b.bbox] + \
                       [i.bbox for i in page_obj.images]
            # The blocks have already been mirrored, so the cells they are
            # matched against have to be mirrored the same way to still line
            # up with them.
            #
            # The cell recorded on a block was measured before mirroring, and
            # mirroring moves different blocks by different rules (a preserved
            # block does not move at all), so the box on the block is not
            # re-used here - only the fact that the block came from a cell.
            # Which cell it now sits in is decided from its final position.
            cells = cells_by_page.get(page_obj.number, [])
            if mode is MirrorMode.FULL:
                cells = [mirror_bbox(c, page_obj.width) for c in cells]
            drawn: list[tuple[BBox, TextBlock]] = []
            for block in page_obj.blocks:
                max_y = page_obj.height - 2
                max_x0, max_x1 = 1.0, page_obj.width - 1
                if block.bbox is not None:
                    for other in occupied:
                        if other is block.bbox:
                            continue
                        horizontal_overlap = (
                            min(block.bbox.x1, other.x1) - max(block.bbox.x0, other.x0)
                        )
                        # Clamp against the next element down. The test is
                        # "starts below this block's top" rather than "below its
                        # bottom", because tightly-led source lines already
                        # overlap each other and a grown box would swallow the
                        # line underneath.
                        #
                        # The ceiling is the neighbour's *baseline area* rather
                        # than its very top: dropping a line entirely is worse
                        # than letting descenders touch, and every remaining
                        # shortfall is reported to the user by QA.
                        if horizontal_overlap > 1 and other.y0 > block.bbox.y0 + 1:
                            ceiling = other.y0 + min(other.height * 0.5, 8.0)
                            max_y = min(max_y, ceiling)

                        # Neighbours on the same line bound how far this block
                        # may widen sideways.
                        vertical_overlap = (
                            min(block.bbox.y1, other.y1) - max(block.bbox.y0, other.y0)
                        )
                        if vertical_overlap > 1:
                            if other.x1 <= block.bbox.x0 + 1:
                                max_x0 = max(max_x0, other.x1 + 2)
                            elif other.x0 >= block.bbox.x1 - 1:
                                max_x1 = min(max_x1, other.x0 - 2)
                cell = _cell_for(block.bbox, cells)
                placed = _draw_block(page, page_obj, block, direction, mode, qa,
                                     underline_enabled, max_y, max_x0, max_x1,
                                     cell=cell)
                if placed is not None:
                    drawn.append((placed, block))

            # Boxes are only final once every block has been fitted: a block
            # that wrapped to more lines than its source may have grown past a
            # neighbour it did not originally touch. Nothing is moved at this
            # point - the text is already on the page - but the collision is
            # reported so it is never silently overprinted.
            _report_layout_collisions(page_obj, drawn, qa)

        pdf.save(output_path, garbage=3, deflate=True)
    finally:
        pdf.close()
    return output_path

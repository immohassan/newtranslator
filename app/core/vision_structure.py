"""Turn a vision-read layout into the DocBlocks the HTML pipeline renders.

`vision_layout.read_layout` says what the page's structure *is* - its columns,
and what each block is for. This module turns that answer into the block list
`html_pipeline` already knows how to translate and render, taking every
character of text from the extracted model rather than from the reply.

The seam is deliberate: `extract_structure` and `read_structure` have the same
signature and return the same type, so the vision reader is a drop-in for the
geometric one and everything downstream - translation, mirroring, rendering,
pagination - is shared. A page the model cannot read falls back to the
geometric path, which stays a first-class route rather than dead code.
"""
from __future__ import annotations

from typing import Optional

from .html_pipeline import (
    DocBlock,
    Fragment,
    _BULLET_RE,
    _ORDERED_RE,
    _body_size,
    _find_tables,
    _inside,
    extract_structure,
)
from .models import Page
from .qa import QAReport
from . import vision_layout


# A column narrower than this share of the page is a sidebar, whatever the
# model called it - the width is what decides how it is set.
SIDEBAR_MAX_SHARE = 0.38


def _fragments_for(block, strip_marker: bool) -> list[Fragment]:
    """Every span of a block, styling intact, as renderable fragments.

    The text is the extractor's, not the model's. A marker character is
    stripped only when the block is being set as a list item, because the
    renderer draws its own.
    """
    out: list[Fragment] = []
    first = True
    for line in block.lines:
        for span in line.spans:
            text = span.text
            if strip_marker and first and text.strip():
                text = _BULLET_RE.sub("", text, count=1)
                text = _ORDERED_RE.sub("", text, count=1)
            if not text:
                continue
            first = False
            out.append(Fragment(text=text, bold=span.bold, italic=span.italic,
                                size=span.size, color=span.color))
        # A block's lines are one flow; the space keeps the last word of a
        # line from running into the first word of the next.
        if out and not out[-1].text.endswith(" "):
            out[-1] = Fragment(text=out[-1].text + " ", bold=out[-1].bold,
                               italic=out[-1].italic, size=out[-1].size,
                               color=out[-1].color)
    return out


def _to_doc_block(entry: dict, page: Page,
                  tables: list = ()) -> Optional[DocBlock]:
    """One entry of the model's answer, as a renderable block.

    A block whose text is already inside a grid is dropped: the grid is
    emitted whole from its own ruling lines, and letting the cell text through
    here as well would print every cell twice - once in the table and once as
    a paragraph beside it.
    """
    block = page.blocks[entry["id"]]
    # Kept so the page's images can be slotted back in beside the text they
    # sit next to; the model is never asked about them.
    source_box = block.bbox
    kind = entry["kind"]

    if tables and any(_inside(source_box, box) for box, _, _ in tables):
        return None

    if kind == "rule":
        out = DocBlock(kind="rule")
        out.source_box = source_box
        return out

    listish = kind == "bullet"
    fragments = _fragments_for(block, strip_marker=listish)
    if not fragments:
        return None

    if kind == "heading":
        out = DocBlock(kind="heading", fragments=fragments,
                       level=min(max(int(entry.get("level") or 2), 1), 4))
        out.source_box = source_box
        return out
    if listish:
        ordered = bool(_ORDERED_RE.match(block.text))
        out = DocBlock(kind="bullet", fragments=fragments, ordered=ordered)
        out.source_box = source_box
        return out
    # A table row read as one block is set as a paragraph rather than being
    # rebuilt into a grid: the cells of a single row carry no column widths,
    # and inventing them is how a row becomes a mangled table.
    out = DocBlock(kind="paragraph", fragments=fragments)
    out.source_box = source_box
    return out


def _image_block(image) -> DocBlock:
    """One of the page's pictures, as a block the renderer can emit."""
    out = DocBlock(kind="image", image=image.data,
                   image_ext=image.ext or "png",
                   width=image.bbox.width, height=image.bbox.height)
    out.source_box = image.bbox
    return out


def _interleave_images(blocks: list[DocBlock], page: Page) -> list[DocBlock]:
    """Put the page's images back among the text, in reading order.

    The vision reader answers about text blocks alone, so an image reaches
    this point unplaced. Each is inserted before the first block that sits
    below it on the source page - which is where a reader met it - and any
    left over go at the end.
    """
    pictures = [image for image in page.images if image.data]
    if not pictures:
        return blocks

    out: list[DocBlock] = []
    pending = sorted(pictures, key=lambda im: (im.bbox.y0, im.bbox.x0))
    for block in blocks:
        box = getattr(block, "source_box", None)
        while pending and box is not None and pending[0].bbox.y0 <= box.y0:
            out.append(_image_block(pending.pop(0)))
        out.append(block)
    out.extend(_image_block(image) for image in pending)
    return out


def _splice_tables(blocks: list[DocBlock], tables: list) -> list[DocBlock]:
    """Put each grid back among the text blocks, in reading order.

    A table is placed before the first block that starts below its top edge,
    mirroring how `_interleave_images` places the page's pictures. Any grid
    that sits below every block goes at the end.
    """
    pending = sorted(tables, key=lambda t: t[0].y0)
    out: list[DocBlock] = []
    for block in blocks:
        box = getattr(block, "source_box", None)
        while pending and box is not None and pending[0][0].y0 <= box.y0:
            _, rows, header = pending.pop(0)
            out.append(DocBlock(kind="table", rows=rows, header=header))
        out.append(block)
    out.extend(DocBlock(kind="table", rows=rows, header=header)
               for _, rows, header in pending)
    return out


def _column_width(column: dict, page: Page) -> float:
    boxes = [page.blocks[b["id"]].bbox for b in column["blocks"]]
    if not boxes:
        return 0.0
    return max(b.x1 for b in boxes) - min(b.x0 for b in boxes)


def read_structure(page: Page, qa: QAReport, source, client=None) -> list[DocBlock]:
    """The page as the vision model reads it, falling back to the geometry.

    Signature-compatible with `html_pipeline.extract_structure`, so the two are
    interchangeable at the call site.
    """
    if not vision_layout.suits_vision_layout(page):
        return extract_structure(page, qa, source)

    columns = vision_layout.read_layout(page, source, qa, client=client)
    if not columns:
        # read_layout has already recorded why. The geometric reader is the
        # fallback rather than an error: it is what runs with no API key.
        return extract_structure(page, qa, source)

    # A grid is read from the page's ruling lines, not from the model: the
    # reader is asked where text sits, and a row it returns as one block
    # carries no column widths, so a table left to that path is flattened into
    # paragraphs and its cells lose the cells beside them - the one structure
    # that cannot survive being read as prose. The cells' own text is then
    # withheld from the column pass below, so the grid is not also emitted as
    # a run of paragraphs sitting underneath it.
    tables = _find_tables(page, source) if source is not None else []

    blocks: list[DocBlock] = []
    for column in columns:
        width = _column_width(column, page)
        sidebar = (column.get("role") == "sidebar"
                   or (width and width <= page.width * SIDEBAR_MAX_SHARE))
        column_blocks = [b for b in
                         (_to_doc_block(entry, page, tables) for entry in column["blocks"])
                         if b is not None]
        if not column_blocks:
            continue
        # The column is marked on its first and last block so the renderer can
        # wrap it. A single-column page carries no marks at all, and renders
        # exactly as it did before this module existed.
        if len(columns) > 1:
            column_blocks[0].column_start = "sidebar" if sidebar else "main"
            column_blocks[-1].column_end = True
        blocks.extend(column_blocks)

    # The grids go back among the text, each before the first block that sits
    # below it on the page - which is where a reader met it. A table is
    # full-width page furniture, so it carries no column marks and is placed
    # relative to the text rather than inside a column.
    if tables:
        blocks = _splice_tables(blocks, tables)

    if not blocks:
        # A page that is nothing but its grid - a rota, a schedule, a form -
        # leaves no text blocks behind, which is not a failure to read it.
        # Falling back here would hand the page to the geometric reader and
        # undo the grid that was just recovered.
        if tables:
            return [DocBlock(kind="table", rows=rows, header=header)
                    for _, rows, header in sorted(tables, key=lambda t: t[0].y0)]
        return extract_structure(page, qa, source)

    # The model is asked about text only, so nothing above has placed the
    # page's pictures - and without this every one of them is dropped, which
    # lost a CV its portrait and its contact icons. Each is put back beside
    # the text it sits next to on the page.
    blocks = _interleave_images(blocks, page)

    qa.add(
        "structure",
        "info",
        f"Page {page.number + 1} was read by the layout model as "
        f"{len(columns)} column(s), {len(blocks)} block(s).",
        page=page.number + 1,
        columns=len(columns),
        count=len(blocks),
        engine="vision",
    )
    return blocks

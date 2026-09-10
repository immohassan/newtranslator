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


def _to_doc_block(entry: dict, page: Page) -> Optional[DocBlock]:
    """One entry of the model's answer, as a renderable block."""
    block = page.blocks[entry["id"]]
    kind = entry["kind"]

    if kind == "rule":
        return DocBlock(kind="rule")

    listish = kind == "bullet"
    fragments = _fragments_for(block, strip_marker=listish)
    if not fragments:
        return None

    if kind == "heading":
        return DocBlock(kind="heading", fragments=fragments,
                        level=min(max(int(entry.get("level") or 2), 1), 4))
    if listish:
        ordered = bool(_ORDERED_RE.match(block.text))
        return DocBlock(kind="bullet", fragments=fragments, ordered=ordered)
    # A table row read as one block is set as a paragraph rather than being
    # rebuilt into a grid: the cells of a single row carry no column widths,
    # and inventing them is how a row becomes a mangled table.
    return DocBlock(kind="paragraph", fragments=fragments)


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

    blocks: list[DocBlock] = []
    for column in columns:
        width = _column_width(column, page)
        sidebar = (column.get("role") == "sidebar"
                   or (width and width <= page.width * SIDEBAR_MAX_SHARE))
        column_blocks = [b for b in
                         (_to_doc_block(entry, page) for entry in column["blocks"])
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

    if not blocks:
        return extract_structure(page, qa, source)

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

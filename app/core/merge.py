"""Merge visually-adjacent text blocks into logical lines before translation.

Designed documents routinely split one sentence across several positioned
blocks - a number in its own box, a phrase set in a different weight, a clause
positioned to wrap around an icon. Translating each piece alone produces
fragments that no longer form a sentence in the target language, and reflowing
them independently scatters them across the page.

Merging by reading order first means the translator sees whole sentences, and
the merged run is redrawn as a single block.
"""
from __future__ import annotations

import re
from typing import Optional

from .mirror import block_is_rtl
from .models import BBox, Line, Page, TextBlock

# Blocks whose vertical centres differ by less than this share a line.
ROW_TOLERANCE = 0.6      # as a fraction of the smaller block's height
# Horizontal gap (in points) still considered "same sentence".
MAX_GAP = 40.0
# A block taller than this many lines is a paragraph, not a fragment.
MAX_MERGE_LINES = 2


def _same_row(a: TextBlock, b: TextBlock) -> bool:
    if a.bbox is None or b.bbox is None:
        return False
    centre_a = (a.bbox.y0 + a.bbox.y1) / 2
    centre_b = (b.bbox.y0 + b.bbox.y1) / 2
    limit = min(a.bbox.height, b.bbox.height) * ROW_TOLERANCE
    return abs(centre_a - centre_b) <= limit


def _gap(a: TextBlock, b: TextBlock) -> float:
    """Horizontal space between two blocks on the same row."""
    if a.bbox.x1 <= b.bbox.x0:
        return b.bbox.x0 - a.bbox.x1
    if b.bbox.x1 <= a.bbox.x0:
        return a.bbox.x0 - b.bbox.x1
    return 0.0  # overlapping: definitely the same visual line


def _mergeable(block: TextBlock) -> bool:
    """Only short runs are merged. A real paragraph already reads correctly."""
    return bool(block.text.strip()) and len(block.lines) <= MAX_MERGE_LINES


# A single letter stranded after terminal punctuation is a stray glyph left
# behind in the source file, not content. Dropping it avoids translating it
# into a meaningless letter of the target alphabet.
_TRAILING_STRAY_RE = re.compile(r"(?<=[.!?؟])\s+\S\s*$")


def strip_stray_glyphs(page: Page) -> int:
    """Remove one-character leftovers that trail a finished sentence.

    Some authoring tools leave an orphaned glyph in the content stream. It
    renders as a stray letter and, once translated, as a stray letter of the
    other alphabet.
    """
    removed = 0
    for block in page.blocks:
        for line in block.lines:
            if len(line.spans) < 2:
                continue
            last = line.spans[-1]
            if len(last.text.strip()) != 1:
                continue
            earlier = "".join(sp.text for sp in line.spans[:-1]).rstrip()
            if earlier.endswith((".", "!", "?", "؟")):
                line.spans.pop()
                removed += 1
        block.lines = [l for l in block.lines if l.spans]
    page.blocks = [b for b in page.blocks if b.lines]
    return removed


# Characters that authoring tools emit as a standalone list marker.
BULLET_CHARS = set("\u2022\u00b7\u25cf\u25aa\u25a0\u25e6\u2023\u2043\u2219-*\u06de")
# A marker sits within this multiple of the text height to its side.
BULLET_ROW_TOLERANCE = 0.8
# ...and within this many points horizontally of the text it introduces.
BULLET_MAX_GAP = 48.0


def is_bullet_text(text: str) -> bool:
    """Whether a run of text is nothing but a list marker.

    Numbered markers ("1.", "a)") count too: they are laid out the same way and
    suffer the same problem when separated from their item.
    """
    stripped = (text or "").strip()
    if not stripped or len(stripped) > 4:
        return False
    if stripped in BULLET_CHARS or all(c in BULLET_CHARS for c in stripped):
        return True
    return bool(re.fullmatch(r"[0-9]{1,2}[.)]|[a-zA-Z][.)]", stripped))


# The marker every list item is rewritten with, and the gap after it. Source
# markers are discarded rather than repositioned: they arrive at whatever size,
# baseline and offset the producer chose - often a 5pt glyph raised above the
# line - and no amount of nudging makes a set of those line up. One consistent
# marker, laid out as part of the text run, always does.
BULLET_MARKER = "\u2022"
BULLET_SEPARATOR = " "
# A leading marker glued to its text ("\u00b7Free training materials"), which is
# how the producer's own layout reaches extraction when the two runs touch.
_LEADING_MARKER_RE = re.compile(
    r"^\s*(?:[" + "".join(re.escape(c) for c in sorted(BULLET_CHARS)) + r"]"
    r"|[0-9]{1,2}[.)]|[a-zA-Z][.)])\s*"
)


# A numbered marker carries information a bullet does not - the ordering - so
# it is normalised in place rather than replaced.
_ORDERED_RE = re.compile(r"^\s*([0-9]{1,2}|[a-zA-Z])([.)])\s*")


def canonical_marker(text: str) -> str:
    """The marker to write for an item whose source marker was `text`.

    Bulleted items all get the same glyph. A numbered item keeps its own
    number and delimiter, because replacing "2." with a bullet would throw the
    sequence away.
    """
    match = _ORDERED_RE.match(text or "")
    if match:
        return f"{match.group(1)}{match.group(2)}"
    # A tab-only marker carries no glyph of its own (the visible bullet was a
    # vector path, now removed), so it becomes the canonical bullet.
    return BULLET_MARKER


def strip_leading_marker(text: str) -> tuple[str, str, bool]:
    """Split a list marker off the head of `text`.

    Returns (marker, remaining text, found). The marker is returned so ordered
    lists can keep their numbering; bulleted ones discard it.
    """
    if not text:
        return "", text, False
    stripped = _LEADING_MARKER_RE.sub("", text, count=1)
    if stripped == text:
        return "", text, False
    # A line that was nothing but a marker is not a list item.
    if not stripped.strip():
        return "", text, False
    return canonical_marker(text), stripped, True


def _is_marker_line(line: Line) -> bool:
    """Whether a line is a list marker rather than content.

    A tab on a line of its own counts: where the marker itself was drawn as a
    vector path, the text layer carries only the tab that indented it, and that
    tab is what the item's text follows.
    """
    if not line.spans:
        return False
    if is_bullet_text(line.text):
        return True
    return line.text.strip("\t\u00a0 ") == "" and "\t" in line.text


# A vector list marker: a small, roughly square filled blob sitting beside a
# line of text. Real bullets are 3-8pt; anything larger is an icon.
VECTOR_BULLET_MAX = 9.0
VECTOR_BULLET_MIN = 1.2
# How square it has to be. A dot or a small square qualifies; a dash does not.
VECTOR_BULLET_ASPECT = 1.8
# It must sit within this many points of the text it introduces, horizontally.
VECTOR_BULLET_GAP = 40.0


def _looks_like_vector_bullet(drawing) -> bool:
    """Whether a drawn path is a list marker rather than artwork."""
    if drawing.kind not in ("path", "rect"):
        return False
    box = drawing.bbox
    w, h = box.width, box.height
    if not (VECTOR_BULLET_MIN <= w <= VECTOR_BULLET_MAX):
        return False
    if not (VECTOR_BULLET_MIN <= h <= VECTOR_BULLET_MAX):
        return False
    if max(w, h) / max(min(w, h), 0.01) > VECTOR_BULLET_ASPECT:
        return False
    return drawing.fill is not None


# Two lines of one block that share no horizontal span are not stacked text -
# they are separate items the extractor happened to group together.
CONTACT_SPLIT_OVERLAP = 2.0


def split_contact_lines(page: Page) -> int:
    """Split a block whose lines are side-by-side contact details.

    A footer often prints a URL under one icon and an email address under
    another. PyMuPDF can return the two as one block, because they share a
    band of the page - but they are separate items, positioned under different
    icons. Left joined they mirror as a unit, which carries each one away from
    the icon it belongs to.

    Only blocks whose lines are genuinely disjoint horizontally are split, so
    an ordinary wrapped paragraph is never broken up.
    """
    split = 0
    out: list[TextBlock] = []
    for block in page.blocks:
        parts = _disjoint_contact_lines(block)
        if parts is None:
            out.append(block)
            continue
        for line in parts:
            piece = TextBlock(lines=[line], bbox=line.bbox,
                              block_no=block.block_no)
            piece.mirror = block_is_rtl(piece)
            out.append(piece)
        split += 1
    page.blocks = out
    return split


def _disjoint_contact_lines(block: TextBlock) -> Optional[list[Line]]:
    """The block's lines, when they are separate items sitting side by side.

    A footer prints one label under each icon - "Learn more" beside a globe,
    "Email us" beside an envelope - and the extractor can return the row as one
    block because the labels share a horizontal band. They are separate items
    though: each is positioned under its own icon, and joined they are laid out
    (and mirrored) as a unit, which carries each away from what it labels.

    Contact details qualify, and so does any short label; running prose never
    does, and neither do lines that overlap horizontally - those are stacked
    text, not items side by side.
    """
    from .mirror import is_contact_detail, is_short_label

    lines = [l for l in block.lines if l.text.strip() and l.bbox]
    if len(lines) < 2 or len(lines) != len(block.lines):
        return None
    if not all(is_contact_detail(l.text) or is_short_label(l.text)
               for l in lines):
        return None
    # Every pair must sit in its own horizontal band.
    for i, first in enumerate(lines):
        for second in lines[i + 1:]:
            overlap = (min(first.bbox.x1, second.bbox.x1)
                       - max(first.bbox.x0, second.bbox.x0))
            if overlap > CONTACT_SPLIT_OVERLAP:
                return None
    return lines


def drop_vector_bullets(page: Page) -> int:
    """Remove list markers that the source drew as vector paths.

    Producers commonly draw a bullet as a filled circle rather than setting it
    as text. Such a marker cannot travel with its item: it is not part of any
    text block, so it keeps the source's coordinates while the translated text
    reflows and remirrors around it - which is how bullets end up sitting
    between rows, or on top of the first letter of their own item.

    They are dropped rather than repositioned, because `attach_bullets` gives
    every list item its own text marker; keeping these as well would double
    them. A blob is only removed when it is beside a line of text, so a small
    filled shape that is part of a logo or chart is left alone.
    """
    if not page.drawings:
        return 0

    rows = [b.bbox for b in page.blocks if b.bbox and b.text.strip()]
    if not rows:
        return 0

    removed = 0
    kept = []
    for drawing in page.drawings:
        if _looks_like_vector_bullet(drawing) and _beside_text(drawing.bbox, rows):
            removed += 1
            continue
        kept.append(drawing)
    page.drawings = kept
    return removed


def _beside_text(box: BBox, rows: list[BBox]) -> bool:
    """True when `box` sits at the edge of a row of text.

    A marker may fall just outside its text block, or - when one block holds a
    whole list - inside the block's bounds at its leading edge. Both count; a
    blob in the middle of a block does not, because that is far more likely to
    be part of a diagram than a bullet.
    """
    centre = (box.y0 + box.y1) / 2
    for row in rows:
        if not (row.y0 <= centre <= row.y1):
            continue
        # Immediately outside the text, on either side.
        if 0 <= row.x0 - box.x1 <= VECTOR_BULLET_GAP:
            return True
        if 0 <= box.x0 - row.x1 <= VECTOR_BULLET_GAP:
            return True
        # Inside the block, hugging its left or right edge - the layout a list
        # gets when every item is extracted as one block. The comparison is
        # given a point of slack because the marker frequently *defines* the
        # block's edge, making the two coordinates equal to within rounding.
        if row.x0 - 1.0 <= box.x0 and box.x1 <= row.x1 + 1.0:
            if (box.x0 - row.x0) <= VECTOR_BULLET_GAP:
                return True
            if (row.x1 - box.x1) <= VECTOR_BULLET_GAP:
                return True
    return False


def attach_bullets(page: Page) -> int:
    """Normalise every list marker on the page.

    A source marker is *discarded* and a clean one written in its place, rather
    than being repositioned. Producers emit markers at arbitrary sizes,
    baselines and offsets - a 5pt glyph raised above the line, or one drawn so
    close to its text that extraction reads them as a single run
    ("\u00b7Free training materials") - and repositioning cannot make a set of
    those agree. Re-emitting one consistent marker as part of the text run
    makes them wrap, shrink and align with the item they belong to.

    Returns the number of items that were given a normalised marker.
    """
    joined = 0

    # 1. Marker on its own line inside an otherwise normal block.
    for block in page.blocks:
        if len(block.lines) < 2:
            continue
        kept: list[Line] = []
        pending: Optional[Line] = None
        for line in block.lines:
            if _is_marker_line(line) and pending is None:
                pending = line
                continue
            if pending is not None:
                _prefix_line(line, pending)
                joined += 1
                pending = None
            kept.append(line)
        if pending is not None:      # trailing marker with nothing to attach to
            kept.append(pending)
        block.lines = kept

    # 2. Marker as a block of its own, beside the block holding the item.
    markers = [b for b in page.blocks
               if b.bbox and len(b.lines) == 1 and _is_marker_line(b.lines[0])]
    for marker in markers:
        target = _bullet_target(marker, page)
        if target is None:
            continue
        _prefix_line(target.lines[0], marker.lines[0])
        if marker.bbox and target.bbox:
            target.bbox = target.bbox.union(marker.bbox)
        marker.lines = []
        joined += 1
    page.blocks = [b for b in page.blocks if b.lines]

    # 3. Whatever the marker's origin - its own line, its own block, or glued
    #    to the front of the text - the item now starts with one. Replace it
    #    with the canonical marker so every item on the page matches.
    for block in page.blocks:
        for line in block.lines:
            if not line.spans or not _renormalise_line(line):
                continue
            joined += 1
    return joined


def _renormalise_line(line: Line) -> bool:
    """Rewrite a line's leading marker as the canonical one. True if changed.

    The marker may be the head of the first span ("\u00b7Free training
    materials") or a span of its own - producers often set it at its own size
    and baseline, which is exactly what makes it land badly. Either way the
    source marker is dropped and the canonical one is written onto the span
    that carries the item's text, so marker and text share one style.
    """
    body = [i for i, sp in enumerate(line.spans) if sp.text.strip()]
    if not body:
        return False

    first = line.spans[body[0]]
    marker, stripped, had_marker = strip_leading_marker(first.text)
    if had_marker:
        wanted = f"{marker}{BULLET_SEPARATOR}{stripped.lstrip()}"
        if first.text == wanted:
            return False        # already normalised - nothing to do
        first.text = wanted
        return True

    # Marker as a span of its own, with the item's text in a later span.
    if len(body) < 2 or not is_bullet_text(first.text):
        return False
    target = line.spans[body[1]]
    marker = canonical_marker(first.text)
    target.text = f"{marker}{BULLET_SEPARATOR}{target.text.lstrip()}"
    # Dropping the marker span takes its odd size and baseline with it.
    line.spans = [sp for i, sp in enumerate(line.spans) if i != body[0]]
    return True


def _bullet_target(marker: TextBlock, page: Page) -> Optional[TextBlock]:
    """The text block a standalone marker introduces, if there is one."""
    if marker.bbox is None:
        return None
    best: Optional[TextBlock] = None
    best_gap = BULLET_MAX_GAP
    for block in page.blocks:
        if block is marker or not block.bbox or not block.lines:
            continue
        if is_bullet_text(block.text):
            continue
        # Same row: the marker's vertical centre falls inside the text's band.
        centre = (marker.bbox.y0 + marker.bbox.y1) / 2
        tolerance = block.bbox.height * BULLET_ROW_TOLERANCE
        if not (block.bbox.y0 - tolerance <= centre <= block.bbox.y1 + tolerance):
            continue
        # The marker introduces text that starts after it, on either side -
        # a right-to-left list puts the marker to the right of its item.
        gap = min(abs(block.bbox.x0 - marker.bbox.x1),
                  abs(marker.bbox.x0 - block.bbox.x1))
        if gap < best_gap:
            best, best_gap = block, gap
    return best


def _prefix_line(line: Line, marker: Line) -> None:
    """Put `marker` at the head of `line`, separated by one space.

    A marker with no glyph of its own - a bare tab, left behind when the
    visible bullet was a vector path - contributes no spans: the canonical
    marker is written onto the item's own span instead, so the two share one
    style and one baseline.
    """
    spans = [sp for sp in marker.spans if sp.text.strip()]
    if not spans:
        # Glyphless marker: write the canonical bullet onto the item's own
        # first span, so marker and text share a size, colour and baseline.
        target = next((sp for sp in line.spans if sp.text.strip()), None)
        if target is not None:
            target.text = (f"{BULLET_MARKER}{BULLET_SEPARATOR}"
                           f"{target.text.lstrip()}")
        return
    if not spans[-1].text.endswith(" "):
        spans[-1].text = spans[-1].text + " "
    line.spans = spans + line.spans
    if marker.bbox and line.bbox:
        line.bbox = line.bbox.union(marker.bbox)


def _combine(blocks: list[TextBlock], rtl: bool) -> TextBlock:
    """Join blocks into one, ordered by reading direction."""
    ordered = sorted(blocks, key=lambda b: -b.bbox.x1 if rtl else b.bbox.x0)

    merged = TextBlock(block_no=min(b.block_no for b in blocks))
    box = ordered[0].bbox
    for block in ordered[1:]:
        box = box.union(block.bbox)
    merged.bbox = box

    # One line holding every span, in reading order, separated by spaces.
    spans = []
    for i, block in enumerate(ordered):
        block_spans = [s for s in block.spans if s.text]
        if not block_spans:
            continue
        # Exactly one space between joined fragments: the next span often
        # carries its own leading space, and two spaces render as a visible gap.
        if i > 0 and spans and not spans[-1].text.endswith(" ") \
                and not block_spans[0].text.startswith(" "):
            spans[-1].text = spans[-1].text + " "
        spans.extend(block_spans)
    merged.lines = [Line(spans=spans, bbox=box)]
    # Recomputed from the joined text, never inherited: merging changes what the
    # block says, so the script decision has to be made again on the result.
    # Without this a merged block silently falls back to the default and an
    # English footer built from several fragments gets mirrored.
    merged.mirror = block_is_rtl(merged)
    return merged


# Consecutive lines belong to one paragraph when they share a left/right edge,
# use the same size, and sit within this multiple of their own height.
PARAGRAPH_LINE_GAP = 1.75
EDGE_TOLERANCE = 12.0
# Share of the narrower line that must sit within the other's span for two
# ragged fragments to count as lines of one paragraph.
COLUMN_OVERLAP_SHARE = 0.7
SIZE_TOLERANCE = 0.6


def _same_paragraph(a: TextBlock, b: TextBlock) -> bool:
    """Whether `b` is the next line of the paragraph `a` belongs to."""
    if a.bbox is None or b.bbox is None:
        return False

    style_a, style_b = a.dominant_style(), b.dominant_style()
    if abs(style_a.size - style_b.size) > SIZE_TOLERANCE:
        return False
    if style_a.bold != style_b.bold:
        return False

    # Measure the line pitch against the font size, not the block height: a
    # block's height already includes ascenders and descenders, which makes the
    # threshold loose enough to swallow the next paragraph.
    gap = b.bbox.y0 - a.bbox.y0
    if gap <= 0 or gap > style_a.size * PARAGRAPH_LINE_GAP:
        return False

    # Lines of a paragraph share an edge: the left one for LTR text, the right
    # one for RTL. Accept either, so the same test works for both scripts.
    shares_left = abs(a.bbox.x0 - b.bbox.x0) <= EDGE_TOLERANCE
    shares_right = abs(a.bbox.x1 - b.bbox.x1) <= EDGE_TOLERANCE
    if shares_left or shares_right:
        return True

    # A line can share neither edge and still belong to the paragraph: where a
    # gloss or a differently-sized number is carved out of the middle of a
    # line, the remainder is a ragged fragment whose box lines up with nothing.
    # Such a fragment overlaps the *assembled* paragraph even when it barely
    # touches the single line beside it, so it is caught by the second pass in
    # `merge_paragraph_lines` rather than here.
    overlap = min(a.bbox.x1, b.bbox.x1) - max(a.bbox.x0, b.bbox.x0)
    narrower = min(a.bbox.width, b.bbox.width)
    return narrower > 0 and overlap >= narrower * COLUMN_OVERLAP_SHARE


def merge_paragraph_lines(page: Page, direction: str = "") -> int:
    """Join consecutive lines of one paragraph into a single block.

    Designed PDFs often emit every visual line as its own block. Laid out
    independently, each line is fitted and shrunk on its own, so one paragraph
    ends up rendered at several different sizes and indents. Merging them first
    means the paragraph is measured, wrapped and drawn as one unit.
    """
    blocks = sorted([b for b in page.blocks if b.text.strip()],
                    key=lambda b: (b.bbox.y0, b.bbox.x0))
    # Every line of the page is a candidate, whatever language it is in.
    # Grouping lines into the paragraph they belong to is a *layout* decision:
    # a paragraph laid out line by line is fitted and indented line by line
    # whether or not its words are about to change. Excluding target-language
    # text here - which the language guard used to do - meant an Arabic page
    # translated en2ar had none of its paragraphs assembled at all, so every
    # line kept its own ragged box.
    eligible = blocks

    merged: list[TextBlock] = []
    used: set[int] = set()
    count = 0

    for block in eligible:
        if id(block) in used:
            continue
        run = [block]
        used.add(id(block))
        while True:
            nxt = next((b for b in eligible
                        if id(b) not in used and _same_paragraph(run[-1], b)), None)
            if nxt is None:
                break
            run.append(nxt)
            used.add(id(nxt))
        if len(run) > 1:
            merged.append(_combine_lines(run))
            count += 1
        else:
            merged.append(block)

    # A line processed before the run it belongs to forms a single-line run of
    # its own and closes, so the backward test above never sees it. Sweep once
    # more over what is left, joining any single line that lines up with a
    # paragraph already assembled.
    changed = True
    while changed:
        changed = False
        for i, single in enumerate(merged):
            if single is None or len(single.lines) != 1:
                continue
            for j, run in enumerate(merged):
                if i == j or run is None or len(run.lines) < 2:
                    continue
                if _same_paragraph(single, run) or _same_paragraph(run, single):
                    pair = sorted((single, run), key=lambda b: b.bbox.y0)
                    merged[j] = _combine_lines(list(pair))
                    merged[i] = None
                    count += 1
                    changed = True
                    break
            if changed:
                break
    merged = [b for b in merged if b is not None]

    untouched = [b for b in page.blocks if not any(b is e for e in eligible)]
    page.blocks = merged + untouched
    page.blocks.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
    return count


def _combine_lines(blocks: list[TextBlock]) -> TextBlock:
    """Stack blocks into one, keeping each source line as its own line."""
    merged = TextBlock(block_no=min(b.block_no for b in blocks))
    box = blocks[0].bbox
    for block in blocks[1:]:
        box = box.union(block.bbox)
    merged.bbox = box
    for block in blocks:
        merged.lines.extend(block.lines)
    merged.mirror = block_is_rtl(merged)
    return merged


# A fragment is absorbed into a paragraph when its own box sits inside the
# paragraph's, with this much slack for ascenders and descenders.
INLINE_MARGIN = 6.0
# Only a short run is inlined; a long block beside a paragraph is its own text.
INLINE_MAX_WORDS = 12


def inline_embedded_fragments(page: Page, direction: str = "") -> int:
    """Fold a fragment that sits inside a paragraph into that paragraph's text.

    A parenthetical gloss - "(Mayor's Office of Immigrant Affairs, MOIA)" - or
    a number lifted out to its own block is *part of the sentence around it*,
    but extraction gives it a box of its own. Kept as a box it has to be placed
    by geometry, and after translation the surrounding lines no longer have a
    hole the right size for it: it lands on top of them.

    Merging it into the sentence removes the placement problem rather than
    solving it. The fragment becomes a run inside the paragraph, so it wraps
    with the text, takes the paragraph's alignment, and cannot collide with the
    lines it belongs to. Where it *lands* is then decided by the reflow, which
    is what "put it where it should be" means.
    """
    inlined = 0
    hosts = [b for b in page.blocks if b.bbox and len(b.lines) >= 1]
    absorbed: list[TextBlock] = []

    for fragment in page.blocks:
        if fragment.bbox is None or not fragment.text.strip():
            continue
        if len(fragment.text.split()) > INLINE_MAX_WORDS:
            continue
        host = _inline_host(fragment, hosts, direction)
        if host is None:
            continue
        _insert_inline(host, fragment)
        absorbed.append(fragment)
        inlined += 1

    if absorbed:
        page.blocks = [b for b in page.blocks
                       if not any(b is gone for gone in absorbed)]
    return inlined


def _inline_host(fragment: TextBlock, hosts: list[TextBlock],
                 direction: str) -> Optional[TextBlock]:
    """The paragraph `fragment` sits inside, if any."""
    box = fragment.bbox
    best: Optional[TextBlock] = None
    best_area = 0.0
    for host in hosts:
        if host is fragment or host.bbox is None:
            continue
        if len(host.lines) < 2:
            continue        # a single line is not a paragraph to fold into
        if not host.text.strip():
            continue
        # Wholly inside the paragraph's box, allowing for line overhang.
        if not (host.bbox.x0 - INLINE_MARGIN <= box.x0
                and box.x1 <= host.bbox.x1 + INLINE_MARGIN
                and host.bbox.y0 - INLINE_MARGIN <= box.y0
                and box.y1 <= host.bbox.y1 + INLINE_MARGIN):
            continue
        area = host.bbox.width * host.bbox.height
        if best is None or area < best_area:
            best, best_area = host, area
    return best


def _insert_inline(host: TextBlock, fragment: TextBlock) -> None:
    """Put `fragment`'s spans into `host` at the reading position they occupy.

    The fragment goes after the line it sits below, so the sentence still reads
    in its original order. Its own box is discarded - that is the point: from
    here it is text, and the paragraph's layout decides where it appears.
    """
    box = fragment.bbox
    centre = (box.y0 + box.y1) / 2

    # The line the fragment belongs to: the last one starting above its centre.
    index = 0
    for i, line in enumerate(host.lines):
        if line.bbox is None:
            continue
        if line.bbox.y0 <= centre:
            index = i
    target = host.lines[index]

    spans = [sp for sp in fragment.spans if sp.text.strip()]
    if not spans:
        return
    # One space between the fragment and the words around it.
    if target.spans and not target.spans[-1].text.endswith(" "):
        target.spans[-1].text = target.spans[-1].text + " "
    if not spans[-1].text.endswith(" "):
        spans[-1].text = spans[-1].text + " "
    target.spans.extend(spans)
    if target.bbox is not None:
        target.bbox = target.bbox.union(box)
    if host.bbox is not None:
        host.bbox = host.bbox.union(box)


def merge_fragments(page: Page, rtl: bool, direction: str = "") -> int:
    """Merge same-row fragments on `page` in place. Returns how many merges ran.

    `rtl` selects the reading direction of the *source* document. When
    `direction` is given, blocks already written in the target language are left
    out of merging entirely - joining them to a source-language fragment would
    send text through the translator that was meant to be preserved.
    """
    candidates = [b for b in page.blocks if _mergeable(b)]
    if direction:
        from .language import is_already_target

        # A preserved block is neither merged into another nor used as a target:
        # joining it to a source-language fragment would send text through the
        # translator that was meant to be left exactly as written.
        candidates = [b for b in candidates
                      if not is_already_target(b.text, direction)]
    if len(candidates) < 2:
        return 0

    used: set[int] = set()
    merged_blocks: list[TextBlock] = []
    merges = 0

    # Work top-down so a run is built in a stable order.
    for i, block in enumerate(sorted(candidates, key=lambda b: b.bbox.y0)):
        if id(block) in used:
            continue
        run = [block]
        used.add(id(block))
        changed = True
        while changed:
            changed = False
            for other in candidates:
                if id(other) in used:
                    continue
                if any(_same_row(member, other) and _gap(member, other) <= MAX_GAP
                       for member in run):
                    run.append(other)
                    used.add(id(other))
                    changed = True
        if len(run) > 1:
            merged_blocks.append(_combine(run, rtl))
            merges += 1
        else:
            merged_blocks.append(block)

    # Anything not eligible for merging passes through untouched.
    untouched = [b for b in page.blocks
                 if not any(b is c for c in candidates)]
    page.blocks = merged_blocks + untouched
    page.blocks.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
    return merges

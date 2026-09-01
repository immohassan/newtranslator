"""Horizontal layout mirroring for LTR <-> RTL conversion.

For a page of width W every element's box is reflected across the page's
vertical centre line:

    new_x0 = W - x1
    new_x1 = W - x0

y is untouched - reading order top-to-bottom does not change between English
and Arabic. The transform is its own inverse, so ar2en uses the same maths.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from .language import is_arabic, is_already_target
from .models import BBox, DrawingElement, Page, TextBlock
from .qa import QAReport


class MirrorMode(str, Enum):
    FULL = "full"      # reflect every element's position across the page
    ALIGN = "align"    # keep positions, only re-align text within its own box
    NONE = "none"      # leave layout untouched


def block_is_rtl(block: TextBlock) -> bool:
    """Whether a block belongs to the right-to-left flow.

    Decided from the block's whole text rather than span-by-span, so a single
    English word inside an Arabic paragraph cannot pull the paragraph out of the
    RTL flow. A block with no text at all keeps the page default.
    """
    text = block.text
    if not text.strip():
        return True
    return is_arabic(text)


# A label is a short standalone caption - "Email us", "Learn more" - rather
# than running prose. Mirroring one detaches it from whatever it names.
LABEL_MAX_WORDS = 5
LABEL_MAX_CHARS = 40
# Prose is recognised by sentence punctuation, which a caption does not carry.
_SENTENCE_END_RE = re.compile(r"[.!?\u061f\u06d4]\s")

# A URL or an email address is a fixed piece of contact detail printed beside
# the icon it belongs to. It is never translated and never reflows, so moving
# it only separates it from its icon - and on a crowded footer drops it onto
# whatever sits opposite.
_CONTACT_RE = re.compile(
    r"""(?xi)
    (?: [\w.+-]+ @ [\w-]+ \. [\w.-]+ )          # an email address
    | (?: (?:https?://|www\.) \S+ )              # an explicit URL
    | (?: \b[\w-]+ (?:\.[\w-]+)+ / \S* )        # host with a path
    | (?: \b[\w-]+ \. (?:gov|com|org|net|edu|io|co) \b )
    """
)


def is_contact_detail(text: str) -> bool:
    """Whether `text` is a URL or an email address.

    Such a string is printed content rather than prose: it is not translated,
    so its width does not change, and it is positioned under the icon it
    belongs to. Mirroring it can only move it away from that icon.
    """
    body = (text or "").strip()
    if not body:
        return False
    return bool(_CONTACT_RE.search(body))


def is_short_label(text: str) -> bool:
    """Whether `text` is short enough to be a caption rather than prose.

    This is only half the test - a heading is short too. `is_anchored_label`
    adds the part that actually separates them.
    """
    body = (text or "").strip()
    if not body:
        return False
    if "\n" in body:
        return False        # more than one line: a block of text, not a caption
    if len(body) > LABEL_MAX_CHARS:
        return False
    if _SENTENCE_END_RE.search(body) or body.endswith((".", "!", "?", "\u061f")):
        return False        # punctuated like a sentence
    return len(body.split()) <= LABEL_MAX_WORDS


# A caption occupies only a small part of the page's width. A heading, though
# just as short, is set across the column it heads.
LABEL_MAX_WIDTH_RATIO = 0.45
# ...and it sits beside its graphic, within this many points of it.
# Distance alone cannot separate a caption from body text that happens to run
# past a graphic - on a real page the two overlap in range - so the gap stays
# generous and `_text_between` does the discriminating.
LABEL_ICON_GAP = 60.0


def is_anchored_label(block: TextBlock, page: Page) -> bool:
    """Whether a block is a caption attached to a graphic beside it.

    Short text alone is not enough - a heading is short too, and a heading
    belongs to the flow and must mirror with it. What marks a caption is that
    it is *attached*: a narrow box sitting next to an icon, a QR code or an
    image on the same row. That attachment is the thing mirroring destroys, so
    it is the thing worth testing for.
    """
    if block.bbox is None or page.width <= 0:
        return False
    if not is_short_label(block.text):
        return False
    if block.bbox.width > page.width * LABEL_MAX_WIDTH_RATIO:
        return False        # set across a column: a heading, not a caption

    # A caption belongs to its icon, so no other text may sit between the two.
    # Body text that merely happens to run past a graphic fails this: the rest
    # of its own sentence is in the way.
    centre = (block.bbox.y0 + block.bbox.y1) / 2
    for box in _graphic_boxes(page):
        if not (box.y0 - LABEL_ICON_GAP <= centre <= box.y1 + LABEL_ICON_GAP):
            continue
        gap = min(abs(box.x0 - block.bbox.x1), abs(block.bbox.x0 - box.x1))
        if not (box.intersects(block.bbox) or gap <= LABEL_ICON_GAP):
            continue
        if _text_between(block, box, page):
            continue
        if _has_stacked_neighbour(block, page):
            continue
        return True
    return False


# Lines of one paragraph interleave: their vertical bands overlap, because a
# line's box includes ascenders and descenders that reach into the row beside
# it. A caption and the separate item below it do not overlap - there is clear
# space between them. That sign, rather than the distance, is what separates a
# line of body text from a standalone label.
STACK_MIN_OVERLAP = 1.0
STACK_MIN_SHARE = 0.5


def _has_stacked_neighbour(block: TextBlock, page: Page) -> bool:
    """Whether `block` is one line of a taller stack of text.

    A caption sits on its own beside an icon. Body text that happens to run
    close to a graphic does not: the rest of its sentence is directly above or
    below it, in the same column. That stacking is what tells the two apart on
    a real page, where the distance to the nearest graphic does not.
    """
    if block.bbox is None:
        return False
    width = max(block.bbox.width, 1.0)
    for other in page.blocks:
        if other is block or other.bbox is None or not other.text.strip():
            continue
        share = (min(block.bbox.x1, other.bbox.x1)
                 - max(block.bbox.x0, other.bbox.x0))
        if share < width * STACK_MIN_SHARE:
            continue
        overlap = (min(block.bbox.y1, other.bbox.y1)
                   - max(block.bbox.y0, other.bbox.y0))
        if overlap >= STACK_MIN_OVERLAP:
            return True
    return False


def _text_between(block: TextBlock, graphic: BBox, page: Page) -> bool:
    """Whether another text block sits between `block` and `graphic`."""
    lo = min(block.bbox.x1, graphic.x1)
    hi = max(block.bbox.x0, graphic.x0)
    if hi <= lo:
        return False
    centre = (block.bbox.y0 + block.bbox.y1) / 2
    for other in page.blocks:
        if other is block or other.bbox is None or not other.text.strip():
            continue
        if not (other.bbox.y0 <= centre <= other.bbox.y1):
            continue
        if other.bbox.x1 > lo and other.bbox.x0 < hi:
            return True
    return False


# A fragment counts as sitting on a paragraph's row when their vertical bands
# touch at all. A gloss set between two lines of the sentence it belongs to
# overlaps each of them only slightly, so demanding a deep overlap finds
# nothing.
EMBEDDED_ROW_OVERLAP = 1.0
# How far above or below the fragment a host's band may start.
EMBEDDED_ROW_REACH = 12.0
# ...and how far away horizontally the host may sit.
EMBEDDED_MAX_GAP = 60.0


def find_host_paragraph(block: TextBlock, page: Page) -> Optional[TextBlock]:
    """The paragraph a short fragment sits inside, if there is one.

    A parenthetical gloss - "(Mayor's Office of Immigrant Affairs, MOIA)" -
    reads as part of the sentence around it even though its script differs.
    Its position is meaningful only relative to that sentence, so when the
    paragraph moves it has to move too. Left where it was, it is orphaned on
    the far side of the page and lands on whatever is there.

    Only a *host* that actually mirrors is returned: a fragment beside text
    that is itself staying put has nothing to follow.
    """
    if block.bbox is None:
        return None
    best: Optional[TextBlock] = None
    best_gap = EMBEDDED_MAX_GAP
    for other in page.blocks:
        if other is block or other.bbox is None or not other.mirror:
            continue
        # A host must be a block that will actually move. One that is itself
        # pinned - an anchored caption, a contact detail - offers a fragment
        # nothing to follow, and picking it would leave the fragment stranded
        # exactly as before.
        if is_contact_detail(other.text) or is_anchored_label(other, page):
            continue
        if not other.text.strip():
            continue
        # The host must be part of the flow this fragment is embedded in, not
        # another short label sitting on the same row. Length is a poor test -
        # a producer splits one sentence across several blocks, any of which
        # may be shorter than the gloss inside it - so what matters is that the
        # candidate is ordinary running text rather than a caption of its own.
        if is_contact_detail(other.text):
            continue
        # On the same visual row: the bands touch, or come within a line of
        # each other. A gloss between two lines of one sentence sits like this.
        overlap = (min(block.bbox.y1, other.bbox.y1)
                   - max(block.bbox.y0, other.bbox.y0))
        if overlap < EMBEDDED_ROW_OVERLAP - EMBEDDED_ROW_REACH:
            continue
        # Horizontally inside it, or touching it.
        gap = max(other.bbox.x0 - block.bbox.x1, block.bbox.x0 - other.bbox.x1)
        if gap > best_gap:
            continue
        best, best_gap = other, max(gap, 0.0)
    return best


# Clear space to leave between a moved fragment and whatever it sits beside.
FRAGMENT_CLEARANCE = 6.0
# How far along the row the fragment may be nudged to find that space.
FRAGMENT_MAX_NUDGE = 220.0
FRAGMENT_NUDGE_STEP = 4.0


def _row_obstacles(block: TextBlock, page: Page, host: TextBlock) -> list[BBox]:
    """Everything on the fragment's row that it must not cover.

    Its host is excluded: a gloss belongs to that sentence and sits against it
    by design. Everything else on the row - other lines, icons - is an
    obstacle, measured where those elements have ended up.
    """
    if block.bbox is None:
        return []
    band_lo, band_hi = block.bbox.y0, block.bbox.y1
    obstacles: list[BBox] = []
    for other in page.blocks:
        if other is block or other is host or other.bbox is None:
            continue
        if not other.text.strip():
            continue
        if other.bbox.y1 > band_lo and other.bbox.y0 < band_hi:
            obstacles.append(other.bbox)
    for box in _graphic_boxes(page):
        if box.y1 > band_lo and box.y0 < band_hi:
            obstacles.append(box)
    return obstacles


def _collides(box: BBox, obstacles: list[BBox]) -> bool:
    for other in obstacles:
        if (box.x1 + FRAGMENT_CLEARANCE > other.x0
                and box.x0 - FRAGMENT_CLEARANCE < other.x1):
            return True
    return False


def _clearance_shift(block: TextBlock, dx: float, page: Page,
                     host: TextBlock) -> float:
    """Extra horizontal shift that frees `block` from its neighbours.

    Returns 0 when the landing spot is already clear, or when nothing within
    reach is - overlapping is then reported by QA rather than swapped for a
    collision somewhere else on the row.
    """
    if block.bbox is None:
        return 0.0
    landed = BBox(block.bbox.x0 + dx, block.bbox.y0,
                  block.bbox.x1 + dx, block.bbox.y1)
    obstacles = _row_obstacles(block, page, host)
    if not _collides(landed, obstacles):
        return 0.0

    step = FRAGMENT_NUDGE_STEP
    while step <= FRAGMENT_MAX_NUDGE:
        for extra in (-step, step):
            candidate = BBox(landed.x0 + extra, landed.y0,
                             landed.x1 + extra, landed.y1)
            if candidate.x0 < 2 or candidate.x1 > page.width - 2:
                continue
            if not _collides(candidate, obstacles):
                return extra
        step += FRAGMENT_NUDGE_STEP
    return 0.0




def _graphic_boxes(page: Page) -> list[BBox]:
    """Every image and grouped vector graphic on the page, as boxes."""
    boxes = [im.bbox for im in page.images]
    for group in _group_artwork(page.drawings, page.width):
        extent = group[0].bbox
        for drawing in group[1:]:
            extent = extent.union(drawing.bbox)
        boxes.append(extent)
    return boxes


def mirror_bbox(bbox: BBox, page_width: float) -> BBox:
    """Reflect a box horizontally. y stays put."""
    return BBox(
        x0=page_width - bbox.x1,
        y0=bbox.y0,
        x1=page_width - bbox.x0,
        y1=bbox.y1,
    )


def mirror_point(x: float, y: float, page_width: float) -> tuple[float, float]:
    return (page_width - x, y)


# Vertical overlap below this share of a path's height starts a new row.
ROW_OVERLAP_RATIO = 0.35
# Horizontal gap (points) that still counts as the same word or graphic.
ARTWORK_GAP = 14.0
# Paths wider than this share of the page are backgrounds, not artwork.
BACKGROUND_WIDTH_RATIO = 0.7


def _group_artwork(
    drawings: list[DrawingElement], page_width: float = 0.0
) -> list[list[DrawingElement]]:
    """Cluster paths that visually form one graphic.

    A logo arrives as one path per glyph. Grouping is done by row and then by
    horizontal proximity: paths that share a vertical band and sit close
    together belong to the same graphic. Row membership is decided by vertical
    overlap rather than by matching top edges, because a capital "I" has no
    descender and sits higher than the letters beside it.

    Each group is mirrored as a unit, so a graphic moves across the page
    without its parts being reordered.
    """
    simple = [d for d in drawings if d.kind in ("line", "rect", "underline")]
    art = [d for d in drawings if d.kind == "path"]

    # Full-width panels are page furniture: mirror them on their own.
    if page_width > 0:
        backgrounds = [d for d in art
                       if d.bbox.width > page_width * BACKGROUND_WIDTH_RATIO]
        art = [d for d in art if d not in backgrounds]
        simple.extend(backgrounds)

    groups: list[list[DrawingElement]] = [[d] for d in simple]

    # 1. Split into rows by vertical overlap.
    rows: list[list[DrawingElement]] = []
    for drawing in sorted(art, key=lambda d: (d.bbox.y0, d.bbox.x0)):
        placed = False
        for row in rows:
            top = min(d.bbox.y0 for d in row)
            bottom = max(d.bbox.y1 for d in row)
            overlap = min(drawing.bbox.y1, bottom) - max(drawing.bbox.y0, top)
            if overlap > drawing.bbox.height * ROW_OVERLAP_RATIO:
                row.append(drawing)
                placed = True
                break
        if not placed:
            rows.append([drawing])

    # 2. Within a row, split on horizontal gaps.
    for row in rows:
        row.sort(key=lambda d: d.bbox.x0)
        cluster = [row[0]]
        for drawing in row[1:]:
            reach = max(d.bbox.x1 for d in cluster)
            if drawing.bbox.x0 - reach > ARTWORK_GAP:
                groups.append(cluster)
                cluster = []
            cluster.append(drawing)
        if cluster:
            groups.append(cluster)
    return groups


def _detect_overlaps(page: Page) -> list[tuple[str, BBox, BBox]]:
    """Overlapping boxes survive mirroring but can land in a worse spot, so we
    report them rather than silently shifting content."""
    issues: list[tuple[str, BBox, BBox]] = []
    boxes: list[tuple[str, BBox]] = []
    for b in page.blocks:
        if b.bbox:
            boxes.append(("text", b.bbox))
    for im in page.images:
        boxes.append(("image", im.bbox))

    for i in range(len(boxes)):
        kind_a, box_a = boxes[i]
        for j in range(i + 1, len(boxes)):
            kind_b, box_b = boxes[j]
            if not box_a.intersects(box_b):
                continue
            overlap_w = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
            overlap_h = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
            area = overlap_w * overlap_h
            smaller = min(box_a.width * box_a.height, box_b.width * box_b.height)
            if smaller > 0 and area / smaller > 0.15:
                issues.append((f"{kind_a}/{kind_b}", box_a, box_b))
    return issues


def _nudge_clear(block, clashes: list[BBox], page: Page) -> bool:
    """Shift `block` horizontally until it no longer covers a preserved box.

    Returns False when there is no free space, so the caller can report the
    clash rather than pretend it was resolved.
    """
    for kept in sorted(clashes, key=lambda b: b.x0):
        if block.bbox is None:
            return False
        # Try the gap on each side of the preserved box, nearest first.
        options = [kept.x0 - block.bbox.width - 4, kept.x1 + 4]
        options.sort(key=lambda x: abs(x - block.bbox.x0))
        for new_x0 in options:
            if new_x0 < 2 or new_x0 + block.bbox.width > page.width - 2:
                continue
            shift = new_x0 - block.bbox.x0
            candidate = BBox(block.bbox.x0 + shift, block.bbox.y0,
                             block.bbox.x1 + shift, block.bbox.y1)
            if any(candidate.intersects(other) for other in clashes):
                continue
            block.bbox = candidate
            for line in block.lines:
                if line.bbox:
                    line.bbox = BBox(line.bbox.x0 + shift, line.bbox.y0,
                                     line.bbox.x1 + shift, line.bbox.y1)
                for span in line.spans:
                    span.bbox = BBox(span.bbox.x0 + shift, span.bbox.y0,
                                     span.bbox.x1 + shift, span.bbox.y1)
                    ox, oy = span.origin
                    span.origin = (ox + shift, oy)
            return True
    return False


# An element is "side" content when it sits clear of the page's middle band -
# a margin icon, a pull-quote graphic, a logo in a top corner. Those carry
# left/right meaning and must move when the page flips. Anything straddling the
# centre, or sitting in the footer strip, is centre-anchored furniture: a footer
# icon row, a centred logo, a divider. Flipping those only scrambles a layout
# that read correctly to begin with.
CENTRE_BAND = 0.25      # share of the page width counted as "the middle"
FOOTER_BAND = 0.12      # bottom share of the page treated as footer
HEADER_BAND = 0.06      # very top, where centred mastheads sit


def _spans_centre(bbox: BBox, page_width: float) -> bool:
    """True when the box is genuinely centred on the page.

    Merely clipping the central band is not enough: a left-margin rule can
    reach past the page's middle and is still left-side content. The test is
    whether the element's own centre sits in the band, or whether it straddles
    the middle roughly evenly - which is what a centred logo or a full-width
    divider does.
    """
    lo = page_width * (0.5 - CENTRE_BAND / 2)
    hi = page_width * (0.5 + CENTRE_BAND / 2)
    centre_x = (bbox.x0 + bbox.x1) / 2
    if lo <= centre_x <= hi:
        return True
    # Straddles the middle with comparable weight on both sides.
    middle = page_width / 2
    left = max(middle - bbox.x0, 0.0)
    right = max(bbox.x1 - middle, 0.0)
    if left <= 0 or right <= 0:
        return False
    return min(left, right) / max(left, right) > 0.6


def should_mirror_element(bbox: BBox, page_width: float,
                          page_height: float) -> bool:
    """Whether a positioned graphic should move when the page is mirrored.

    Only elements that are clearly on one side of the page carry a left/right
    relationship worth preserving. An icon row in the footer, or a graphic
    centred on the page, is positioned relative to the page as a whole rather
    than to the reading direction, so mirroring it moves it away from the text
    it belongs with - the footer icons in the reported output being exactly
    that case.
    """
    if page_width <= 0 or page_height <= 0:
        return True
    if _spans_centre(bbox, page_width):
        return False
    centre_y = (bbox.y0 + bbox.y1) / 2
    if centre_y >= page_height * (1.0 - FOOTER_BAND):
        return False    # footer strip
    if centre_y <= page_height * HEADER_BAND:
        return False    # masthead strip
    return True


def _shift_span(span, dx: float) -> None:
    """Move a span horizontally without reflecting it."""
    span.bbox = BBox(span.bbox.x0 + dx, span.bbox.y0,
                     span.bbox.x1 + dx, span.bbox.y1)
    ox, oy = span.origin
    span.origin = (ox + dx, oy)


def _reanchor_ltr_block(block, page_width: float) -> None:
    """Move a left-to-right block to the position the mirrored layout gives it.

    The block's *box* is reflected, so it stays attached to the content it
    belongs to on a page that now reads right-to-left. Its contents are then
    translated by that single shift rather than being reflected individually,
    which keeps every span in left-to-right order inside the box.
    """
    if block.bbox is None:
        return
    mirrored = mirror_bbox(block.bbox, page_width)
    dx = mirrored.x0 - block.bbox.x0
    block.bbox = mirrored
    for line in block.lines:
        if line.bbox:
            line.bbox = BBox(line.bbox.x0 + dx, line.bbox.y0,
                             line.bbox.x1 + dx, line.bbox.y1)
        for span in line.spans:
            _shift_span(span, dx)


def mirror_page(page: Page, mode: MirrorMode, qa: QAReport,
                direction: str = "") -> None:
    """Mirror the positioned elements on `page` in place.

    Each block decides for itself, from its own dominant script (`block.mirror`,
    set at extraction and recomputed after any merge). There is no per-block
    configuration and none is needed: an English footer, heading or caption is
    detected and left unflipped by exactly the same rule that mirrors the Arabic
    body around it.

    `mode` remains a global override for whole documents - MirrorMode.ALIGN
    skips the layout pass entirely - but it does not participate in the
    block-by-block decision.
    """
    if mode is not MirrorMode.FULL:
        return

    w = page.width
    preserved_boxes: list[BBox] = []
    # Fragments that must follow a host paragraph. Every block's box is
    # recorded up front: a host may sit before or after its fragment in the
    # list, so reading the host's box when the fragment is reached would give
    # the already-mirrored value for half of them and a zero shift.
    original_boxes: dict[int, BBox] = {
        id(b): BBox(*b.bbox.as_tuple()) for b in page.blocks if b.bbox
    }
    # Hosts are resolved up front, on the untouched page. Asking mid-loop would
    # match against boxes that have already been mirrored, which picks the
    # wrong neighbour and can carry a fragment clean off the page.
    hosts: dict[int, TextBlock] = {}
    if direction:
        for candidate in page.blocks:
            if candidate.bbox is None or candidate.mirror:
                continue
            if not is_already_target(candidate.text, direction):
                continue
            found = find_host_paragraph(candidate, page)
            if found is not None:
                hosts[id(candidate)] = found
    embedded: list[tuple[TextBlock, TextBlock]] = []

    for issue_kind, box_a, box_b in _detect_overlaps(page):
        qa.mirror_issue(
            page.number + 1,
            f"overlapping {issue_kind} elements were mirrored together and may "
            f"need a visual check",
            element_a=box_a.as_tuple(),
            element_b=box_b.as_tuple(),
        )

    for block in page.blocks:
        if block.bbox is None:
            continue

        # Two separate questions, previously conflated into one:
        #
        #   * is this block part of the right-to-left flow?  -> block.mirror,
        #     a property of the block's own script, so an English heading in an
        #     Arabic document is never flipped whichever way we translate;
        #   * was it already in the target language?         -> is_already_target,
        #     a property of the job, which additionally freezes its content.
        #
        # The language guard freezes a block only when leaving it alone is also
        # the right *layout* answer - that is, when it is not part of the page's
        # right-to-left flow. An Arabic block on an en2ar job is "already
        # target" (its content is not re-translated) but it is still RTL text on
        # a page being mirrored, so its position must follow the flow. Letting
        # the content rule drive the layout is what flipped English headings on
        # Arabic-source documents while pinning the Arabic in place.
        if direction and is_already_target(block.text, direction) \
                and not block.mirror:
            # ...unless it is embedded in a paragraph that *is* mirroring. A
            # parenthetical gloss inside an Arabic sentence reads as part of
            # that sentence; freezing it while the sentence moves strands it on
            # the far side of the page. It travels with its host and is
            # re-anchored there, keeping its own left-to-right run.
            host = hosts.get(id(block))
            if host is None:
                preserved_boxes.append(block.bbox)
                continue
            embedded.append((block, host))
            continue

        # A short standalone caption is anchored to the thing it names - an
        # icon, a QR code - rather than to the reading direction, so it keeps
        # its position and is only translated. Reflecting it detaches it from
        # its icon and, in a full footer, drops it onto whatever is opposite.
        # A URL or email address is printed contact detail: not translated, so
        # it never reflows, and positioned under the icon it belongs to. It
        # holds its place whether or not a graphic sits close enough to make it
        # an anchored label.
        if is_contact_detail(block.text) or is_anchored_label(block, page):
            preserved_boxes.append(block.bbox)
            continue

        if not block.mirror:
            # Latin text keeps its left-to-right run, but its box still has to
            # follow the page: a heading that labelled the block now on the
            # other side would otherwise float away from what it names. The box
            # is reflected as a whole to find its new anchor, then the text is
            # laid back out left-to-right inside it - so the heading moves with
            # its section without being flipped within itself.
            _reanchor_ltr_block(block, w)
            # Deliberately *not* added to preserved_boxes. That list means
            # "this did not move", and it is what stops artwork being
            # mirrored out from under content that stayed put. A re-anchored
            # block has moved with the page, so freezing artwork against its
            # new box pins icons that should have travelled - which is what
            # left the body icons stranded on the wrong side, overlapping the
            # text, on an Arabic page translated en2ar.
            continue

        block.bbox = mirror_bbox(block.bbox, w)
        for line in block.lines:
            if line.bbox:
                line.bbox = mirror_bbox(line.bbox, w)
            for span in line.spans:
                if not span.mirror:
                    # An LTR run inside an RTL block (a brand name, a URL): the
                    # block around it moves, so the span moves with it, but the
                    # span is not reflected about its own centre.
                    _shift_span(span, mirror_bbox(span.bbox, w).x0 - span.bbox.x0)
                    continue
                span.bbox = mirror_bbox(span.bbox, w)
                ox, oy = span.origin
                span.origin = (w - ox, oy)

    # Now that every host has moved, carry each embedded fragment by the same
    # shift. The fragment is translated, never reflected: it keeps its own
    # reading direction inside the paragraph that moved around it.
    for block, host in embedded:
        if block.bbox is None or host.bbox is None:
            continue
        host_before = original_boxes.get(id(host))
        if host_before is None:
            continue
        dx = host.bbox.x0 - host_before.x0
        # Following the host puts the fragment in the right region, but the
        # text there has reflowed to a different width and the graphics have
        # settled, so the landing spot may now be occupied. Nudge along the row
        # until it is clear.
        dx += _clearance_shift(block, dx, page, host)
        block.bbox = BBox(block.bbox.x0 + dx, block.bbox.y0,
                          block.bbox.x1 + dx, block.bbox.y1)
        for line in block.lines:
            if line.bbox:
                line.bbox = BBox(line.bbox.x0 + dx, line.bbox.y0,
                                 line.bbox.x1 + dx, line.bbox.y1)
            for span in line.spans:
                _shift_span(span, dx)

        # A fragment whose row has no space left is *not* moved further. On a
        # page of continuous text there is no clear slot within reach - a drop
        # or a longer nudge only lands it on a different neighbour, and moving
        # it away from its own sentence makes it harder to read, not easier.
        # It stays with its host and the clash is reported for a human to
        # resolve, which is the honest outcome when the page is genuinely full.
        if _collides(block.bbox, _row_obstacles(block, page, host)):
            qa.mirror_issue(
                page.number + 1,
                "a foreign-language fragment inside a paragraph could not be "
                "placed clear of the translated text around it; the row has no "
                "free space and it needs a manual layout fix",
                element=block.bbox.as_tuple(),
                excerpt=block.text[:80],
            )
        preserved_boxes.append(block.bbox)

    def overlaps_preserved(box: BBox) -> bool:
        return any(box.intersects(kept) for kept in preserved_boxes)

    for image in page.images:
        # Side graphics move with the layout; footer and centred ones hold
        # their position, because theirs is not a left/right relationship.
        if should_mirror_element(image.bbox, w, page.height):
            image.bbox = mirror_bbox(image.bbox, w)

    # A preserved block holds its ground, so a mirrored block that lands on top
    # of one is nudged clear into whatever space is free beside it. Overprinting
    # two pieces of text is never acceptable; if nothing can be freed, the clash
    # is reported instead of being hidden.
    for block in page.blocks:
        if block.bbox is None:
            continue
        if any(block.bbox is kept for kept in preserved_boxes):
            continue
        clashes = [kept for kept in preserved_boxes if block.bbox.intersects(kept)]
        if not clashes:
            continue

        moved = _nudge_clear(block, clashes, page)
        if not moved:
            qa.mirror_issue(
                page.number + 1,
                "translated text was mirrored onto content that was preserved "
                "in place; the two overlap and need a visual check",
                element=block.bbox.as_tuple(),
                excerpt=block.text[:80],
            )

    held_artwork: list[BBox] = []
    for group in _group_artwork(page.drawings, page.width):
        if preserved_boxes and any(overlaps_preserved(d.bbox) for d in group):
            continue  # artwork sitting on preserved content stays with it

        # The whole graphic is judged as a unit by where it sits on the page, so
        # a row of footer icons is kept together and kept in place rather than
        # each icon being decided - and moved - on its own.
        extent = group[0].bbox
        for drawing in group[1:]:
            extent = extent.union(drawing.bbox)
        if not should_mirror_element(extent, w, page.height):
            # A graphic that holds its position becomes an obstacle: text
            # mirrored into its space would overprint it, so it is registered
            # alongside the preserved text blocks and reported below.
            held_artwork.append(extent)
            continue

        if len(group) == 1:
            drawing = group[0]
            mirrored = mirror_bbox(drawing.bbox, w)
            drawing.shift_x += mirrored.x0 - drawing.bbox.x0
            drawing.bbox = mirrored
            continue
        mirrored = mirror_bbox(extent, w)
        shift = mirrored.x0 - extent.x0   # one shift for the whole group
        for drawing in group:
            drawing.shift_x += shift
            drawing.bbox = BBox(
                drawing.bbox.x0 + shift, drawing.bbox.y0,
                drawing.bbox.x1 + shift, drawing.bbox.y1,
            )

    # Text mirrored on top of a graphic that deliberately held its position -
    # a footer icon row, a QR code - is reported. The graphic is anchored to
    # the page rather than to the reading direction, so moving it would undo
    # the placement rule; moving the text risks a worse collision on a full
    # page. The user is told which block needs a look.
    for block in page.blocks:
        if block.bbox is None:
            continue
        for art in held_artwork:
            if not block.bbox.intersects(art):
                continue
            overlap_w = min(block.bbox.x1, art.x1) - max(block.bbox.x0, art.x0)
            overlap_h = min(block.bbox.y1, art.y1) - max(block.bbox.y0, art.y0)
            smaller = min(block.bbox.width * block.bbox.height,
                          max(art.width * art.height, 0.01))
            if smaller <= 0 or (overlap_w * overlap_h) / smaller <= 0.15:
                continue
            qa.mirror_issue(
                page.number + 1,
                "translated text was mirrored onto a graphic that stays in "
                "place (a footer or centred icon); the two overlap and need a "
                "visual check",
                element=block.bbox.as_tuple(),
                graphic=art.as_tuple(),
                excerpt=block.text[:80],
            )
            break

    # A graphic that lands on preserved content is only reported, not moved.
    # Shifting it sideways or back to its source position both push it into
    # other content - the page is full - and a small overlap with a caption is
    # far less damaging than a logo dropped onto a headline.
    if preserved_boxes:
        for group in _group_artwork(page.drawings, page.width):
            if not any(overlaps_preserved(d.bbox) for d in group):
                continue
            extent = group[0].bbox
            for drawing in group[1:]:
                extent = extent.union(drawing.bbox)
            qa.mirror_issue(
                page.number + 1,
                "a graphic overlaps text that was preserved in place; check "
                "whether the graphic needs moving by hand",
                element=extent.as_tuple(),
            )


def mirror_document_pages(pages: list[Page], mode: MirrorMode, qa: QAReport,
                          direction: str = "") -> None:
    for page in pages:
        mirror_page(page, mode, qa, direction)


def flip_image_bytes(data: bytes) -> bytes:
    """Horizontally flip image content - for arrows and other graphics whose
    meaning depends on reading direction. Returns the original bytes on failure.
    """
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format or "PNG"
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            out = io.BytesIO()
            if fmt.upper() in ("JPEG", "JPG") and flipped.mode in ("RGBA", "P", "LA"):
                flipped = flipped.convert("RGB")
            flipped.save(out, format=fmt)
            return out.getvalue()
    except Exception:
        return data


def resolve_mode(mirror_enabled: bool) -> MirrorMode:
    return MirrorMode.FULL if mirror_enabled else MirrorMode.ALIGN

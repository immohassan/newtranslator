"""Carry a page's vector artwork through the reflowing rebuild.

The HTML pipeline reads a page's structure and lets a browser lay it out
again. That is what stops a grown translation from overlapping, and it is
also why the artwork used to vanish: the drawings live at fixed coordinates,
and once the text they belonged to has moved, those coordinates are wrong.
Emitting them anyway would scatter boxes and dots across unrelated text, so
`_section_rules` kept the wide rules and dropped everything else.

Dropping it is not the only option. A drawing is not decoration at a
position - it is decoration *of something*, and once that relationship is
recovered the drawing can travel with what it belongs to:

    a rating row     five dots beside a skill      -> emitted with that skill
    a panel          a full-height sidebar tint    -> painted behind a column
    a frame          a box around a heading        -> a border on that heading
    a rule           a line across a column        -> the <hr> it already was

Anything left over is genuine page furniture with no owner: it is drawn on a
fixed layer at its own coordinates, which is right for a logo or a watermark
and harmless for anything else.

What this cannot do is keep the page looking identical. Arabic sets longer
than English, so the content below a box moves down whether or not the box
travels with it. The promise here is that every element survives and stays
attached to its own content - not that the page is a pixel copy, which no
reflowing translation can be.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import BBox, DrawingElement, Page

# --- rating rows ----------------------------------------------------------
# A rating pip is a small mark. Templates draw it as a dot, a square, or a
# short dash, so the shape test has to admit an 18x4 dash without admitting a
# rule: it is the *size* that says "pip", and the row it sits in that says
# "rating".
PIP_MAX_SIZE = 24.0
PIP_MAX_THICKNESS = 9.0
PIP_SQUARENESS = 0.15
# Pips of one row share a baseline to within this, and sit no further apart
# than this. Both are generous multiples of a 5pt pip on a 10pt pitch.
PIP_ROW_TOLERANCE = 2.5
PIP_MAX_GAP = 30.0
# Fewer than this in a row is a bullet or a stray mark, not a rating.
PIP_MIN_PER_ROW = 3
# A pip belongs to the text within this distance above it.
PIP_MAX_TEXT_GAP = 26.0

# --- panels ---------------------------------------------------------------
# A filled rect covering this much of the page's height is a panel behind a
# column rather than a box around a paragraph.
PANEL_MIN_HEIGHT_SHARE = 0.55
# ...and it must be this narrow to be a column's panel and not the page's own
# background, which is better left to the renderer.
PANEL_MAX_WIDTH_SHARE = 0.55

# --- banners --------------------------------------------------------------
# A filled band spanning this share of the page's width is a banner - the
# coloured header a CV puts its name in. It is a panel turned on its side:
# wide and short where a sidebar is narrow and tall, and it belongs to the
# content it sits behind rather than to a column.
BANNER_MIN_WIDTH_SHARE = 0.9
# ...reaching no further down the page than this, or it is a background.
BANNER_MAX_HEIGHT_SHARE = 0.35
# A band this thin is a rule the page draws across itself, not a banner with
# anything in it - but it is still the page's own decoration and is kept.
BANNER_MIN_HEIGHT = 8.0

# --- frames ---------------------------------------------------------------
# A frame is a box drawn around text: it must enclose some, and be no larger
# than this share of the page, so a full-page border is not read as one.
FRAME_MAX_HEIGHT_SHARE = 0.4
# A stroke this much longer than it is thick is one side of a frame.
FRAME_SIDE_RATIO = 8.0
# Four sides of one frame agree on their corners to within this.
FRAME_CORNER_TOLERANCE = 3.0


@dataclass
class Rating:
    """A row of pips - a skill rating - and the text it scores."""

    filled: int
    total: int
    size: float
    color: tuple[int, int, int]
    empty_color: Optional[tuple[int, int, int]]
    bbox: BBox
    owner: Optional[int] = None     # index of the block it scores
    # The pip's own proportions, so a template that scores with dashes is not
    # redrawn with dots.
    pip_width: float = 5.0
    pip_height: float = 5.0


@dataclass
class Panel:
    """A tint painted behind a column."""

    bbox: BBox
    fill: tuple[int, int, int]


@dataclass
class Banner:
    """A coloured band across the page, and the text set into it."""

    bbox: BBox
    fill: tuple[int, int, int]
    # The blocks it sits behind, in reading order. A banner with no text is
    # kept too: a bare coloured band is part of the design.
    owners: list[int] = field(default_factory=list)


@dataclass
class Frame:
    """A box drawn around some of the page's text."""

    bbox: BBox
    color: tuple[int, int, int]
    width: float
    fill: Optional[tuple[int, int, int]] = None
    owner: Optional[int] = None     # index of the first block it encloses


@dataclass
class Artwork:
    """Everything a page draws, sorted by what it belongs to."""

    ratings: list[Rating] = field(default_factory=list)
    panels: list[Panel] = field(default_factory=list)
    banners: list[Banner] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    # Drawings with no owner, kept for the fixed layer at their own positions.
    loose: list[DrawingElement] = field(default_factory=list)


    def __bool__(self) -> bool:
        return bool(self.ratings or self.panels or self.banners
                    or self.frames or self.loose)


def _is_pip(drawing: DrawingElement) -> bool:
    box = drawing.bbox
    if box.width > PIP_MAX_SIZE or box.height > PIP_MAX_SIZE:
        return False
    if box.width <= 0.5 or box.height <= 0.5:
        return False
    longer, shorter = max(box.width, box.height), min(box.width, box.height)
    # A dash is allowed, a hairline is not: a pip has body, and a rule of the
    # same length is a fraction of a point thick.
    if shorter > PIP_MAX_THICKNESS:
        return False
    return (shorter / longer) >= PIP_SQUARENESS and drawing.fill is not None


def _pip_rows(pips: list[DrawingElement]) -> list[list[DrawingElement]]:
    """Pips grouped into the rows they were drawn in.

    A rating is a run of evenly spaced pips on one baseline. Grouping by
    baseline alone would join two ratings that happen to sit level in
    different columns, so a gap wider than the pitch ends the row.
    """
    rows: list[list[DrawingElement]] = []
    for pip in sorted(pips, key=lambda d: (d.bbox.y0, d.bbox.x0)):
        for row in rows:
            last = row[-1]
            if (abs(pip.bbox.y0 - last.bbox.y0) <= PIP_ROW_TOLERANCE
                    and 0 <= pip.bbox.x0 - last.bbox.x1 <= PIP_MAX_GAP):
                row.append(pip)
                break
        else:
            rows.append([pip])
    return [row for row in rows if len(row) >= PIP_MIN_PER_ROW]


def _dominant(colors: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    counts: dict[tuple[int, int, int], int] = {}
    for color in colors:
        counts[color] = counts.get(color, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _read_rating(row: list[DrawingElement]) -> Optional[Rating]:
    """A row of pips as a score, if the row reads as one.

    A rating is drawn as filled pips and unfilled ones - "3 of 5". Which
    colour means filled is not knowable from one row in isolation, so the
    darker colour is taken as the filled one: a rating is drawn dark-on-light
    in every template this has been seen in, and a row of one colour is a
    full score either way.
    """
    colors = [pip.fill for pip in row if pip.fill]
    if not colors:
        return None
    if len(set(colors)) == 1:
        filled, empty = len(row), None
    else:
        # Luminance, not equality: the two colours differ per template.
        def luma(c: tuple[int, int, int]) -> float:
            return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]

        dark = min(set(colors), key=luma)
        light = max(set(colors), key=luma)
        filled = sum(1 for c in colors if c == dark)
        empty = light
        colors = [dark]
    box = BBox(min(p.bbox.x0 for p in row), min(p.bbox.y0 for p in row),
               max(p.bbox.x1 for p in row), max(p.bbox.y1 for p in row))
    return Rating(filled=filled, total=len(row),
                  size=max(row[0].bbox.width, row[0].bbox.height),
                  color=_dominant(colors), empty_color=empty, bbox=box,
                  pip_width=row[0].bbox.width, pip_height=row[0].bbox.height)


def _owner_above(box: BBox, page: Page) -> Optional[int]:
    """The block a decoration sits under - the text it decorates.

    A rating row is drawn beneath its label, so the owner is the nearest
    block whose foot is above the row and whose columns overlap it.
    """
    best, best_gap = None, PIP_MAX_TEXT_GAP
    for index, block in enumerate(page.blocks):
        if not block.text.strip():
            continue
        gap = box.y0 - block.bbox.y1
        if gap < -2.0 or gap > best_gap:
            continue
        overlap = (min(block.bbox.x1, box.x1) - max(block.bbox.x0, box.x0))
        if overlap <= 0:
            continue
        best, best_gap = index, gap
    return best


def _frame_sides(drawings: list[DrawingElement]) -> list[Frame]:
    """Boxes assembled from the four strokes a producer draws them as.

    A frame rarely arrives as one rectangle. It is four thin strokes - two
    tall, two wide - and read individually every one of them looks like a
    rule, which is how a name box became four stray lines across the page.
    """
    verticals, horizontals = [], []
    for drawing in drawings:
        box = drawing.bbox
        if box.width <= 2.5 and box.height >= box.width * FRAME_SIDE_RATIO:
            verticals.append(drawing)
        elif box.height <= 2.5 and box.width >= box.height * FRAME_SIDE_RATIO:
            horizontals.append(drawing)

    frames: list[Frame] = []
    used: set[int] = set()
    for i, left in enumerate(verticals):
        for j, right in enumerate(verticals):
            if j <= i or id(left) in used or id(right) in used:
                continue
            if abs(left.bbox.y0 - right.bbox.y0) > FRAME_CORNER_TOLERANCE:
                continue
            if abs(left.bbox.y1 - right.bbox.y1) > FRAME_CORNER_TOLERANCE:
                continue
            span = BBox(min(left.bbox.x0, right.bbox.x0), left.bbox.y0,
                        max(left.bbox.x1, right.bbox.x1), left.bbox.y1)
            if span.width < 20:
                continue
            top = bottom = None
            for rule in horizontals:
                if id(rule) in used:
                    continue
                if abs(rule.bbox.x0 - span.x0) > FRAME_CORNER_TOLERANCE:
                    continue
                if abs(rule.bbox.x1 - span.x1) > FRAME_CORNER_TOLERANCE:
                    continue
                if abs(rule.bbox.y0 - span.y0) <= FRAME_CORNER_TOLERANCE:
                    top = rule
                elif abs(rule.bbox.y1 - span.y1) <= FRAME_CORNER_TOLERANCE:
                    bottom = rule
            if top is None or bottom is None:
                continue
            for part in (left, right, top, bottom):
                used.add(id(part))
            frames.append(Frame(bbox=span, color=left.color or (0, 0, 0),
                                width=max(left.width, 0.6)))
    return frames


def pip_images(page: Page) -> list:
    """Images that are really rating pips, and are drawn as vectors as well.

    A template can lay its pips down twice: a small PNG per pip *and* the same
    row stroked as vector paths on top. The vector row is what `read_artwork`
    reads a score from, so the images are redundant - but the structure pass
    turns every image into a block of its own, which put thirty-five loose red
    dashes through the rebuilt text.

    Only an image that sits under a vector pip is returned, so a template
    whose pips are images *alone* keeps them: there the picture is the only
    record of the score, and dropping it would lose the rating entirely.
    """
    vectors = [d for d in page.drawings if _is_pip(d)]
    if not vectors:
        return []

    doubled = []
    for image in page.images:
        box = image.bbox
        if box.width > PIP_MAX_SIZE or box.height > PIP_MAX_SIZE:
            continue
        if min(box.width, box.height) > PIP_MAX_THICKNESS:
            continue
        if any(abs(v.bbox.x0 - box.x0) <= PIP_MAX_SIZE
               and abs(v.bbox.y0 - box.y0) <= PIP_ROW_TOLERANCE
               for v in vectors):
            doubled.append(image)
    return doubled


def read_artwork(page: Page) -> Artwork:
    """Sort a page's drawings by what each one belongs to."""
    art = Artwork()
    remaining = [d for d in page.drawings if d.kind != "underline"]

    # Panels first: a tall filled rect is a column's ground, and it must not
    # be mistaken for a frame or dragged into the loose layer where it would
    # paint over the text.
    rest: list[DrawingElement] = []
    for drawing in remaining:
        box = drawing.bbox
        if drawing.fill is None:
            rest.append(drawing)
            continue
        if (box.height >= page.height * PANEL_MIN_HEIGHT_SHARE
                and box.width <= page.width * PANEL_MAX_WIDTH_SHARE
                and box.width > 20):
            art.panels.append(Panel(bbox=box, fill=drawing.fill))
        elif (box.width >= page.width * BANNER_MIN_WIDTH_SHARE
                and box.height <= page.height * BANNER_MAX_HEIGHT_SHARE):
            art.banners.append(Banner(bbox=box, fill=drawing.fill,
                                      owners=_blocks_inside(box, page)))
        else:
            rest.append(drawing)
    remaining = rest

    # Two bands at the same place are one banner drawn twice - a producer
    # emits a thin strip and the full band on top of it. The taller wins.
    art.banners = _merge_banners(art.banners)

    # Ratings next, so their pips are not offered to the frame reader.
    pips = [d for d in remaining if _is_pip(d)]
    consumed: set[int] = set()
    for row in _pip_rows(pips):
        rating = _read_rating(row)
        if rating is None:
            continue
        rating.owner = _owner_above(rating.bbox, page)
        art.ratings.append(rating)
        for pip in row:
            consumed.add(id(pip))
        # A pip is rarely one drawing. A producer draws the fill and then
        # strokes its sides, so each dot arrives as a filled square plus two
        # or three thin slivers lying on top of it. Those slivers are part of
        # the pip and are consumed with it - left loose, they are emitted
        # separately and scatter dashes across the rebuilt page.
        for drawing in remaining:
            if id(drawing) not in consumed and _within(drawing.bbox, rating.bbox):
                consumed.add(id(drawing))
    # A pip-sized mark that is *not* part of a row stays in the running: it is
    # a bullet glyph or part of a logo, not a rating.
    remaining = [d for d in remaining if id(d) not in consumed]

    frames = _frame_sides(remaining)
    framed: set[int] = set()
    for frame in frames:
        if frame.bbox.height > page.height * FRAME_MAX_HEIGHT_SHARE:
            continue
        frame.owner = _first_block_inside(frame.bbox, page)
        if frame.owner is None:
            continue
        art.frames.append(frame)
        framed.add(id(frame))

    # Whatever is left has no owner. It is drawn where the source drew it.
    used_by_frames = {id(d) for d in remaining
                      if any(_within(d.bbox, f.bbox) for f in art.frames)}
    art.loose = [d for d in remaining if id(d) not in used_by_frames]
    return art


def _within(inner: BBox, outer: BBox) -> bool:
    return (inner.x0 >= outer.x0 - 2 and inner.x1 <= outer.x1 + 2
            and inner.y0 >= outer.y0 - 2 and inner.y1 <= outer.y1 + 2)


def buried_duplicates(page: Page) -> set[int]:
    """Blocks the source draws twice, of which one copy is buried in a panel.

    A template routinely carries a second copy of its header inside a solid
    banner - the copy a reader sees is the one below the banner, and the
    buried one is a leftover of how the template was built. Both extract like
    any other text, so a rebuild that ignores the paint prints the header
    twice, which is what put two contact rows at the top of a rebuilt CV.

    The evidence is the *duplication*, not the colour: text inside a panel is
    ordinary and must be kept, and judging visibility by contrast alone is
    guesswork - dark-on-mid-tone is hard to read but not absent, and a rule
    strict enough to catch it would delete legitimate text. So a block is
    dropped only when the same words appear elsewhere on the page outside any
    panel. That copy is the one a reader saw, and the other is redundant
    whether or not it was visible.
    """
    panels = [d for d in page.drawings
              if d.fill is not None and d.fill_opacity >= 0.95
              and d.bbox.width > 40 and d.bbox.height > 20]
    if not panels:
        return set()

    def key(block) -> str:
        return " ".join(block.text.split()).casefold()

    inside: dict[str, list[int]] = {}
    outside: set[str] = set()
    for index, block in enumerate(page.blocks):
        text = key(block)
        if not text:
            continue
        if any(_within(block.bbox, panel.bbox) for panel in panels):
            inside.setdefault(text, []).append(index)
        else:
            outside.add(text)

    return {index for text, indexes in inside.items() if text in outside
            for index in indexes}


def _blocks_inside(box: BBox, page: Page) -> list[int]:
    """Every block a band sits behind, in reading order."""
    inside = [index for index, block in enumerate(page.blocks)
              if block.text.strip() and _within(block.bbox, box)]
    return sorted(inside, key=lambda i: (page.blocks[i].bbox.y0,
                                         page.blocks[i].bbox.x0))


def _merge_banners(banners: list[Banner]) -> list[Banner]:
    """Collapse bands drawn on top of one another into one.

    A producer commonly lays a thin strip and the full band at the same top
    edge; emitted separately they become two coloured bars where the source
    showed one.
    """
    banners.sort(key=lambda b: (b.bbox.y0, -b.bbox.height))
    kept: list[Banner] = []
    for banner in banners:
        if any(abs(other.bbox.y0 - banner.bbox.y0) <= 6.0 for other in kept):
            continue
        kept.append(banner)
    return kept


def _first_block_inside(box: BBox, page: Page) -> Optional[int]:
    """The first block a frame encloses, in reading order."""
    inside = [index for index, block in enumerate(page.blocks)
              if block.text.strip() and _within(block.bbox, box)]
    if not inside:
        return None
    return min(inside, key=lambda i: (page.blocks[i].bbox.y0,
                                      page.blocks[i].bbox.x0))

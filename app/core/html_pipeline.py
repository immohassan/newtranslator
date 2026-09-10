"""PDF -> semantic HTML -> translate -> render, for documents that must reflow.

The coordinate-based rebuild in `rebuild_pdf` redraws each block at the place
it occupied in the source. That works while the translation is about the same
size as the original, and fails structurally when it is not: a block that grows
pushes nothing down, so every block below it keeps its old position and the
error accumulates. On a dense document the result is clean at the top and an
unreadable pile-up at the bottom - the signature of positional drift with no
reflow.

This module takes the other route. It reads the *structure* of the page rather
than its coordinates, emits semantic HTML, and lets a browser lay the document
out. Reflow, right-to-left mirroring, the bidi algorithm and Arabic shaping are
then handled by an engine that already implements them correctly, instead of
being approximated here.

The old path is untouched and remains the default; see `pipeline.run_pipeline`.
"""
from __future__ import annotations

import base64
import html
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import fitz

from . import fonts as fontlib
from .language import is_arabic
from .models import BBox, Document, Page, Span, TextBlock
from .qa import QAReport

BlockKind = Literal["heading", "paragraph", "bullet", "two_sided", "image",
                    "rule", "table"]

# --- structure detection thresholds --------------------------------------
# A line is a heading when it is bold and this much larger than body text.
HEADING_SIZE_RATIO = 1.05
# A line this much larger than body text is a heading even when it is not bold.
HEADING_SIZE_JUMP = 1.25
# ...provided it is short enough to be a title rather than large body text.
HEADING_MAX_CHARS = 80
# Characters a producer uses to open a list item. These are decoration: the
# list itself carries the meaning, so the marker is stripped and the renderer
# draws its own.
BULLET_CHARS = "•●▪◦‣⁃∙-*"
# Symbols that open a list item but are *not* decoration - a tick and a cross
# say "do" and "do not". They mark the item as an entry just as a bullet does,
# but they are kept, because dropping them would change what the line means.
MEANINGFUL_MARKERS = "✅✔☑❌✖✗⚠️⚠"
_MEANINGFUL_RE = re.compile(
    r"^[\s\u200b]*[" + re.escape(MEANINGFUL_MARKERS) + r"][\s\u200b]*")
_BULLET_RE = re.compile(
    r"^[\s\u200b]*[" + re.escape(BULLET_CHARS) + r"][\s\u200b]*")
# The separator after the number is often a zero-width space rather than an
# ordinary one - a producer emits it to hold the marker apart from the text -
# so it has to count as whitespace here or every numbered item is missed.
_ORDERED_RE = re.compile(r"^[\s\u200b]*(\d{1,2}|[a-zA-Z])[.)][\s\u200b]+")
# A line whose right edge reaches this share of the text column, and which
# holds a date, is the right-hand half of a "Job Title .... Date" row.
RIGHT_EDGE_SHARE = 0.86
_DATE_RE = re.compile(
    r"(19|20)\d\d\s*[-–—]|present\b|\b(19|20)\d\d\s*$", re.I)
# Two clusters on one line separated by at least this gap are a two-sided row.
TWO_SIDED_GAP = 60.0
# Two separate lines whose baselines differ by less than this share a row.
SAME_ROW_TOLERANCE = 4.0
# A drawn line this thin is a rule rather than a filled shape.
RULE_MAX_THICKNESS = 2.5
# ...and it must span this share of the text column to count as a section rule
# rather than an underline belonging to one phrase.
RULE_MIN_WIDTH_SHARE = 0.55
# A line indented from both margins by at least this much is centred, not a
# paragraph line that happens to fall short of the right edge.
CENTRED_MIN_INDENT = 24.0
# ...and its two indents must agree to within this share of the column. A
# fixed point tolerance is too tight: the column's right edge is set by the
# widest line on the page, so a genuinely centred line's two indents differ by
# a few points more than the eye can see.
CENTRED_BALANCE_SHARE = 0.12

# Text that must keep its own direction inside an RTL page.
_LTR_RE = re.compile(
    r"""(?xi)
      [\w.+-]+@[\w-]+\.[\w.-]+
    | (?:https?://|www\.)\S+
    | \b[\w-]+(?:\.[\w-]+)+/\S*
    | \+?\d[\d\s().-]{6,}\d
    """
)


@dataclass
class Fragment:
    """A run of text with the styling it should keep."""

    text: str
    bold: bool = False
    italic: bool = False
    size: float = 11.0
    color: tuple[int, int, int] = (0, 0, 0)
    translated: Optional[str] = None

    @property
    def output(self) -> str:
        return self.translated if self.translated is not None else self.text


@dataclass
class DocBlock:
    """One semantic element of the document."""

    kind: BlockKind
    fragments: list[Fragment] = field(default_factory=list)
    level: int = 2                  # heading rank
    right: list[Fragment] = field(default_factory=list)   # two_sided only
    # two_sided rows past the second cell. A row of three or more cells - a CV
    # banner, a court caption - keeps the middle ones here so they stay in the
    # row instead of being emitted as loose paragraphs beneath it.
    extra: list[list[Fragment]] = field(default_factory=list)
    image: Optional[bytes] = None
    image_ext: str = "png"
    centred: bool = False
    ordered: bool = False           # numbered list item
    standalone: bool = False        # an entry the source set on its own line
    # Where a standalone line starts, so a run of them can be tested for the
    # shared edge that tells a real list from a stack of title lines.
    indent: float = 0.0
    caption: bool = False           # two_sided: two independent text columns
    float_side: str = ""            # image: "start" when text runs beside it
    rows: list[list[str]] = field(default_factory=list)   # table only
    header: bool = False            # table's first row is a header
    width: float = 0.0
    height: float = 0.0
    # Set on the first and last block of a column when the page is read as
    # more than one, so the renderer can set the columns side by side instead
    # of stacking them. "" on every block of a single-column page, which is
    # what leaves that page rendering exactly as it did before.
    column_start: str = ""          # "main" | "sidebar"
    column_end: bool = False
    # Artwork that belongs to this block and travels with it: a rating row
    # drawn under it, a box drawn around it. See `artwork`.
    rating: Optional[object] = None
    frame: Optional[object] = None
    # Set on a column's first block when the source painted a tint behind it.
    tinted: Optional[tuple[int, int, int]] = None
    # A coloured band the source drew across the page. Marked on the first and
    # last block it covers, so the band wraps exactly that run and grows with
    # the translation instead of being a rectangle at a fixed height.
    banner_start: Optional[tuple[int, int, int]] = None
    banner_end: bool = False
    # Where this block sat on the source page. The vision reader keeps it so
    # the page's images, which the model is never asked about, can be slotted
    # back among the text in reading order.
    source_box: Optional[BBox] = None

    @property
    def text(self) -> str:
        return "".join(f.text for f in self.fragments)


# --------------------------------------------------------------------------
# 1. structural extraction
# --------------------------------------------------------------------------

def _line_text(line) -> str:
    return "".join(s.text for s in line.spans)


def _body_size(page: Page) -> float:
    """The most common span size on the page - its body text size."""
    counts: dict[float, int] = {}
    for block in page.blocks:
        for span in block.spans:
            if span.text.strip():
                key = round(span.size, 1)
                counts[key] = counts.get(key, 0) + len(span.text)
    if not counts:
        return 11.0
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _fragments(spans: list[Span], strip_marker: bool = False) -> list[Fragment]:
    out: list[Fragment] = []
    first = True
    for span in spans:
        text = span.text
        if strip_marker and first and text.strip():
            text = _BULLET_RE.sub("", text, count=1)
            text = _ORDERED_RE.sub("", text, count=1)
        if not text:
            continue
        first = False
        out.append(Fragment(text=text, bold=span.bold, italic=span.italic,
                            size=span.size, color=span.color))
    return out


def _is_two_sided(line) -> Optional[int]:
    """Index of the span that starts the right-hand cluster, if any."""
    spans = [s for s in line.spans if s.text.strip()]
    if len(spans) < 2:
        return None
    for i in range(len(spans) - 1):
        gap = spans[i + 1].bbox.x0 - spans[i].bbox.x1
        if gap >= TWO_SIDED_GAP:
            return i + 1
    return None


def _classify(line, body_size: float, page_width: float,
              text_left: float, text_right: float) -> tuple[BlockKind, dict]:
    """Decide what a single line is."""
    text = _line_text(line).strip()
    if not text:
        return "paragraph", {}

    spans = [s for s in line.spans if s.text.strip()]
    size = max((s.size for s in spans), default=body_size)
    # Weighted by characters, not by span count: a heading often carries a
    # trailing unbold space or zero-width joiner as a span of its own, and
    # requiring *every* span to be bold lets that one character veto the
    # heading - which is what folded section titles into the body text.
    bold_chars = sum(len(s.text.strip()) for s in spans if s.bold)
    total_chars = sum(len(s.text.strip()) for s in spans)
    bold = total_chars > 0 and bold_chars / total_chars >= 0.6

    ordered = _ORDERED_RE.match(text)
    if ordered:
        return "bullet", {"ordered": True, "marker": ordered.group(0).strip()}
    if _BULLET_RE.match(text):
        return "bullet", {}
    if _MEANINGFUL_RE.match(text):
        # An entry opened by a tick or a cross. The symbol stays in the text.
        return "bullet", {"keep_marker": True}

    split = _is_two_sided(line)
    if split is not None:
        return "two_sided", {"split": split}

    # A short right-aligned line carrying a date is the date half of a
    # "Job Title .... Date" row that the producer set as its own line.
    column = max(text_right - text_left, 1.0)
    reaches_right = (line.bbox.x1 - text_left) / column >= RIGHT_EDGE_SHARE
    if reaches_right and _DATE_RE.search(text) and len(text) < 60:
        return "two_sided", {"date_only": True}

    if bold and size >= body_size * HEADING_SIZE_RATIO:
        return "heading", {"size": size}
    # A heading need not be bold. A run set well above body size, short, and
    # not punctuated like a sentence is a title however its weight is set -
    # requiring bold folded whole section titles into the paragraph below.
    if size >= body_size * HEADING_SIZE_JUMP and len(text) <= HEADING_MAX_CHARS \
            and not text.rstrip().endswith((".", "!", "?", "،", "؛")):
        return "heading", {"size": size}

    return "paragraph", {}


def _inside(inner: BBox, outer: BBox) -> bool:
    """Whether `inner` sits within `outer`, allowing a point of slack."""
    return (inner.x0 >= outer.x0 - 2 and inner.x1 <= outer.x1 + 2
            and inner.y0 >= outer.y0 - 2 and inner.y1 <= outer.y1 + 2)


# A real table relates the cells across each of its rows. This share of rows
# must carry two or more filled cells for the detection to be believed.
GRID_MIN_FILLED_ROW_SHARE = 0.5

# A "table" covering this share of the page in both directions is the page's
# own layout frame - a ruled sidebar beside a body column - and not a grid.
PAGE_SPAN_SHARE = 0.9

# A grid cell carries a value. A cell holding more text than this is a column
# of prose that happens to align with its neighbour.
GRID_MAX_CELL_CHARS = 200


def _is_really_a_grid(rows: list[list[str]]) -> bool:
    """Whether a detected table is a grid rather than side-by-side text.

    PyMuPDF infers a table from alignment, so any passage set in two columns
    is reported as one - a court caption ("Plaintiff, ... STATEMENT OF NET
    WORTH ... Index No.") is the standard case. The two halves of a caption
    are independent runs of text that merely share a margin: the "rows" are
    the lines of the taller half, and each holds a cell on one side only.

    A real table is the opposite: a row exists precisely to relate the cells
    along it, so most of its rows carry more than one filled cell. That
    distinction is what separates the two here - the caption is released back
    to the paragraph path, and a genuine grid is still lifted out whole.
    """
    if not rows:
        return False
    # A cell holding a passage is a column of text whatever the row count: a
    # grid relates values, and a value does not run to a paragraph. This is
    # what a page set in two columns trips - each "cell" is half the page.
    if any(len(cell) > GRID_MAX_CELL_CHARS for row in rows for cell in row):
        return False
    if len(rows) < 2:
        # One row cannot show that its cells are related, so the share test
        # below has nothing to measure. Believe it only when the row reads as
        # a row of values - two or more filled cells side by side.
        return sum(1 for cell in rows[0] if cell.strip()) >= 2
    related = sum(1 for row in rows
                  if sum(1 for cell in row if cell.strip()) >= 2)
    return related >= len(rows) * GRID_MIN_FILLED_ROW_SHARE


# A rule this far outside a table's box is still that table's own border: the
# stroke has width, and a page's coordinates rarely land on the exact edge.
TABLE_RULE_SLACK = 6.0


def _rule_belongs_to_table(y: float, box: BBox) -> bool:
    """Whether a horizontal rule at `y` is part of this table's grid."""
    return box.y0 - TABLE_RULE_SLACK <= y <= box.y1 + TABLE_RULE_SLACK


def _find_tables(page: Page, source: "fitz.Page") -> list[tuple[BBox, list, bool]]:
    """Tables on the page, as (bbox, rows, has_header).

    A table is a grid, and a grid is the one structure that cannot survive
    being flattened into paragraphs: its cells only mean anything next to the
    cells beside them. PyMuPDF finds them from the ruling lines, which is also
    why those lines must not then be emitted as section rules.
    """
    found: list[tuple[BBox, list, bool]] = []
    try:
        tables = source.find_tables()
    except Exception:
        return found
    for table in tables.tables:
        try:
            rows = [[(cell or "").strip() for cell in row]
                    for row in table.extract()]
        except Exception:
            continue
        rows = [row for row in rows if any(cell for cell in row)]
        if len(rows) < 1 or len(rows[0]) < 2:
            continue
        if not _is_really_a_grid(rows):
            continue
        box = BBox(*table.bbox)
        # A "table" the size of the page is the page's own frame - the rules
        # around a sidebar read as a two-cell row - and lifting it out takes
        # the whole page with it, leaving one block where the layout was.
        if (box.width >= page.width * PAGE_SPAN_SHARE
                and box.height >= page.height * PAGE_SPAN_SHARE):
            continue
        # A header row is one whose cells are all short labels.
        header = all(len(cell) <= 40 for cell in rows[0])
        found.append((box, rows, header))
    return found


def _shares_row(line, others) -> bool:
    """Whether another line sits on this line's baseline."""
    return any(other is not line
               and abs(other.bbox.y0 - line.bbox.y0) <= SAME_ROW_TOLERANCE
               for other in others)


def _is_centred(line, text_left: float, text_right: float,
                siblings=()) -> bool:
    """Whether a line is centred in its column rather than set flush.

    Judged on the indent at each end, not on where the line's midpoint falls:
    a full-width paragraph line sits near the middle too. A centred line is
    pulled in from *both* margins by a comparable amount. Trailing whitespace
    is ignored, since a producer pads the run rather than moving the cursor and
    that padding would otherwise hide the right-hand indent.
    """
    spans = [s for s in line.spans if s.text.strip()]
    if not spans:
        return False
    # A line with neighbours on its own baseline is one cell of a row, not a
    # centred line. Its indents are set by the cells either side of it, and
    # for a middle cell they are naturally close to equal - which is exactly
    # the test below, so without this check every middle cell of a banner was
    # centred and broken out of its row.
    if _shares_row(line, siblings):
        return False
    right = spans[-1].bbox.x1
    left_pad = line.bbox.x0 - text_left
    right_pad = text_right - right
    if left_pad < CENTRED_MIN_INDENT or right_pad < CENTRED_MIN_INDENT:
        return False
    column = max(text_right - text_left, 1.0)
    return abs(left_pad - right_pad) <= column * CENTRED_BALANCE_SHARE


def _section_rules(page: Page, text_left: float,
                   text_right: float) -> list[tuple[float, float]]:
    """Horizontal rules on the page, as (y, width share) pairs.

    Only the wide ones are kept. A short rule under a phrase is that phrase's
    underline and travels with its text; a rule spanning most of the column is
    a section divider and is part of the document's structure.
    """
    column = max(text_right - text_left, 1.0)
    rules: list[tuple[float, float]] = []
    for drawing in page.drawings:
        box = drawing.bbox
        if box.height > RULE_MAX_THICKNESS:
            continue
        if drawing.kind not in ("underline", "line", "rect"):
            continue
        share = box.width / column
        if share >= RULE_MIN_WIDTH_SHARE:
            rules.append((box.y0, share))
    return sorted(rules)


# A caption's two columns must each hold at least this many lines - two lines
# beside two lines is a caption, one beside one is an ordinary row.
CAPTION_MIN_LINES = 2
# The right column must start beyond this share of the text column, so an
# indented continuation line is never mistaken for a second column.
CAPTION_COLUMN_SHARE = 0.45
# Above this share of right-hand lines sharing a baseline with a left-hand
# line, the region is a run of rows rather than two independent columns.
CAPTION_MAX_PAIRED_SHARE = 0.5
# A vertical gap wider than this ends the caption's right-hand column, so a
# page footer set in the same margin is not drawn into it.
CAPTION_MAX_GAP = 40.0
# An image needs this many lines beside it before it is floated rather than
# set as a block of its own.
IMAGE_FLOAT_MIN_LINES = 2
# A floated image is placed this far above its own top edge, so it precedes a
# header line whose baseline starts just above it.
IMAGE_FLOAT_LEAD = 12.0


def _float_side(image, lines) -> str:
    """"start" when the page runs text beside this image, else "".

    A portrait in the corner of a CV has the header set alongside it, not
    underneath. Emitted as a block element it would break that header in two
    and push everything below it down a page; floated, the text wraps beside
    it as the source had it.
    """
    beside = [l for l in lines
              if l.bbox.y1 > image.bbox.y0 and l.bbox.y0 < image.bbox.y1
              and l.bbox.x0 >= image.bbox.x1 - 2]
    if len(beside) < IMAGE_FLOAT_MIN_LINES:
        return ""
    # It must sit at a margin. An image with text on both sides is a figure
    # the producer placed inline, and floating it would reorder the page.
    if any(l.bbox.x1 <= image.bbox.x0 + 2 for l in lines
           if l.bbox.y1 > image.bbox.y0 and l.bbox.y0 < image.bbox.y1):
        return ""
    return "start"


def _column_fragments(column: list) -> list[Fragment]:
    """One column of a caption, its lines joined top to bottom."""
    out: list[Fragment] = []
    for line in sorted(column, key=lambda l: l.bbox.y0):
        if out and not out[-1].text.endswith(" "):
            out[-1].text += " "
        out.extend(_fragments([sp for sp in line.spans if sp.text]))
    return out


def _caption_region(lines: list, text_left: float,
                    text_right: float) -> Optional[tuple[list, list]]:
    """The two columns of a caption block, if the page opens with one.

    Detected as a contiguous run of lines that sit in two clusters - one at
    the left margin, one well to its right - where both clusters carry
    several lines and their vertical spans overlap. That is what makes a
    caption different from a run of "Job Title .... Date" rows: the columns
    are independent of each other, so their lines interleave instead of
    sharing baselines.
    """
    column = max(text_right - text_left, 1.0)
    split = text_left + column * CAPTION_COLUMN_SHARE

    left: list = []
    right: list = []
    for _, line in lines:
        (right if line.bbox.x0 >= split else left).append(line)
    if len(left) < CAPTION_MIN_LINES or len(right) < CAPTION_MIN_LINES:
        return None

    # The caption is the *first contiguous run* of the right-hand column. It
    # is bounded on the right column rather than on the overlap of the two,
    # because the left column carries straight on into the body text below;
    # and the run stops at the first large vertical gap, because a page
    # footer ("Page 1") also sits to the right of the split and would
    # otherwise stretch the caption over the whole page.
    right.sort(key=lambda l: l.bbox.y0)
    run = [right[0]]
    for line in right[1:]:
        if line.bbox.y0 - run[-1].bbox.y1 > CAPTION_MAX_GAP:
            break
        run.append(line)
    right = run
    if len(right) < CAPTION_MIN_LINES:
        return None
    top = min(l.bbox.y0 for l in right)
    bottom = max(l.bbox.y1 for l in right)
    if not any(l.bbox.y1 > top and l.bbox.y0 < bottom for l in left):
        return None

    # Only the overlapping band is the caption; text above or below it is
    # ordinary body copy and stays on the paragraph path.
    left = [l for l in left if l.bbox.y1 > top and l.bbox.y0 < bottom]
    if len(left) < CAPTION_MIN_LINES or len(right) < CAPTION_MIN_LINES:
        return None

    # The columns must be *independent*, which is what a caption is and what
    # separates it from a page of "Job Title .... Date" rows. In such a page
    # each right-hand line sits on a left-hand line's baseline; in a caption
    # the two columns are set to their own rhythms and mostly do not line up.
    # Without this test a whole resume reads as one caption.
    paired = sum(1 for r in right
                 if any(abs(r.bbox.y0 - l.bbox.y0) <= SAME_ROW_TOLERANCE
                        for l in left))
    if paired > len(right) * CAPTION_MAX_PAIRED_SHARE:
        return None
    return left, right


def extract_structure(page: Page, qa: QAReport,
                      source: Optional["fitz.Page"] = None) -> list[DocBlock]:
    """Read a page as a sequence of semantic blocks.

    Classification is done per *line*, not per block: PyMuPDF's blocks cut
    across list boundaries on a dense page, so a block is not a reliable unit
    of meaning. Consecutive lines of the same kind are then joined back into
    paragraphs and lists.
    """
    body = _body_size(page)
    lines = [(b, l) for b in page.blocks for l in b.lines if _line_text(l).strip()]
    if not lines:
        return []

    # Read the page in the order it is *seen*, not the order the producer
    # happened to store it in. A PDF's content stream carries no guarantee of
    # sequence, and a template-built CV routinely writes its section titles
    # last - so taken as stored, every heading on the page piled up at the
    # foot of it, stranded from the content it introduces, while that content
    # ran together under whichever heading came before.
    #
    # Lines sharing a baseline keep their left-to-right order, so the cells of
    # a row still arrive in sequence for the grouping passes below.
    lines.sort(key=lambda pair: (round(pair[1].bbox.y0 / SAME_ROW_TOLERANCE),
                                 pair[1].bbox.x0))

    # Tables are lifted out whole. Their cell text and their ruling lines are
    # then withheld from everything below, so a grid is not also emitted as a
    # run of paragraphs with stray rules between them.
    tables = _find_tables(page, source) if source is not None else []
    if tables:
        lines = [(b, l) for b, l in lines
                 if not any(_inside(l.bbox, box) for box, _, _ in tables)]
        if not lines:
            # The page is nothing but its grid - a sign-off sheet, a rota, a
            # form. Every line was consumed as a cell, so there is no running
            # text left to measure the column from, and everything below this
            # point describes text that is not there. The tables are the whole
            # page: emit them and stop.
            #
            # This is a crash rather than a layout fault when it is missed:
            # the text extents are taken with min()/max() over the lines, and
            # over none of them that raises, which failed the whole job.
            return [DocBlock(kind="table", rows=rows, header=header)
                    for _, rows, header in sorted(tables,
                                                  key=lambda t: t[0].y0)]

    text_left = min(l.bbox.x0 for _, l in lines)
    text_right = max(l.bbox.x1 for _, l in lines)
    # Kept before the row grouping below removes the cells it consumes, so the
    # centring test can still see that a line had neighbours on its baseline.
    all_lines = [l for _, l in lines]

    # A grid's own ruling lines are not section dividers. They are excluded by
    # the table's full box rather than by its y-range alone: a table's top
    # border sits exactly on that boundary, and the row separators just inside
    # it, so a rule tested only against the interior escaped and was drawn
    # again as a divider - the stray lines that appeared under a letterhead.
    rules = [r for r in _section_rules(page, text_left, text_right)
             if not any(_rule_belongs_to_table(r[0], box) for box, _, _ in tables)]

    # A caption sets two independent columns side by side: the parties down
    # the left, the document's own labels down the right. Its lines interleave
    # rather than pair off, so neither the row grouping below nor the
    # paragraph run can read it - flattened in document order the two columns
    # comb together into nonsense ("Index No. Date Action Commenced:
    # Defendant."). It is lifted out first, as one block holding each column
    # whole.
    caption_lines: set[int] = set()
    caption_block: Optional[DocBlock] = None
    caption_y = 0.0
    region = _caption_region(lines, text_left, text_right)
    if region:
        left_col, right_col = region
        caption_y = min(l.bbox.y0 for l in left_col + right_col)
        caption_block = DocBlock(
            kind="two_sided",
            fragments=_column_fragments(left_col),
            right=_column_fragments(right_col),
            caption=True,
        )
        caption_lines = {id(l) for l in left_col + right_col}
        lines = [pair for pair in lines if id(pair[1]) not in caption_lines]

    # A producer often sets "Job Title .... Date" as separate lines that share
    # a baseline: the title flush left, the trailing half flush right. PyMuPDF
    # reports them as separate lines, so they are grouped here before anything
    # else looks at them - left alone they become separate paragraphs and each
    # trailing half wraps into a narrow column of its own.
    #
    # A row is *every* line on that baseline, not just the first two. A CV
    # banner ("Place of birth: ... | Nationality: ... | Gender: ... | Phone
    # number:") and a court caption both set three or more cells across one
    # row; pairing only the first with the last left the cells between them to
    # be emitted as stray paragraphs in the middle of the header.
    column = max(text_right - text_left, 1.0)
    row_cells: dict[int, list[Any]] = {}   # id(first line) -> lines after it
    consumed: set[int] = set()
    for i, (_, first) in enumerate(lines):
        if id(first) in consumed:
            continue
        rest: list[Any] = []
        cursor = first
        for j in range(i + 1, len(lines)):
            second = lines[j][1]
            if id(second) in consumed:
                continue
            if abs(second.bbox.y0 - first.bbox.y0) > SAME_ROW_TOLERANCE:
                continue
            if second.bbox.x0 <= cursor.bbox.x1:
                continue
            if not [sp for sp in second.spans if sp.text.strip()]:
                continue
            rest.append(second)
            cursor = second
        if not rest:
            continue
        # The row must actually span the column. A line broken into two by a
        # font change mid-sentence also shares a baseline, and turning that
        # into a two-column row would push its own tail to the far margin.
        last_spans = [sp for sp in rest[-1].spans if sp.text.strip()]
        if (last_spans[-1].bbox.x1 - text_left) / column < RIGHT_EDGE_SHARE:
            continue
        row_cells[id(first)] = rest
        for line in rest:
            consumed.add(id(line))

    lines = [pair for pair in lines if id(pair[1]) not in consumed]

    heading_sizes: set[float] = set()
    classified: list[tuple[str, dict, Any]] = []
    for owner, line in lines:
        kind, meta = _classify(line, body, page.width, text_left, text_right)
        if kind == "heading":
            heading_sizes.add(round(meta.get("size", body), 1))
        meta["centred"] = _is_centred(line, text_left, text_right,
                                      all_lines)
        meta["y"] = line.bbox.y0
        meta["block"] = id(owner)
        meta["solo"] = len([l for l in owner.lines if _line_text(l).strip()]) == 1
        classified.append((kind, meta, line))

    # Heading rank: the largest heading size on the page is h1, next is h2...
    ranks = {size: i + 1 for i, size in enumerate(sorted(heading_sizes,
                                                         reverse=True))}

    blocks: list[DocBlock] = []
    pending: list[Fragment] = []

    pending_centred = False
    pending_block: Any = None

    def flush_paragraph() -> None:
        nonlocal pending, pending_centred, pending_block
        if pending:
            blocks.append(DocBlock(kind="paragraph", fragments=pending,
                                   centred=pending_centred))
            pending = []
            pending_centred = False
            pending_block = None

    pending_tables = sorted(tables, key=lambda t: t[0].y0)
    table_at = 0
    rule_at = 0

    # Images are placed by where they sit on the page, in step with the text,
    # rather than being appended after it. A header photograph sits above the
    # first line of a CV; emitted last it landed at the foot of the page's
    # content and reflow then carried it onto the following page.
    pending_images = sorted(
        (im for im in page.images if im.data), key=lambda im: im.bbox.y0)
    image_at = 0

    def emit_images_above(y: float) -> None:
        nonlocal image_at
        # A floated image is emitted *before* the text that wraps beside it,
        # so its placement allows for a header line that starts fractionally
        # above the image's own top edge.
        while image_at < len(pending_images) and (
                pending_images[image_at].bbox.y0
                - (IMAGE_FLOAT_LEAD
                   if _float_side(pending_images[image_at], all_lines)
                   else 0.0)) <= y:
            flush_paragraph()
            image = pending_images[image_at]
            blocks.append(DocBlock(kind="image", image=image.data,
                                   image_ext=image.ext or "png",
                                   width=image.bbox.width,
                                   height=image.bbox.height,
                                   source_box=image.bbox,
                                   float_side=_float_side(image, all_lines)))
            image_at += 1

    def emit_tables_above(y: float) -> None:
        nonlocal table_at
        while table_at < len(pending_tables) and pending_tables[table_at][0].y0 <= y:
            flush_paragraph()
            box, rows, header = pending_tables[table_at]
            blocks.append(DocBlock(kind="table", rows=rows, header=header))
            table_at += 1

    def emit_caption_above(y: float) -> None:
        """Place the caption block at the point its own lines occupied."""
        nonlocal caption_block
        if caption_block is not None and caption_y <= y:
            flush_paragraph()
            blocks.append(caption_block)
            caption_block = None

    def emit_rules_above(y: float) -> None:
        """Place any section rule that sits above this line, in order."""
        nonlocal rule_at
        while rule_at < len(rules) and rules[rule_at][0] <= y:
            flush_paragraph()
            blocks.append(DocBlock(kind="rule"))
            rule_at += 1

    for index, (kind, meta, line) in enumerate(classified):
        emit_caption_above(meta.get("y", 0.0))
        emit_images_above(meta.get("y", 0.0))
        emit_rules_above(meta.get("y", 0.0))
        emit_tables_above(meta.get("y", 0.0))
        spans = [s for s in line.spans if s.text]
        trailing_cells = row_cells.get(id(line))
        trailing = trailing_cells[0] if trailing_cells else None

        if kind == "heading" and trailing is None:
            flush_paragraph()
            level = min(ranks.get(round(meta.get("size", body), 1), 2), 6)
            blocks.append(DocBlock(kind="heading", fragments=_fragments(spans),
                                   level=level,
                                   centred=bool(meta.get("centred"))))
        elif trailing_cells:
            # Grouped with every line sharing its baseline.
            flush_paragraph()
            rest = [_fragments([sp for sp in cell.spans if sp.text])
                    for cell in trailing_cells]
            # `right` is the far end of the row, so it takes the *last* cell;
            # anything between the two ends keeps its order in `extra`.
            blocks.append(DocBlock(
                kind="two_sided",
                fragments=_fragments(spans),
                right=rest[-1],
                extra=rest[:-1],
            ))
        elif kind == "bullet":
            flush_paragraph()
            blocks.append(DocBlock(
                kind="bullet",
                fragments=_fragments(
                    spans, strip_marker=not meta.get("keep_marker")),
                ordered=bool(meta.get("ordered")),
            ))
        elif kind == "two_sided":
            if meta.get("date_only"):
                # Pair it with the title it belongs to. That title is normally
                # the line immediately above, which at this point is still
                # sitting in `pending` - it has not been flushed into a
                # paragraph yet - so the open run is checked first.
                if pending:
                    title = pending
                    pending = []
                    blocks.append(DocBlock(kind="two_sided",
                                           fragments=title,
                                           right=_fragments(spans)))
                    continue
                if blocks and blocks[-1].kind in ("paragraph", "heading") \
                        and not blocks[-1].right:
                    blocks[-1].right = _fragments(spans)
                    blocks[-1].kind = "two_sided"
                    continue
                blocks.append(DocBlock(kind="paragraph",
                                       fragments=_fragments(spans)))
            else:
                flush_paragraph()
                split = meta["split"]
                strong = [s for s in spans if s.text.strip()]
                blocks.append(DocBlock(
                    kind="two_sided",
                    fragments=_fragments(strong[:split]),
                    right=_fragments(strong[split:]),
                ))
        else:
            # Consecutive body lines belong to one paragraph. A centred line
            # is its own paragraph: joining it to flush text beside it would
            # lose the centring for both.
            centred = bool(meta.get("centred"))
            if centred != pending_centred:
                flush_paragraph()
            pending_centred = centred

            # A producer that emits every item of a list as its own one-line
            # block is writing a list without marker characters. Running those
            # together makes a paragraph out of what the reader sees as
            # separate entries, so a line that stands alone in its block ends
            # the run rather than joining it.
            solo = bool(meta.get("solo"))
            block_id = meta.get("block")
            if solo or block_id != pending_block:
                flush_paragraph()
            pending_block = block_id
            if solo:
                blocks.append(DocBlock(kind="paragraph",
                                       fragments=_fragments(spans),
                                       centred=centred, standalone=True,
                                       indent=line.bbox.x0))
                continue
            if pending and not pending[-1].text.endswith(" "):
                pending[-1].text += " "
            pending.extend(_fragments(spans))
    flush_paragraph()
    emit_caption_above(float("inf"))
    emit_rules_above(float("inf"))
    emit_tables_above(float("inf"))
    emit_images_above(float("inf"))

    qa.add(
        "structure",
        "info",
        f"Page {page.number + 1} was read as {len(blocks)} semantic block(s).",
        page=page.number + 1,
        count=len(blocks),
    )
    return blocks


# --------------------------------------------------------------------------
# 2. HTML generation
# --------------------------------------------------------------------------

def _dir_for(text: str, rtl_page: bool) -> str:
    """The `dir` a fragment needs, or "" when it inherits correctly.

    A URL, an email address or a phone number reads left-to-right whatever
    surrounds it, and so does any Latin-script run on an Arabic page. Marking
    it is one attribute; the browser's bidi implementation does the rest.
    """
    body = (text or "").strip()
    if not body:
        return ""
    if not rtl_page:
        return "rtl" if is_arabic(body) else ""
    if _LTR_RE.search(body):
        return "ltr"
    return "" if is_arabic(body) else "ltr"


def _style_for(fragment: Fragment, base: float, scale: float) -> str:
    bits = []
    if fragment.bold:
        bits.append("font-weight:700")
    if fragment.italic:
        bits.append("font-style:italic")
    size = fragment.size * scale
    if abs(size - base) > 0.4:
        bits.append(f"font-size:{size:.1f}pt")
    if fragment.color != (0, 0, 0):
        bits.append("color:#%02x%02x%02x" % fragment.color)
    return ";".join(bits)


def _render_fragments(fragments: list[Fragment], rtl_page: bool,
                      base: float, scale: float) -> str:
    parts = []
    for fragment in fragments:
        text = fragment.output
        if not text:
            continue
        style = _style_for(fragment, base, scale)
        direction = _dir_for(text, rtl_page)
        escaped = html.escape(text)
        if not style and not direction:
            parts.append(escaped)
            continue
        attrs = ""
        if direction:
            attrs += f' dir="{direction}"'
        if style:
            attrs += f' style="{style}"'
        parts.append(f"<span{attrs}>{escaped}</span>")
    return "".join(parts)


def _main_column_end(blocks: list[DocBlock]) -> Optional[int]:
    """Index of the block that closes the page's main column, if it has one.

    A column runs from the block carrying its `column_start` to the next
    `column_end` after it, so the two are paired by walking the list rather
    than by assuming the main column is the first or the last.
    """
    current = ""
    for index, block in enumerate(blocks):
        if block.column_start:
            current = block.column_start
        if block.column_end:
            if current == "main":
                return index
            current = ""
    return None


def _join_continuation_pages(
        pages: list[list[DocBlock]]) -> list[list[DocBlock]]:
    """Merge a page that continues a column into the page that opened it.

    The pages are emitted as one continuous flow, but each is rendered on its
    own and the column marks are decided per page: a page read as several
    columns marks its first and last block, a page read as one marks nothing.
    Rendered separately, a single-column page that follows a two-column one is
    closed out of the flex row and set full width - stacked underneath both
    columns rather than continuing the one it belongs to.

    That is what turned a two-page CV into three. The sidebar and the main
    column both ended with page one, and page two's text - the tail of the
    main column - was laid out across the whole measure below them, taking a
    page and a half to say what had taken half a page.

    The continuation's blocks are appended to the *same* list as the column
    they continue, so one `_render_page` call sees the whole column and closes
    it once, in the right place. Moving the mark alone is not enough: the
    closing tag has to be emitted by the call that opened it.
    """
    if len(pages) < 2:
        return pages

    joined: list[list[DocBlock]] = [list(pages[0])]
    for blocks in pages[1:]:
        target = joined[-1]
        if (blocks and target
                # A page that reads as its own columns starts a new layout.
                and not any(b.column_start for b in blocks)
                and any(b.column_end for b in target)):
            # The continuation belongs to the *main* column, which is not
            # necessarily the last one on the page: a CV sets the body first
            # and the sidebar after it, so appending to the end of the list
            # would drop page two's text under the sidebar instead of
            # carrying on the column it continues. The blocks are spliced in
            # where that column ends.
            main_end = _main_column_end(target)
            if main_end is not None:
                target[main_end].column_end = False
                for offset, block in enumerate(blocks, start=1):
                    target.insert(main_end + offset, block)
                target[main_end + len(blocks)].column_end = True
                continue
        joined.append(list(blocks))
    return joined


# The body size to fall back on when a document carries no usable span sizes.
# The page's own margin, in inches. Bound to one name because the full-bleed
# banner has to pull out by exactly this much to reach the paper's edge.
PAGE_MARGIN_IN = 0.75

DEFAULT_BODY_PT = 11.0
# A rebuild is held within this much of the source's own body size. A template
# may set body copy very small to fit a dense page; following it below this
# would rebuild an unreadable document.
MIN_BODY_PT = 8.0
MAX_BODY_PT = 12.0


def _source_body_size(pages: list[list[DocBlock]]) -> float:
    """The body text size of the source, from the text it uses most.

    Measured by weight of characters rather than by counting fragments, so a
    page's few large headings cannot outvote the body copy underneath them.
    """
    weight: dict[float, int] = {}
    for blocks in pages:
        for block in blocks:
            # Headings carry their own sizes and are what this must not follow.
            if block.kind == "heading":
                continue
            for fragment in block.fragments:
                size = round(fragment.size, 1)
                if size > 0:
                    weight[size] = weight.get(size, 0) + len(fragment.text)
    if not weight:
        return DEFAULT_BODY_PT
    body = max(weight.items(), key=lambda item: item[1])[0]
    return min(max(body, MIN_BODY_PT), MAX_BODY_PT)


def build_html(pages: list[list[DocBlock]], direction: str, page_size: BBox,
               qa: QAReport, fit: float = 1.0) -> str:
    """Turn the extracted structure into a standalone HTML document.

    `fit` tightens the whole page - type size and leading together - so a
    rebuild that spilled just past the source's last page can be drawn back
    onto it. 1.0 is the natural setting; see `_fit_to_source_pages`.
    """
    rtl = direction == "en2ar"
    # Set from the source's own body text rather than a fixed 11pt. A template
    # that sets its body at 9pt was being rebuilt a fifth larger, and on a page
    # that was already full that alone pushed the tail of it onto another
    # sheet. A document that carries no usable sizes falls back to 11pt.
    base = _source_body_size(pages)
    # Arabic reads smaller than Latin at the same point size, so it is set a
    # little larger; going the other way it is set smaller. Same rule the
    # coordinate pipeline applies per span, expressed once in CSS.
    scale = ((base + fontlib.ARABIC_SIZE_BONUS) / base if rtl
             else fontlib.AR_TO_EN_SIZE_SCALE)
    scale *= fit
    # Leading is tightened with the type, but only half as hard: squeezing the
    # line boxes as much as the glyphs is what makes a shrunk page look
    # cramped rather than merely smaller.
    leading = 1.5 - (1.0 - fit) * 0.75

    width_in = page_size.width / 72.0
    height_in = page_size.height / 72.0

    arabic_font = fontlib._locate(fontlib.ARABIC_FONTS["regular"])
    arabic_bold = fontlib._locate(fontlib.ARABIC_FONTS["bold"])
    latin_font = fontlib._locate(fontlib.LATIN_FONTS["regular"])
    latin_bold = fontlib._locate(fontlib.LATIN_FONTS["bold"])

    def face(name: str, path: Optional[str], weight: int) -> str:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "rb") as fh:
            blob = base64.b64encode(fh.read()).decode("ascii")
        return (f"@font-face{{font-family:'{name}';font-weight:{weight};"
                f"src:url(data:font/ttf;base64,{blob}) format('truetype');}}")

    faces = "".join([
        face("DocArabic", arabic_font, 400),
        face("DocArabic", arabic_bold, 700),
        face("DocLatin", latin_font, 400),
        face("DocLatin", latin_bold, 700),
    ])

    # The source page boundaries are deliberately *not* reproduced. They record
    # where the original text happened to run out of room, which is no longer
    # where the translation does - forcing a break there leaves one page half
    # empty and pushes its remainder onto the next. The content is emitted as
    # one flow and the renderer paginates it.
    body_parts = [_render_page(blocks, rtl, base, scale)
                  for blocks in _join_continuation_pages(pages)]

    return f"""<!doctype html>
<html lang="{'ar' if rtl else 'en'}" dir="{'rtl' if rtl else 'ltr'}">
<head>
<meta charset="utf-8">
<style>
{faces}
/* The vertical margin is set here because `@page` is the only box that
   repeats on every printed page - body padding is applied once to the
   whole flow, so a document set that way keeps its margin on page one
   and runs off the top of every page after it. The sides are zero, and
   the text's side margin is set on the body instead: that is what lets a
   full-bleed masthead cancel it and reach the paper's edge. */
@page {{ size: {width_in:.2f}in {height_in:.2f}in;
        margin: {PAGE_MARGIN_IN}in 0; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  /* Only the side margin is set here, so a full-bleed element can cancel it
     with a negative margin of the same size. The top and bottom margin is
     the printer's, because body padding applies once to the whole flow
     rather than to every page. */
  padding: 0 {PAGE_MARGIN_IN}in;
  box-sizing: border-box;
  font-family: 'DocArabic', 'DocLatin', serif;
  font-size: {base * scale:.1f}pt;
  /* Real line boxes, computed by the layout engine, rather than a
     hand-rolled leading value. */
  line-height: {leading:.3f};
  color: #000;
}}
h1, h2, h3, h4, h5, h6 {{ margin: 0.7em 0 0.3em; line-height: 1.3; }}
h1 {{ font-size: {base * scale * 1.45:.1f}pt; }}
h2 {{ font-size: {base * scale * 1.18:.1f}pt; }}
h3 {{ font-size: {base * scale * 1.05:.1f}pt; }}
p {{ margin: 0 0 0.55em; }}
ul, ol {{ margin: 0 0 0.6em; padding-inline-start: 1.6em; }}
li {{ margin: 0 0 0.25em; }}
/* A page set in columns. `row` follows the document direction, so an Arabic
   page puts the first column on the right with no per-column logic, exactly
   as .two-sided below does for a single row. */
.columns {{
  display: flex;
  flex-direction: row;
  gap: 1.6em;
  align-items: flex-start;
}}
.column {{ min-width: 0; }}
/* A column's tint is painted on the column itself, so it grows with the text
   instead of being a rectangle at a coordinate the text has moved away from. */
.column-sidebar.tinted {{
  padding: 0.8em 1em;
  margin-block-start: -0.8em;
}}
/* A rating row. `--pip` carries the source's own dot size, and the row
   follows the page direction, so an Arabic page fills from the right. */
.rating {{
  display: flex;
  gap: calc(var(--pip-h) * 0.7);
  align-items: center;
  margin: 0.15em 0 0.7em;
}}
.rating i {{
  width: var(--pip-w);
  height: var(--pip-h);
  border-radius: var(--pip-r);
  display: inline-block;
  flex: none;
}}
/* A coloured band across the page. Full-bleed - the negative margin cancels
   the page's own padding - because a banner that stops at the text margin
   reads as a box, which is not what the source drew. */
.banner {{
  /* Full bleed. The band is pulled out by the page margin on three sides so
     it runs to the paper's edge, as the source draws it: a header that stops
     at the text margin reads as a box on the page rather than as the page's
     own masthead. The margin is a fixed 0.75in, so the pull is too. */
  /* Pulled up into the page's own top margin as well as out to both sides,
     so the band starts at the paper's edge exactly as the source draws it. */
  margin: -{PAGE_MARGIN_IN}in -{PAGE_MARGIN_IN}in 1.2em;
  /* The padding puts back everything the negative margin just took: the sides
     restore the text margin, so the words inside the band line up with the
     columns below it, and the top restores the page margin the band was
     pulled up through - without it the name and the portrait are dragged off
     the top of the sheet along with the band. */
  padding: calc({PAGE_MARGIN_IN}in + 0.6em) {PAGE_MARGIN_IN}in 1.4em;
  /* The portrait sits beside the name, as the source sets it, rather than
     above it: a stacked header is twice the height and reads as a different
     design. `center` keeps the name level with the middle of the photo. */
  display: flex;
  align-items: center;
  gap: 1em;
  flex-wrap: wrap;
}}
/* The picture keeps its own size and never stretches to the flex line. */
.banner img {{
  flex: none;
  border-radius: 50%;
  object-fit: cover;
}}
/* The text beside the picture is one column of its own, so the name and the
   title stack against each other rather than sitting side by side. */
.banner-text {{
  flex: 1 1 0;
  min-width: 0;
}}
/* Text colour is set per banner from its own fill - see `_open_banner` - so a
   pale band keeps dark text and only a dark one is reversed out. `!important`
   is needed because each span carries its own colour inline, and the source
   stores those to suit the band it painted behind them, not this one. */
.banner * {{ color: inherit !important; }}
.banner h1, .banner h2, .banner h3, .banner p {{ margin: 0 0 0.2em; }}
/* A box drawn around a block travels with it and grows to fit the
   translation, which a fixed rectangle could not do. */
.frame {{
  padding: 0.7em 1em;
  margin: 0 0 0.9em;
}}
/* The body takes the room left over; the sidebar keeps its narrower measure,
   which is what makes it read as a sidebar rather than a second body. */
/* `flex-basis` is the share of the row the body column starts from, rather
   than `auto` - which sizes it from its content and lets a long unbreakable
   word in the sidebar squeeze it toward nothing. That collapse is what set a
   whole CV one word to a line, on about half of otherwise identical runs.
   `min-width` is the floor it can never be squeezed below. */
.column-main {{ flex: 1 1 60%; min-width: 50%; }}
.column-sidebar {{ flex: 0 0 30%; min-width: 0; }}
/* One rule mirrors every "Job Title .... Date" row: under dir=rtl the two
   ends swap without any per-row logic. */
.two-sided {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1.5em;
  margin: 0 0 0.35em;
}}
.two-sided .lead {{ flex: 0 1 auto; }}
.two-sided .cell {{ flex: 0 1 auto; }}
.two-sided .trail {{ flex: 0 1 auto; }}
/* A two-cell row is the "Job Title .... Date" case: the trailing half is a
   date and must not be broken across lines. A row of three or more is a
   banner of labelled fields, which has to be free to wrap. */
.two-sided.pair .lead {{ flex: 1 1 auto; }}
.two-sided.pair .trail {{ flex: 0 0 auto; white-space: nowrap; }}
/* The cells of a banner are sized to their content and spread across the
   row. They may wrap when the translation is too wide to fit, but each cell
   wraps as a unit - a cell is a labelled field and splitting one across two
   lines separates the label from its value. */
.two-sided.row-wrap {{ flex-wrap: wrap; justify-content: flex-start;
  gap: 0.35em 1.2em; }}
.two-sided.row-wrap > span {{ flex: 0 1 auto; }}
/* A caption is two independent columns of text, so each half takes half the
   width and wraps inside it rather than being held on one line. */
.two-sided.caption {{ align-items: flex-start; }}
.two-sided.caption > span {{ flex: 1 1 0; white-space: normal; }}
img {{ max-width: 100%; height: auto; display: block; margin: 0.6em 0; }}
/* An image the source set alongside its text keeps the text beside it. The
   float is on the start edge, so it moves to the right under dir=rtl without
   any per-image logic. */
img.float-start {{
  float: inline-start;
  margin: 0 0 0.5em 0;
  margin-inline-end: 0.9em;
}}
/* A table is free to break across pages. Held together it can only move as
   a unit, so a grid taller than the room left below the heading above it
   jumps to the next page whole and leaves the first one nearly empty - which
   is worse than a break, and is what a long schedule did. Rows themselves
   stay intact, and the header repeats on each page the table continues onto,
   so a split is readable. */
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 0.5em 0 0.8em;
}}
thead {{ display: table-header-group; }}
tr {{ break-inside: avoid; }}
th, td {{
  border: 0.5pt solid currentColor;
  padding: 0.3em 0.5em;
  text-align: start;
  vertical-align: top;
}}
th {{ font-weight: 700; }}
/* Section dividers, kept from the source. Drawn as a border rather than a
   filled box so it stays a hairline at any zoom. */
hr.section-rule {{
  border: 0;
  border-top: 0.75pt solid currentColor;
  margin: 0.15em 0 0.5em;
  opacity: 0.75;
  /* A divider introduces what follows it, so it must not be the last thing on
     a page: broken away from its section it reads as a line ruled under the
     page rather than as the start of anything. */
  break-after: avoid;
}}
/* Centred text stays centred whichever way the page reads. */
.centred {{ text-align: center; }}
/* Keep a heading with the text it introduces, and never strand a single
   line of a paragraph across a page boundary. */
h1, h2, h3, h4, h5, h6 {{ break-after: avoid; }}
li, p {{ orphans: 2; widows: 2; }}
.two-sided {{ break-inside: avoid; }}
[dir="ltr"] {{ unicode-bidi: isolate; }}
</style>
</head>
<body>
{"".join(body_parts)}
</body>
</html>"""


def _render_table(block: DocBlock, rtl: bool, base: float,
                  scale: float) -> str:
    """A real <table>, so its cells stay beside the cells they belong with."""
    rows = block.rows
    if not rows:
        return ""
    out = ["<table>"]
    body_from = 0
    if block.header:
        cells = "".join(
            f"<th>{_render_fragments([Fragment(c)], rtl, base, scale)}</th>"
            for c in rows[0])
        out.append(f"<thead><tr>{cells}</tr></thead>")
        body_from = 1
    out.append("<tbody>")
    for row in rows[body_from:]:
        cells = "".join(
            f"<td>{_render_fragments([Fragment(c)], rtl, base, scale)}</td>"
            for c in row)
        out.append(f"<tr>{cells}</tr>")
    out.append("</tbody></table>")
    return "".join(out)


# A run of markerless standalone lines is only a list when it is long enough
# to read as one. A pair of lines under a title is a subtitle, not two bullets.
MIN_MARKERLESS_LIST = 3
# ...and its entries must start from the same edge, within this much slack.
LIST_EDGE_TOLERANCE = 4.0


def _markerless_lists(blocks: list[DocBlock]) -> set[int]:
    """Which standalone paragraphs should be rendered as list items.

    A producer that sets every entry of a list on its own line writes a list
    with no marker characters, and running those entries together loses the
    separation the reader sees. But a designed page sets *every* line as its
    own block - the title, the subtitle, the byline under it - and treating
    each of those as an entry decorates the whole title block with bullets
    that were never in the source.

    A real list is told from a title block by the two things that make it one:
    its entries stack up from a shared left edge, and there are enough of them
    to be a list rather than a couple of lines under a heading. A centred line
    is never an entry - centring is what a title does and what a list cannot.
    """
    listed: set[int] = set()
    run: list[DocBlock] = []

    def close() -> None:
        if len(run) >= MIN_MARKERLESS_LIST:
            edge = min(b.indent for b in run)
            if all(abs(b.indent - edge) <= LIST_EDGE_TOLERANCE for b in run):
                listed.update(id(b) for b in run)
        run.clear()

    for block in blocks:
        if block.kind == "bullet":
            # An explicit marker is a list on its own account; it neither joins
            # a markerless run nor breaks one.
            continue
        if block.kind == "paragraph" and block.standalone and not block.centred:
            run.append(block)
        else:
            close()
    close()
    return listed


def _drop_buried_duplicates(page: Page, qa: QAReport) -> None:
    """Remove a block the source also draws outside a panel.

    Done before the structure is read, so the duplicate never becomes part of
    a paragraph and cannot be translated or paid for twice.
    """
    from .artwork import buried_duplicates

    buried = buried_duplicates(page)
    if not buried:
        return
    page.blocks = [b for i, b in enumerate(page.blocks) if i not in buried]
    qa.add(
        "layout",
        "info",
        f"Page {page.number + 1}: {len(buried)} block(s) the template also "
        f"draws inside a coloured panel were dropped, since the same text "
        f"appears again outside it.",
        page=page.number + 1,
        count=len(buried),
    )


def _consume_pip_images(page: Page) -> None:
    """Take the images that are really rating pips out of the page.

    Run before the structure is read, because that pass turns every remaining
    image into a block of its own: a template that draws its pips as small
    PNGs would otherwise scatter thirty-odd inline pictures through the text.
    The pips themselves are re-read from `page.drawings` by `read_artwork`,
    which is given the same images and groups them into rows.
    """
    from .artwork import pip_images

    doubled = pip_images(page)
    if not doubled:
        return
    taken = {id(image) for image in doubled}
    page.images = [im for im in page.images if id(im) not in taken]


# A coloured band at least this wide is the page's own masthead rather than
# decoration inside one column, and is set across the top of the page.
BANNER_PAGE_SPAN = 0.9


def _hoist(blocks: list[DocBlock], run: list[DocBlock]) -> list[DocBlock]:
    """Move `run` to the front of the page, keeping the columns well formed.

    A hoisted block may be carrying the mark that opens or closes a column -
    a CV's header is the first thing in its body column, so it holds the
    `column_start`. Lifting it out with the mark still on it would open the
    column outside the flex row and leave the rest of the column unwrapped,
    so each mark is handed to the first block that stays behind.
    """
    lifted = [b for b in blocks if b in run]
    rest = [b for b in blocks if b not in run]
    if not rest:
        return blocks

    for block in lifted:
        if block.column_start:
            for other in rest:
                if not other.column_start:
                    other.column_start = block.column_start
                    break
            block.column_start = ""
        if block.column_end:
            block.column_end = False
            rest[-1].column_end = True
    return lifted + rest


def attach_artwork(page: Page, blocks: list[DocBlock],
                   qa: QAReport) -> list[DocBlock]:
    """Give each block the artwork drawn for it.

    The readers above turn a page into blocks and, in doing so, lose the link
    back to the source blocks the artwork was matched against - a paragraph
    may be several source blocks joined, and a heading may be one of many. The
    link is rebuilt here by text: a rating's owner is the source block it sits
    under, so the emitted block carrying that block's words is the one that
    should carry its pips.

    Matching on text rather than on position is what lets this work for both
    readers, and for a page whose blocks were reordered by the vision reader.
    """
    from .artwork import read_artwork

    art = read_artwork(page)
    if not art:
        return blocks

    def find(owner: Optional[int]) -> Optional[DocBlock]:
        if owner is None:
            return None
        wanted = " ".join(page.blocks[owner].text.split())
        if not wanted:
            return None
        # The longest run of the owner's words that a block still contains.
        # A translated block is matched before translation runs, so its text
        # is still the source's.
        for block in blocks:
            text = " ".join(block.text.split())
            if text and (wanted in text or text in wanted):
                return block
        return None

    placed = 0
    for rating in art.ratings:
        target = find(rating.owner)
        if target is not None and target.rating is None:
            target.rating = rating
            placed += 1
    for frame in art.frames:
        target = find(frame.owner)
        if target is not None and target.frame is None:
            target.frame = frame
            placed += 1

    for banner in art.banners:
        # Matched by text, like everything else here: the buried-duplicate
        # pass runs before this and shifts every index after the ones it drops.
        covered = [b for b in (find(owner) for owner in banner.owners)
                   if b is not None]
        if not covered:
            continue
        # A picture inside the band belongs to it - a CV's portrait sits in
        # its header - so any image whose place on the source page falls in
        # the band is taken in too. Tested by position rather than by block
        # order: the portrait is emitted above the name, so it sits outside
        # the run the text alone would define.
        inside = [b for b in blocks
                  if b.kind == "image" and b not in covered
                  and b.source_box is not None
                  and _inside(b.source_box, banner.bbox)]
        if inside:
            # Only the largest: a header holds one portrait, and the contact
            # icons that also sit in the band belong with their own lines.
            covered.append(max(inside, key=lambda b: b.width * b.height))
        covered.sort(key=blocks.index)
        covered[0].banner_start = banner.fill
        covered[-1].banner_end = True
        placed += 1
        # A band that runs the width of the page is the page's masthead, not
        # decoration inside a column. Left where the reader put it - a CV's
        # header is read as the top of the body column - it can only be as
        # wide as that column, so the full-bleed band came back as an inset
        # box beside the sidebar. Hoisting the run it covers to the front of
        # the page puts it above the columns, which is where the source drew
        # it and the only place it can span both.
        if banner.bbox.width >= page.width * BANNER_PAGE_SPAN:
            blocks = _hoist(blocks, covered)

    if art.panels:
        # The panel backs whichever column it overlaps. Its own rectangle is
        # not reproduced: a fixed rectangle cannot grow with the translation,
        # and a tint that stops halfway down its column looks worse than none.
        for block in blocks:
            if block.column_start == "sidebar":
                block.tinted = art.panels[0].fill
                placed += 1
                break

    if placed:
        qa.add(
            "artwork",
            "info",
            f"Page {page.number + 1}: {placed} piece(s) of artwork - rating "
            f"rows, boxes, panels - were carried through with the text they "
            f"belong to.",
            page=page.number + 1,
            count=placed,
        )
    lost = len(art.loose)
    if lost:
        qa.add(
            "artwork",
            "info",
            f"Page {page.number + 1}: {lost} drawing(s) belong to no block - "
            f"a logo or a background - and were left out of the reflowed "
            f"page.",
            page=page.number + 1,
            count=lost,
        )
    return blocks


def _rgb(color) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"


def _open_banner(fill: tuple[int, int, int]) -> str:
    """The opening tag of a coloured band, with text set to suit its ground.

    The colour is decided here rather than taken from the spans: a banner's
    text is often stored dark and painted light by the band behind it, so
    keeping the stored colour makes it vanish on a dark ground.
    """
    luma = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
    ink = "#fff" if luma < 140 else "#111"
    return (f'<div class="banner" style="background:{_rgb(fill)};'
            f'color:{ink}">')


def _open_frame(frame) -> str:
    """The opening tag of a box drawn around a block."""
    fill = f";background:{_rgb(frame.fill)}" if frame.fill else ""
    return (f'<div class="frame" style="border:{max(frame.width, 0.6):.1f}pt '
            f'solid {_rgb(frame.color)}{fill}">')


def _render_rating(rating) -> str:
    """A row of pips as its own markup.

    Drawn as elements rather than as an image so it inherits the page's
    direction: under `dir=rtl` the filled pips lead from the right, which is
    how a score reads in Arabic.
    """
    pips = []
    for position in range(rating.total):
        on = position < rating.filled
        color = rating.color if on else (rating.empty_color or (222, 222, 222))
        pips.append(f'<i style="background:{_rgb(color)}"></i>')
    width = max(getattr(rating, "pip_width", rating.size), 2.0)
    height = max(getattr(rating, "pip_height", rating.size), 2.0)
    # A pip keeps the source's own proportions: a template that scores with
    # dashes is drawn with dashes, and only a square one is rounded to a dot.
    radius = "50%" if abs(width - height) <= 1.0 else "0"
    return (f'<div class="rating" style="--pip-w:{width:.1f}pt;'
            f'--pip-h:{height:.1f}pt;--pip-r:{radius}">'
            f'{"".join(pips)}</div>')


def _render_page(blocks: list[DocBlock], rtl: bool, base: float,
                 scale: float) -> str:
    out: list[str] = []
    in_list = ""          # "ul", "ol", or "" when no list is open
    markerless = _markerless_lists(blocks)

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append(f"</{in_list}>")
            in_list = ""

    # A page read as several columns wraps them in a flex row, so they are set
    # side by side. The columns' order in the markup is their reading order,
    # and `flex-direction` follows the page's own direction, so an Arabic page
    # puts the first column on the right without any of this knowing about it.
    columns_open = False
    banner_open = False
    banner_text_open = False

    for block in blocks:
        if block.column_start:
            close_list()
            if not columns_open:
                out.append('<div class="columns">')
                columns_open = True
            klass = f"column column-{block.column_start}"
            style = ""
            if block.tinted:
                klass += " tinted"
                style = f' style="background:{_rgb(block.tinted)}"'
            out.append(f'<div class="{klass}"{style}>')

        if block.banner_start and not banner_open:
            close_list()
            out.append(_open_banner(block.banner_start))
            banner_open = True
            banner_text_open = False

        # Inside a banner the picture is a direct child - it is the thing the
        # text sits beside - and everything else goes in one column next to it.
        if banner_open and block.kind != "image" and not banner_text_open:
            out.append('<div class="banner-text">')
            banner_text_open = True
        elif banner_open and block.kind == "image" and banner_text_open:
            out.append("</div>")
            banner_text_open = False

        # Where this block's own output begins, so a frame wraps exactly what
        # the block produced - taken after any column tag, which belongs to
        # the column and must stay outside the box.
        mark = len(out)

        # A run of entries the source set on their own lines is a list, even
        # though it carries no marker characters. Rendering it as one keeps
        # the entries visually separate instead of running them together.
        listish = block.kind == "bullet" or id(block) in markerless
        wanted = ("ol" if block.ordered else "ul") if listish else ""
        if in_list and wanted != in_list:
            # A run of a different kind ends the current list, so a numbered
            # sequence never continues into a bulleted one.
            close_list()

        if not listish:
            close_list()

        if block.kind == "table":
            out.append(_render_table(block, rtl, base, scale))
        elif block.kind == "rule":
            out.append('<hr class="section-rule">')
        elif block.kind == "heading":
            inner = _render_fragments(block.fragments, rtl, base, scale)
            level = max(1, min(block.level, 6))
            klass = ' class="centred"' if block.centred else ""
            out.append(f"<h{level}{klass}>{inner}</h{level}>")
        elif listish:
            if not in_list:
                # The browser numbers an <ol> itself, so the source marker is
                # dropped rather than written back into the text - it would
                # otherwise be numbered twice.
                out.append(f"<{wanted}>")
                in_list = wanted
            out.append(f"<li>{_render_fragments(block.fragments, rtl, base, scale)}</li>")
        elif block.kind == "two_sided":
            lead = _render_fragments(block.fragments, rtl, base, scale)
            cells = [f'<span class="lead">{lead}</span>']
            for extra in block.extra:
                cells.append(
                    f'<span class="cell">'
                    f'{_render_fragments(extra, rtl, base, scale)}</span>')
            trail = _render_fragments(block.right, rtl, base, scale)
            cells.append(f'<span class="trail">{trail}</span>')
            if block.caption:
                klass = "two-sided caption"
            elif block.extra:
                klass = "two-sided row-wrap"
            else:
                klass = "two-sided pair"
            out.append(f'<div class="{klass}">{"".join(cells)}</div>')
        elif block.kind == "image" and block.image:
            blob = base64.b64encode(block.image).decode("ascii")
            style = f"width:{block.width:.0f}pt" if block.width else ""
            klass = ' class="float-start"' if block.float_side else ""
            out.append(f'<img{klass} src="data:image/{block.image_ext};'
                       f'base64,{blob}" style="{style}" alt="">')
        else:
            inner = _render_fragments(block.fragments, rtl, base, scale)
            if inner.strip():
                klass = ' class="centred"' if block.centred else ""
                out.append(f"<p{klass}>{inner}</p>")

        # Artwork the block owns is emitted with it, so it travels wherever
        # the reflow puts the block instead of staying at a coordinate the
        # text has since left.
        if block.frame is not None and len(out) > mark:
            # The frame wraps everything this block produced. Inserting the
            # opening tag at the mark rather than appending a wrapper keeps a
            # list or a two-sided row intact inside its box.
            out.insert(mark, _open_frame(block.frame))
            out.append("</div>")
        if block.rating is not None:
            out.append(_render_rating(block.rating))

        if block.banner_end and banner_open:
            close_list()
            if banner_text_open:
                out.append("</div>")
                banner_text_open = False
            out.append("</div>")
            banner_open = False

        if block.column_end and columns_open:
            close_list()
            out.append("</div>")

    close_list()
    if banner_open:
        if banner_text_open:
            out.append("</div>")
        out.append("</div>")
    if columns_open:
        out.append("</div>")
    return "\n".join(out)


# --------------------------------------------------------------------------
# 3. rendering
# --------------------------------------------------------------------------

def render_html_to_pdf(html_text: str, output_path: str,
                       page_size: BBox) -> str:
    """Render `html_text` with headless Chromium.

    Chromium is used rather than a lighter engine because its text stack
    shapes Arabic and applies the bidi algorithm correctly and without
    configuration - the two things this pipeline exists to delegate.
    """
    from playwright.sync_api import sync_playwright

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(html_text)
        html_path = fh.name

    try:
        with sync_playwright() as api:
            browser = api.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(f"file://{html_path}")
                page.pdf(
                    path=output_path,
                    print_background=True,
                    width=f"{page_size.width / 72.0:.2f}in",
                    height=f"{page_size.height / 72.0:.2f}in",
                    # The margins come from the stylesheet's `@page` rule,
                    # which is the box that repeats on every printed page.
                    prefer_css_page_size=True,
                )
            finally:
                browser.close()
    finally:
        try:
            os.unlink(html_path)
        except OSError:
            pass
    return output_path


def is_available() -> bool:
    """Whether a browser is installed for this pipeline to use."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False
    try:
        with sync_playwright() as api:
            browser = api.chromium.launch()
            browser.close()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# 4. the pipeline
# --------------------------------------------------------------------------

def _translate_blocks(pages: list[list[DocBlock]], direction: str,
                      qa: QAReport) -> None:
    """Translate every fragment, in place.

    Fragments are sent as whole elements - a list item, a paragraph, a heading -
    rather than as the arbitrary runs a PDF happens to be cut into, which gives
    the translator coherent units to work with.
    """
    from . import translate as translate_mod
    from .language import should_translate

    targets: list[Fragment] = []
    # Table cells are plain strings rather than fragments, so their positions
    # are recorded and written back after the batch returns.
    cells: list[tuple[DocBlock, int, int]] = []

    for blocks in pages:
        for block in blocks:
            row_middle = [f for cell in block.extra for f in cell]
            for fragment in (list(block.fragments) + row_middle
                             + list(block.right)):
                if should_translate(fragment.text, direction):
                    targets.append(fragment)
            for r, row in enumerate(block.rows):
                for c, cell in enumerate(row):
                    if should_translate(cell, direction):
                        cells.append((block, r, c))

    payload = [f.text for f in targets] + \
              [block.rows[r][c] for block, r, c in cells]
    if not payload:
        return
    results = translate_mod.translate_batch(
        payload, direction, on_error=qa.translation_failure)

    for fragment, translated in zip(targets, results[:len(targets)]):
        fragment.translated = translated
    for (block, r, c), translated in zip(cells, results[len(targets):]):
        block.rows[r][c] = translated


# How far the page may be tightened to hold the source's length, and in what
# steps. Below this the rebuild is smaller than the source in a way a reader
# would notice, which is worse than an extra sheet.
FIT_MIN = 0.86
FIT_STEP = 0.04


def _fit_to_source_pages(pages, direction: str, page_size: BBox, qa: QAReport,
                         output_path: str, source_pages: int) -> None:
    """Render, and tighten the page if the rebuild spilled past the source.

    A translation that grows needs the room, and this does not try to deny it
    one: the reflow is what stops a grown paragraph overlapping the text below
    it. What it corrects is the *other* reason a page count grows - that the
    rebuild sets the document a little larger and looser than its source, so a
    page that was already full spills a few lines onto a sheet of their own.

    The coordinate path has always done this per box, shrinking a string until
    it fits. Here it is done once for the document, so the type stays even.
    Each attempt is a whole render, so the steps are coarse and the floor is
    close: this is for the page that just overran, not for cramming a document
    onto half its length.
    """
    fit = 1.0
    best_pages = None
    while True:
        html_text = build_html(pages, direction, page_size, qa, fit=fit)
        render_html_to_pdf(html_text, output_path, page_size)
        try:
            with fitz.open(output_path) as pdf:
                produced = pdf.page_count
        except Exception:
            return
        if best_pages is None:
            best_pages = produced
        if produced <= source_pages or fit <= FIT_MIN:
            break
        fit = round(fit - FIT_STEP, 4)

    if fit < 1.0 and produced <= source_pages:
        qa.add(
            "layout_review",
            "info",
            f"The rebuilt text was set {round((1 - fit) * 100)}% tighter so "
            f"the document still fits its original {source_pages} page(s); "
            f"without it the translation ran to {best_pages}.",
            source_pages=source_pages,
            fit=fit,
        )


def _check_pagination(source_pages: int, output_path: str,
                      qa: QAReport) -> None:
    """Report when the translation changed the document's length."""
    try:
        with fitz.open(output_path) as pdf:
            produced = pdf.page_count
    except Exception:
        return
    if produced > source_pages:
        qa.add(
            "layout_review",
            "info",
            f"The translation is longer than the original: {source_pages} "
            f"page(s) became {produced}. The text reflowed onto the extra "
            f"page(s) rather than being crushed or clipped.",
            source_pages=source_pages,
            output_pages=produced,
        )
    elif produced < source_pages:
        qa.add(
            "layout_review",
            "warning",
            f"The translated document is shorter than the original "
            f"({source_pages} page(s) became {produced}); check nothing was "
            f"dropped.",
            source_pages=source_pages,
            output_pages=produced,
        )


# A page carrying this much vector artwork is a designed page - a flyer, a
# poster - whose panels, logos and icons *are* the document. Re-flowing it as
# semantic HTML throws that away.
ARTWORK_PER_PAGE = 40
# Designed pages are also sparse: a page of running text with some decoration
# still belongs in the reflowing pipeline.
DENSE_TEXT_CHARS = 1200


def suits_html_pipeline(doc: Document) -> bool:
    """Whether re-flowing this document is better than redrawing it.

    Reflow is the right answer for a text document: the coordinate path cannot
    push content down when a translation grows, so a dense page accumulates
    overlap. It is the wrong answer for a designed page, where filled panels,
    logos and icons carry the meaning and the text is sparse - the HTML pass
    keeps the words and discards the design.

    The two are told apart by how much vector artwork a page carries against
    how much text. Nothing here is about the file format; a PDF can be either.

    A table overrides that judgement. A grid is the one structure the
    coordinate path cannot keep: its cells are sized for the source words, and
    a translation that grows has nowhere to go but across the ruling line into
    the cell beside it. A letterhead logo is easily enough filled artwork to
    read as a "designed page", so a plain one-page schedule or invoice was
    being redrawn instead of re-flowed, and its table came apart. Reflow keeps
    the grid, so any document with a real table takes this path.
    """
    if not doc.pages:
        return False
    if has_tables(doc):
        return True
    artwork = sum(
        len([d for d in page.drawings if d.fill and d.kind == "path"])
        for page in doc.pages
    )
    chars = sum(len(block.text) for page in doc.pages for block in page.blocks)
    per_page = artwork / max(len(doc.pages), 1)
    if per_page >= ARTWORK_PER_PAGE and chars < DENSE_TEXT_CHARS:
        return False
    return True


def has_tables(doc: Document) -> bool:
    """Whether any page of the source carries a real grid.

    Detection needs the source page's ruling lines, which the extracted model
    does not carry, so the file is reopened here. A file that cannot be
    reopened simply reports no tables and the ordinary heuristics decide.
    """
    try:
        with fitz.open(doc.source_path) as source:
            for page in doc.pages:
                if page.number >= len(source):
                    continue
                if _find_tables(page, source[page.number]):
                    return True
    except Exception:
        return False
    return False


def run_html_pipeline(doc: Document, output_path: str, direction: str,
                      qa: QAReport, vision: bool = False) -> str:
    """Extract structure, translate, and render through a browser.

    With `vision` set, each page's structure is read by a vision model rather
    than inferred from its geometry - see `vision_structure`. The model is
    asked only where the text goes, never what it says, and a page it cannot
    read falls back to the geometric reader page by page.
    """
    if not doc.pages:
        raise ValueError("the document has no pages")

    read = extract_structure
    if vision:
        from . import vision_layout
        from .vision_structure import read_structure

        if vision_layout.is_available():
            read = read_structure
        else:
            qa.add(
                "structure",
                "warning",
                "The layout model is not configured (ANTHROPIC_API_KEY is "
                "unset), so each page's structure was inferred from its "
                "geometry instead.",
            )

    # The source page is handed through so table detection can use its ruling
    # lines, which the extracted model does not carry.
    #
    # Reading a page's structure is a best-effort analysis of someone else's
    # file, and one page that defeats it must not cost the reader the whole
    # document. A page that raises is reported and skipped rather than
    # aborting the job, which is what an unusual layout used to do.
    with fitz.open(doc.source_path) as source:
        pages = []
        for page in doc.pages:
            try:
                _drop_buried_duplicates(page, qa)
                _consume_pip_images(page)
                blocks = read(page, qa, source[page.number])
                # Before translation, so the artwork is matched against the
                # source's own words rather than against a translation.
                blocks = attach_artwork(page, blocks, qa)
                pages.append(blocks)
            except Exception as exc:
                qa.add(
                    "structure",
                    "warning",
                    f"Page {page.number + 1} could not be read as a structured "
                    f"document and was left out of the rebuilt file - check "
                    f"this page against the original.",
                    page=page.number + 1,
                    error=f"{type(exc).__name__}: {exc}",
                )
                pages.append([])
    if not any(pages):
        raise ValueError(
            "no page of this document could be read as structured content")
    _translate_blocks(pages, direction, qa)

    first = doc.pages[0]
    page_size = BBox(0, 0, first.width, first.height)
    _fit_to_source_pages(pages, direction, page_size, qa, output_path,
                         len(doc.pages))
    _check_pagination(len(doc.pages), output_path, qa)

    qa.add(
        "summary",
        "info",
        "Rebuilt through the reflowing HTML pipeline, so translated text that "
        "grew pushed the content below it down instead of overlapping it.",
        engine="vision" if read is not extract_structure else "html",
        direction=direction,
    )
    return output_path

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
    image: Optional[bytes] = None
    image_ext: str = "png"
    centred: bool = False
    ordered: bool = False           # numbered list item
    standalone: bool = False        # an entry the source set on its own line
    rows: list[list[str]] = field(default_factory=list)   # table only
    header: bool = False            # table's first row is a header
    width: float = 0.0
    height: float = 0.0

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
        box = BBox(*table.bbox)
        # A header row is one whose cells are all short labels.
        header = all(len(cell) <= 40 for cell in rows[0])
        found.append((box, rows, header))
    return found


def _is_centred(line, text_left: float, text_right: float) -> bool:
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

    # Tables are lifted out whole. Their cell text and their ruling lines are
    # then withheld from everything below, so a grid is not also emitted as a
    # run of paragraphs with stray rules between them.
    tables = _find_tables(page, source) if source is not None else []
    if tables:
        lines = [(b, l) for b, l in lines
                 if not any(_inside(l.bbox, box) for box, _, _ in tables)]
        if not lines and not tables:
            return []

    text_left = min(l.bbox.x0 for _, l in lines)
    text_right = max(l.bbox.x1 for _, l in lines)

    rules = [r for r in _section_rules(page, text_left, text_right)
             if not any(box.y0 - 2 <= r[0] <= box.y1 + 2 for box, _, _ in tables)]

    # A producer often sets "Job Title .... Date" as two separate lines that
    # share a baseline: the title flush left, the trailing half flush right.
    # PyMuPDF reports them as two lines, so they are paired here before
    # anything else looks at them - left alone they become two paragraphs and
    # the trailing half wraps into a narrow column of its own.
    column = max(text_right - text_left, 1.0)
    trailing_for: dict[int, Any] = {}      # id(line) -> the line to its right
    consumed: set[int] = set()
    for i, (_, first) in enumerate(lines):
        if id(first) in consumed:
            continue
        for j in range(i + 1, len(lines)):
            second = lines[j][1]
            if id(second) in consumed:
                continue
            if abs(second.bbox.y0 - first.bbox.y0) > SAME_ROW_TOLERANCE:
                continue
            if second.bbox.x0 <= first.bbox.x1:
                continue
            right_spans = [sp for sp in second.spans if sp.text.strip()]
            if not right_spans:
                continue
            if (right_spans[-1].bbox.x1 - text_left) / column < RIGHT_EDGE_SHARE:
                continue
            trailing_for[id(first)] = second
            consumed.add(id(second))
            break

    lines = [pair for pair in lines if id(pair[1]) not in consumed]

    heading_sizes: set[float] = set()
    classified: list[tuple[str, dict, Any]] = []
    for owner, line in lines:
        kind, meta = _classify(line, body, page.width, text_left, text_right)
        if kind == "heading":
            heading_sizes.add(round(meta.get("size", body), 1))
        meta["centred"] = _is_centred(line, text_left, text_right)
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

    def emit_tables_above(y: float) -> None:
        nonlocal table_at
        while table_at < len(pending_tables) and pending_tables[table_at][0].y0 <= y:
            flush_paragraph()
            box, rows, header = pending_tables[table_at]
            blocks.append(DocBlock(kind="table", rows=rows, header=header))
            table_at += 1

    def emit_rules_above(y: float) -> None:
        """Place any section rule that sits above this line, in order."""
        nonlocal rule_at
        while rule_at < len(rules) and rules[rule_at][0] <= y:
            flush_paragraph()
            blocks.append(DocBlock(kind="rule"))
            rule_at += 1

    for index, (kind, meta, line) in enumerate(classified):
        emit_rules_above(meta.get("y", 0.0))
        emit_tables_above(meta.get("y", 0.0))
        spans = [s for s in line.spans if s.text]
        trailing = trailing_for.get(id(line))

        if kind == "heading" and trailing is None:
            flush_paragraph()
            level = min(ranks.get(round(meta.get("size", body), 1), 2), 6)
            blocks.append(DocBlock(kind="heading", fragments=_fragments(spans),
                                   level=level,
                                   centred=bool(meta.get("centred"))))
        elif trailing is not None:
            # Paired with the line sharing its baseline.
            flush_paragraph()
            blocks.append(DocBlock(
                kind="two_sided",
                fragments=_fragments(spans),
                right=_fragments([sp for sp in trailing.spans if sp.text]),
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
                                       centred=centred, standalone=True))
                continue
            if pending and not pending[-1].text.endswith(" "):
                pending[-1].text += " "
            pending.extend(_fragments(spans))
    flush_paragraph()
    emit_rules_above(float("inf"))
    emit_tables_above(float("inf"))

    for image in page.images:
        if not image.data:
            continue
        blocks.append(DocBlock(kind="image", image=image.data,
                               image_ext=image.ext or "png",
                               width=image.bbox.width,
                               height=image.bbox.height))

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


def build_html(pages: list[list[DocBlock]], direction: str, page_size: BBox,
               qa: QAReport) -> str:
    """Turn the extracted structure into a standalone HTML document."""
    rtl = direction == "en2ar"
    base = 11.0
    # Arabic reads smaller than Latin at the same point size, so it is set a
    # little larger; going the other way it is set smaller. Same rule the
    # coordinate pipeline applies per span, expressed once in CSS.
    scale = ((base + fontlib.ARABIC_SIZE_BONUS) / base if rtl
             else fontlib.AR_TO_EN_SIZE_SCALE)

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
    body_parts = [_render_page(blocks, rtl, base, scale) for blocks in pages]

    return f"""<!doctype html>
<html lang="{'ar' if rtl else 'en'}" dir="{'rtl' if rtl else 'ltr'}">
<head>
<meta charset="utf-8">
<style>
{faces}
@page {{ size: {width_in:.2f}in {height_in:.2f}in; margin: 0.75in; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  font-family: 'DocArabic', 'DocLatin', serif;
  font-size: {base * scale:.1f}pt;
  /* Real line boxes, computed by the layout engine, rather than a
     hand-rolled leading value. */
  line-height: 1.5;
  color: #000;
}}
h1, h2, h3, h4, h5, h6 {{ margin: 0.7em 0 0.3em; line-height: 1.3; }}
h1 {{ font-size: {base * scale * 1.45:.1f}pt; }}
h2 {{ font-size: {base * scale * 1.18:.1f}pt; }}
h3 {{ font-size: {base * scale * 1.05:.1f}pt; }}
p {{ margin: 0 0 0.55em; }}
ul, ol {{ margin: 0 0 0.6em; padding-inline-start: 1.6em; }}
li {{ margin: 0 0 0.25em; }}
/* One rule mirrors every "Job Title .... Date" row: under dir=rtl the two
   ends swap without any per-row logic. */
.two-sided {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1.5em;
  margin: 0 0 0.35em;
}}
.two-sided .lead {{ flex: 1 1 auto; }}
.two-sided .trail {{ flex: 0 0 auto; white-space: nowrap; }}
img {{ max-width: 100%; height: auto; display: block; margin: 0.6em 0; }}
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 0.5em 0 0.8em;
  break-inside: avoid;
}}
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


def _render_page(blocks: list[DocBlock], rtl: bool, base: float,
                 scale: float) -> str:
    out: list[str] = []
    in_list = ""          # "ul", "ol", or "" when no list is open

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append(f"</{in_list}>")
            in_list = ""

    for block in blocks:
        # A run of entries the source set on their own lines is a list, even
        # though it carries no marker characters. Rendering it as one keeps
        # the entries visually separate instead of running them together.
        listish = block.kind == "bullet" or (
            block.kind == "paragraph" and block.standalone)
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
            trail = _render_fragments(block.right, rtl, base, scale)
            out.append(
                f'<div class="two-sided"><span class="lead">{lead}</span>'
                f'<span class="trail">{trail}</span></div>'
            )
        elif block.kind == "image" and block.image:
            blob = base64.b64encode(block.image).decode("ascii")
            style = f"width:{block.width:.0f}pt" if block.width else ""
            out.append(f'<img src="data:image/{block.image_ext};base64,{blob}"'
                       f' style="{style}" alt="">')
        else:
            inner = _render_fragments(block.fragments, rtl, base, scale)
            if inner.strip():
                klass = ' class="centred"' if block.centred else ""
                out.append(f"<p{klass}>{inner}</p>")

    close_list()
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
                    margin={"top": "0.75in", "bottom": "0.75in",
                            "left": "0.75in", "right": "0.75in"},
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
            for fragment in list(block.fragments) + list(block.right):
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
    """
    if not doc.pages:
        return False
    artwork = sum(
        len([d for d in page.drawings if d.fill and d.kind == "path"])
        for page in doc.pages
    )
    chars = sum(len(block.text) for page in doc.pages for block in page.blocks)
    per_page = artwork / max(len(doc.pages), 1)
    if per_page >= ARTWORK_PER_PAGE and chars < DENSE_TEXT_CHARS:
        return False
    return True


def run_html_pipeline(doc: Document, output_path: str, direction: str,
                      qa: QAReport) -> str:
    """Extract structure, translate, and render through a browser."""
    if not doc.pages:
        raise ValueError("the document has no pages")

    # The source page is handed through so table detection can use its ruling
    # lines, which the extracted model does not carry.
    with fitz.open(doc.source_path) as source:
        pages = [extract_structure(page, qa, source[page.number])
                 for page in doc.pages]
    _translate_blocks(pages, direction, qa)

    first = doc.pages[0]
    page_size = BBox(0, 0, first.width, first.height)
    html_text = build_html(pages, direction, page_size, qa)
    render_html_to_pdf(html_text, output_path, page_size)
    _check_pagination(len(doc.pages), output_path, qa)

    qa.add(
        "summary",
        "info",
        "Rebuilt through the reflowing HTML pipeline, so translated text that "
        "grew pushed the content below it down instead of overlapping it.",
        engine="html",
        direction=direction,
    )
    return output_path

"""Shared data structures for the extraction -> translate -> rebuild pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

Direction = Literal["en2ar", "ar2en"]


@dataclass
class BBox:
    """Axis-aligned bounding box in PDF points, origin top-left."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_tuple(cls, t) -> "BBox":
        return cls(float(t[0]), float(t[1]), float(t[2]), float(t[3]))

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.x1 <= other.x0
            or other.x1 <= self.x0
            or self.y1 <= other.y0
            or other.y1 <= self.y0
        )


@dataclass
class Span:
    """A run of text sharing one set of style attributes."""

    text: str
    font: str
    size: float
    color: tuple[int, int, int]  # 0-255 RGB
    bold: bool
    italic: bool
    underline: bool
    bbox: BBox
    origin: tuple[float, float] = (0.0, 0.0)  # baseline origin from PyMuPDF
    translated: Optional[str] = None
    # Whether this span belongs to the right-to-left flow and so has its
    # coordinates reflected when the page is mirrored. Set from the span's own
    # script at extraction time; see language.is_arabic.
    mirror: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox"] = self.bbox.as_tuple()
        return d


@dataclass
class Line:
    spans: list[Span] = field(default_factory=list)
    bbox: Optional[BBox] = None

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)


@dataclass
class TextBlock:
    """A paragraph-ish grouping of lines, the unit we translate."""

    lines: list[Line] = field(default_factory=list)
    bbox: Optional[BBox] = None
    block_no: int = 0
    translated: Optional[str] = None
    # Whether the block as a whole is part of the right-to-left flow. A block
    # of English spans inside a mirrored Arabic page is re-anchored relative to
    # the mirrored layout rather than flipped; see mirror.mirror_page.
    mirror: bool = True

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines).strip()

    @property
    def spans(self) -> list[Span]:
        return [s for l in self.lines for s in l.spans]

    def dominant_style(self) -> Span:
        """Style of the span with the most characters - used when redrawing
        the whole block as a single translated string."""
        spans = [s for s in self.spans if s.text.strip()]
        if not spans:
            return Span("", "helv", 11.0, (0, 0, 0), False, False, False,
                        self.bbox or BBox(0, 0, 0, 0))
        return max(spans, key=lambda s: len(s.text))


@dataclass
class ImageElement:
    bbox: BBox
    data: bytes = b""
    ext: str = "png"
    xref: int = 0
    directional: bool = False  # arrows/flowcharts: candidate for content flip


@dataclass
class DrawingElement:
    """A vector path from the source page.

    `items` holds the original PyMuPDF path segments ("l" lines, "c" beziers,
    "re" rects, "qu" quads) so artwork - logos, icons - can be replayed exactly
    rather than approximated by its bounding box. `kind` only classifies the
    simple cases the layout logic cares about (rules and underlines).
    """

    bbox: BBox
    kind: str = "line"  # line | rect | path | underline
    color: Optional[tuple[int, int, int]] = (0, 0, 0)
    width: float = 0.5
    fill: Optional[tuple[int, int, int]] = None
    underline_for: Optional[int] = None  # index into page.blocks
    items: list = field(default_factory=list)   # raw path segments
    close_path: bool = False
    even_odd: bool = True
    fill_opacity: float = 1.0
    stroke_opacity: float = 1.0
    line_cap: int = 0
    line_join: float = 0.0
    dashes: Optional[str] = None
    shift_x: float = 0.0   # horizontal shift applied by mirroring
    seqno: int = 0         # drawing order in the source content stream


@dataclass
class Page:
    number: int
    width: float
    height: float
    blocks: list[TextBlock] = field(default_factory=list)
    images: list[ImageElement] = field(default_factory=list)
    drawings: list[DrawingElement] = field(default_factory=list)
    rotation: int = 0


@dataclass
class DocxRun:
    """Reference to a run inside a python-docx document, plus its style."""

    para_index: int
    run_index: int
    text: str
    bold: Optional[bool]
    italic: Optional[bool]
    underline: Optional[bool]
    font_name: Optional[str]
    font_size: Optional[float]  # points
    color: Optional[tuple[int, int, int]]
    location: str = "body"  # body | table | header | footer
    translated: Optional[str] = None


@dataclass
class DocxParagraph:
    index: int
    runs: list[DocxRun] = field(default_factory=list)
    style_name: str = "Normal"
    alignment: Optional[int] = None
    location: str = "body"
    translated: Optional[str] = None

    @property
    def text(self) -> str:
        return "".join(r.text for r in self.runs)


@dataclass
class Document:
    """Format-agnostic container handed between pipeline stages."""

    source_path: str
    kind: Literal["pdf", "docx"]
    pages: list[Page] = field(default_factory=list)          # pdf only
    paragraphs: list[DocxParagraph] = field(default_factory=list)  # docx only
    image_count: int = 0

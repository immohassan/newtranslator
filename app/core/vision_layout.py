"""Read a page's layout with a vision model instead of inferring it from geometry.

`html_pipeline.extract_structure` works out what a page *is* - heading, list,
sidebar, table - from the geometry alone: gaps, alignment, font sizes, ruling
lines. That reasoning is sound on the shapes it was written for and silently
wrong on the ones it was not. A CV with a ruled sidebar is the standard case:
nothing in the coordinates says "sidebar", so the two columns are read as one
run of text and comb together.

A model looking at the rendered page does not have that problem - it sees two
columns because they look like two columns.

The division of labour here matters, and is the whole reason this is safe:

    the model decides *structure*     - what is a heading, what is a column,
                                        which block follows which
    PyMuPDF decides *content*         - the exact characters, taken from the
                                        file itself

The model is never asked to transcribe. It is given the page image *and* the
already-extracted text blocks, each with an id, and it answers with ids. A
model that misreads a digit therefore cannot corrupt a phone number, because
it never writes one - the worst it can do is put a block in the wrong place,
which is visible and recoverable. Free-form transcription would trade a
layout bug for a content bug, which is a far worse trade in a translation
tool: a wrong date in a plausible-looking CV is not something a reader can
catch.

Blocks the model fails to place are appended in reading order rather than
dropped, so a partial answer degrades instead of losing text.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Optional

import fitz

from .models import Page
from .qa import QAReport

log = logging.getLogger(__name__)

ANTHROPIC_MODEL = "claude-opus-5"
OPENAI_MODEL = "gpt-4o"
# The page is rendered for the model at this resolution. 150 DPI is enough to
# read body text at 9pt - which is what the model needs in order to match a
# block to the text it was given - without making the image needlessly large.
RENDER_DPI = 150
# A page with fewer text blocks than this has no layout worth asking about;
# the geometric reader handles it, and an API call would be pure latency.
MIN_BLOCKS = 4
# Blocks are identified to the model by index. Anything past this is a page so
# dense that the reply would be mostly ids, and the geometric path is used.
MAX_BLOCKS = 300

_KINDS = ("heading", "paragraph", "bullet", "rule", "table_row")

# The reply shape. Structured output is used rather than prose-with-JSON so a
# malformed answer is a validation error here rather than a mis-parse later.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["columns"],
    "properties": {
        "columns": {
            "type": "array",
            "description": (
                "The page's columns, in the order they should be read. A page "
                "set as one column has exactly one entry."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["role", "blocks"],
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["main", "sidebar"],
                        "description": (
                            "'sidebar' for a narrow supporting column beside "
                            "the body; 'main' for the body itself."
                        ),
                    },
                    "blocks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            # Every property is listed: OpenAI's strict mode
                            # rejects a schema with optional keys, so a field
                            # that does not apply is sent as null instead.
                            "required": ["id", "kind", "level"],
                            "properties": {
                                "id": {
                                    "type": "integer",
                                    "description":
                                        "The id of the supplied text block.",
                                },
                                "kind": {
                                    "type": "string",
                                    "enum": list(_KINDS),
                                },
                                "level": {
                                    "type": ["integer", "null"],
                                    "description":
                                        "Heading rank, 1-4. Null unless the "
                                        "block is a heading.",
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

_SYSTEM = """\
You read the layout of a document page. You are given an image of the page and \
the text blocks that were extracted from it, each with an id.

Report the page's structure by referring to those ids. Never write out the text \
of a block, and never invent a block: the ids you are given are the only ones \
that exist, and the text belongs to the file, not to you.

Order matters. List the columns in the order a reader would read them, and the \
blocks within each column in the order they are read down that column.

A column is a track of text running down most of the page with a clear gutter \
beside it - a CV's sidebar next to its body, a newsletter's two halves. Report \
those as separate columns; never flatten them into one interleaved run.

Almost every page has one or two columns. Three is rare and four is nearly \
unheard of, so before reporting more than two, satisfy yourself that each one \
really is a full-height track of its own. These are NOT columns:
  - a date, a place or a label set to the right of a heading (that is part of \
the same block's row, so it belongs in the same column as the heading)
  - a short run of right-aligned text beside a paragraph
  - a header or footer spanning the page
  - the second half of a line that happens to start further right
When in doubt, use fewer columns: a wrongly split column is far worse than a \
missed one, because it squeezes the body text into a strip.

Classify each block:
  heading    a title introducing what follows
  paragraph  running text
  bullet     one entry of a list
  rule       a horizontal divider
  table_row  one row of a genuine grid, where the cells relate across the row

Include every id exactly once."""

# Stated in the prompt for a provider whose API cannot enforce the schema.
_SHAPE = """\
Reply with a JSON object of exactly this shape and no other fields:

{"columns": [
  {"role": "main" | "sidebar",
   "blocks": [{"id": <integer>,
               "kind": "heading"|"paragraph"|"bullet"|"rule"|"table_row",
               "level": <1-4 for a heading, otherwise null>}]}
]}"""


def _page_image(source: "fitz.Page") -> str:
    """The page rendered as a base64 PNG for the model to look at."""
    pixmap = source.get_pixmap(dpi=RENDER_DPI)
    return base64.standard_b64encode(pixmap.tobytes("png")).decode("ascii")


def _block_digest(page: Page) -> list[tuple[int, str]]:
    """The page's text blocks as (id, text), in the order they were extracted.

    The id is the block's index in `page.blocks`, so the model's answer maps
    straight back onto the extracted model with no lookup table to drift.
    """
    out: list[tuple[int, str]] = []
    for index, block in enumerate(page.blocks):
        text = " ".join(block.text.split())
        if text:
            out.append((index, text))
    return out


def _prompt(digest: list[tuple[int, str]], page: Page) -> str:
    lines = [f"Page size: {page.width:.0f} x {page.height:.0f} points.",
             "", "Text blocks:"]
    for index, text in digest:
        # The position is given because two blocks can read alike and be told
        # apart only by where they sit - "Jan 2021" in a sidebar and the same
        # date in the body.
        box = page.blocks[index].bbox
        lines.append(f"[{index}] (x={box.x0:.0f}, y={box.y0:.0f}) {text}")
    return "\n".join(lines)


class LayoutReader:
    """A vision model that can be asked where a page's text goes.

    Two backends implement this, and which one runs is decided the same way
    `translate.get_provider` decides: by whichever key is set. The reply shape
    is identical either way, so nothing downstream knows the difference.
    """

    name = ""

    def read(self, image_b64: str, prompt: str) -> str:
        """The model's raw JSON reply."""
        raise NotImplementedError


class AnthropicReader(LayoutReader):
    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(f"the anthropic package is not installed: {exc}")
        self.model = model or os.environ.get("VISION_LAYOUT_MODEL",
                                             ANTHROPIC_MODEL)
        # An identity-linked key must name the workspace it acts in, or the
        # API rejects the request. Ordinary keys ignore the header.
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        self._client = anthropic.Anthropic(
            api_key=key, timeout=180.0, max_retries=2,
            default_headers=({"anthropic-workspace-id": workspace}
                             if workspace else None))

    def read(self, image_b64: str, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png",
                                "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise _Refused()
        return "".join(b.text for b in response.content if b.type == "text").strip()


class OpenAIReader(LayoutReader):
    name = "openai"
    # See OpenRouterReader: only a host that reserves credit up front needs a
    # ceiling, and OpenAI is not one.
    MAX_TOKENS = None

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(f"the openai package is not installed: {exc}")
        self.model = model or os.environ.get("VISION_LAYOUT_MODEL", OPENAI_MODEL)
        self._client = OpenAI(api_key=key, timeout=180.0, max_retries=2)

    # The schema is enforced rather than merely requested, so a reply that
    # does not fit it is the API's error and not a parse failure here.
    # `strict` requires the schema be closed, which _SCHEMA is.
    RESPONSE_FORMAT = {
        "type": "json_schema",
        "json_schema": {"name": "page_layout", "strict": True,
                        "schema": _SCHEMA},
    }

    def read(self, image_b64: str, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            response_format=self.RESPONSE_FORMAT,
            temperature=0,
            **({"max_tokens": self.MAX_TOKENS} if self.MAX_TOKENS else {}),
        )
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise _Refused()
        return (choice.message.content or "").strip()


class OpenRouterReader(OpenAIReader):
    """A vision model hosted by OpenRouter - Claude included.

    OpenRouter speaks the OpenAI protocol, including image content parts and
    JSON-schema response formats, so only the endpoint, key and model differ.
    """

    name = "openrouter"
    MODEL = "anthropic/claude-opus-4.1"
    BASE_URL = "https://openrouter.ai/api/v1"
    # Plain JSON rather than a strict schema: `strict` is an OpenAI feature and
    # the models routed here do not all implement it - a Claude model rejects
    # the request outright. The reply is validated by `_repair` either way, so
    # nothing downstream depends on the API having enforced the shape.
    RESPONSE_FORMAT = {"type": "json_object"}
    # OpenRouter reserves credit for the whole of max_tokens before running
    # the request, so the model's own 32k ceiling is refused outright on a
    # small balance. A layout reply is a list of ids and kinds - a page of 300
    # blocks fits in a fraction of this.
    MAX_TOKENS = 8000

    def read(self, image_b64: str, prompt: str) -> str:
        # With no schema on the request, the shape has to be stated in the
        # prompt or the model invents its own field names.
        return super().read(image_b64, prompt + "\n\n" + _SHAPE)

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(f"the openai package is not installed: {exc}")
        self.model = model or os.environ.get("VISION_LAYOUT_MODEL", self.MODEL)
        from .translate import _openrouter_headers

        self._client = OpenAI(
            api_key=key,
            base_url=os.environ.get("OPENROUTER_BASE_URL", self.BASE_URL),
            timeout=180.0, max_retries=2,
            default_headers=_openrouter_headers(),
        )


class _Refused(Exception):
    """The model declined to read the page."""


_READERS = {"anthropic": AnthropicReader, "openai": OpenAIReader,
            "openrouter": OpenRouterReader}


def _reader() -> LayoutReader:
    """The configured reader, chosen the way the translator chooses its own."""
    name = os.environ.get("VISION_LAYOUT_PROVIDER", "").strip().lower()
    if not name:
        if os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        else:
            name = "openrouter"
    cls = _READERS.get(name)
    if cls is None:
        raise RuntimeError(f"unknown layout provider '{name}'")
    return cls()


def is_available() -> bool:
    """Whether the vision reader can run at all."""
    try:
        _reader()
    except Exception:
        return False
    return True


def provider_name() -> str:
    try:
        return _reader().name
    except Exception:
        return "none"


def suits_vision_layout(page: Page) -> bool:
    """Whether asking the model about this page is worth the call."""
    return MIN_BLOCKS <= len(page.blocks) <= MAX_BLOCKS


def read_layout(page: Page, source: "fitz.Page", qa: QAReport,
                client=None) -> Optional[list[dict]]:
    """The page's columns as the model reads them, or None if it could not.

    Returns a list of `{"role": str, "blocks": [{"id", "kind", "level"}]}`.
    Every returned id is a real index into `page.blocks`, appears once, and
    every block of the page appears somewhere - the reply is repaired against
    the extracted text before it is returned, so a caller never has to trust
    the model's bookkeeping.

    None means the page could not be read and the geometric path should be
    used. That is a fallback, not a failure: it is what happens with no API
    key, and it must stay a working route.
    """
    digest = _block_digest(page)
    if len(digest) < MIN_BLOCKS:
        return None

    try:
        reader = client or _reader()
        raw = reader.read(_page_image(source), _prompt(digest, page))
    except _Refused:
        qa.add(
            "structure",
            "warning",
            f"Page {page.number + 1}: the model declined to read this page, "
            f"so it was read from its geometry instead.",
            page=page.number + 1,
        )
        return None
    except Exception as exc:
        qa.add(
            "structure",
            "warning",
            f"Page {page.number + 1}: the layout could not be read by the "
            f"vision model, so the page was read from its geometry instead.",
            page=page.number + 1,
            error=f"{type(exc).__name__}: {exc}",
        )
        return None

    try:
        parsed = json.loads(raw)
        columns = parsed["columns"]
    except (ValueError, KeyError, TypeError) as exc:
        qa.add(
            "structure",
            "warning",
            f"Page {page.number + 1}: the layout reply could not be read, so "
            f"the page was read from its geometry instead.",
            page=page.number + 1,
            error=f"{type(exc).__name__}: {exc}",
        )
        return None

    return _repair(columns, {index for index, _ in digest}, page, qa)


def _repair(columns: Any, valid: set[int], page: Page,
            qa: QAReport) -> Optional[list[dict]]:
    """Make the model's answer safe to act on.

    Three things are checked, and each has a reason:

    - an id that is not a real block is dropped, because acting on it would
      raise deep inside the renderer;
    - an id given twice is kept once, because the second copy would print the
      same sentence twice;
    - a block the model never mentioned is appended in reading order, because
      dropping it would lose text from the document silently - the one failure
      a translation tool must never have.
    """
    if not isinstance(columns, list) or not columns:
        return None

    cleaned: list[dict] = []
    seen: set[int] = set()
    for column in columns:
        if not isinstance(column, dict):
            continue
        blocks = []
        for entry in column.get("blocks") or []:
            if not isinstance(entry, dict):
                continue
            index = entry.get("id")
            if not isinstance(index, int) or index not in valid or index in seen:
                continue
            seen.add(index)
            kind = entry.get("kind")
            blocks.append({
                "id": index,
                "kind": kind if kind in _KINDS else "paragraph",
                "level": entry.get("level") or 2,
            })
        if blocks:
            role = column.get("role")
            cleaned.append({"role": role if role in ("main", "sidebar") else "main",
                            "blocks": blocks})

    if not cleaned:
        return None

    cleaned = _merge_thin_columns(cleaned, page, qa)

    missed = sorted(valid - seen)
    if missed:
        # Appended to the column they sit closest to, so a block the model
        # forgot lands near its neighbours rather than at the end of the page.
        qa.add(
            "structure",
            "info",
            f"Page {page.number + 1}: {len(missed)} block(s) were not placed by "
            f"the layout reader and were added in reading order so none of the "
            f"page's text was lost.",
            page=page.number + 1,
            count=len(missed),
        )
        for index in missed:
            cleaned[_nearest_column(cleaned, index, page)]["blocks"].append(
                {"id": index, "kind": "paragraph", "level": 2})
        for column in cleaned:
            column["blocks"].sort(key=lambda b: page.blocks[b["id"]].bbox.y0)

    return cleaned


# A column must run down this share of the page's text to be a column at all.
# A run of right-aligned dates beside a heading does not, however much it
# looks like a track of its own.
COLUMN_MIN_HEIGHT_SHARE = 0.35
# ...and hold at least this many blocks. Two labels are not a column.
COLUMN_MIN_BLOCKS = 3


def _merge_thin_columns(columns: list[dict], page: Page,
                        qa: QAReport) -> list[dict]:
    """Fold anything too slight to be a column into the one beside it.

    The model over-segments in one particular way: a run of right-aligned
    labels - a place name beside each job title, a date beside each degree -
    reads as a narrow track and is reported as its own column. Believing it
    squeezes the body into a strip, which is a worse page than the one the
    split was meant to fix. A column has to earn the name by running down the
    page and carrying real content.
    """
    if len(columns) < 2:
        return columns

    def extent(column: dict) -> tuple[float, float]:
        boxes = [page.blocks[b["id"]].bbox for b in column["blocks"]]
        return min(b.y0 for b in boxes), max(b.y1 for b in boxes)

    top = min(extent(c)[0] for c in columns)
    bottom = max(extent(c)[1] for c in columns)
    height = max(bottom - top, 1.0)

    keep, thin = [], []
    for column in columns:
        low, high = extent(column)
        if ((high - low) >= height * COLUMN_MIN_HEIGHT_SHARE
                and len(column["blocks"]) >= COLUMN_MIN_BLOCKS):
            keep.append(column)
        else:
            thin.append(column)

    if not thin or not keep:
        return columns

    for column in thin:
        # Merged into the column it sits nearest, then re-sorted, so a label
        # lands beside the heading it belongs to rather than at the foot.
        target = keep[_nearest_column(keep, column["blocks"][0]["id"], page)]
        target["blocks"].extend(column["blocks"])
        target["blocks"].sort(key=lambda b: (page.blocks[b["id"]].bbox.y0,
                                             page.blocks[b["id"]].bbox.x0))

    qa.add(
        "structure",
        "info",
        f"Page {page.number + 1}: {len(thin)} narrow run(s) of text were read "
        f"as columns of their own and folded back into the column beside "
        f"them, which is where they belong.",
        page=page.number + 1,
        count=len(thin),
    )
    return keep


def _nearest_column(columns: list[dict], index: int, page: Page) -> int:
    """The column whose blocks sit closest, horizontally, to this one."""
    box = page.blocks[index].bbox
    centre = (box.x0 + box.x1) / 2
    best, best_gap = 0, float("inf")
    for position, column in enumerate(columns):
        xs = [page.blocks[b["id"]].bbox for b in column["blocks"]]
        if not xs:
            continue
        column_centre = sum((b.x0 + b.x1) / 2 for b in xs) / len(xs)
        gap = abs(column_centre - centre)
        if gap < best_gap:
            best, best_gap = position, gap
    return best

"""Arabic presentation shaping.

Raw Unicode Arabic renders as disconnected isolated letterforms in PDF and in
any renderer without a shaping engine. Two transforms fix that, in order:

  1. arabic_reshaper - picks the initial/medial/final/isolated glyph per letter.
  2. python-bidi     - reorders the string into visual order (UAX #9).

The result is a *display* string. It is not the text any more: it must never be
stored, translated, diffed or re-shaped. Shape once, immediately before drawing.
"""
from __future__ import annotations

import re
from functools import lru_cache

import arabic_reshaper
from bidi.algorithm import get_display

# Arabic, Arabic Supplement, Extended-A, Presentation Forms A/B
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_ARABIC_CHAR_RE = re.compile(
    "[" + "".join(f"\\u{lo:04x}-\\u{hi:04x}" for lo, hi in _ARABIC_RANGES) + "]"
)
# Everything above U+FB50 is already a presentation form, i.e. already shaped.
_PRESENTATION_RE = re.compile(r"[ﭐ-﷿ﹰ-﻿]")

_RESHAPER = arabic_reshaper.ArabicReshaper(
    configuration={
        "delete_harakat": False,       # keep diacritics; they carry meaning
        "support_ligatures": True,
        "shift_harakat_position": False,
    }
)


# PyMuPDF reports a space drawn with an Arabic face as U+00A0: the font maps
# the space glyph there. The character is therefore not something this pipeline
# writes - it arrives from extraction - but it must not survive into the text we
# translate, measure or wrap, because a no-break space is not a word boundary.
_NBSP = "\u00a0"
# Other fixed-width spaces PDF producers use for justification, plus the zero
# width no-break space (U+FEFF) that appears as a stray BOM mid-string.
_SPACE_LIKE = {
    "\u2007": " ",   # figure space
    "\u202f": " ",   # narrow no-break space
    "\u2060": "",    # word joiner
    "\ufeff": "",    # zero width no-break space / BOM
}


def normalize_spaces(text: str) -> str:
    """Turn no-break and other fixed-width spaces into ordinary spaces.

    Wrapping splits on U+0020, so an nbsp-joined line is a single unbreakable
    "word" that can never wrap - which is how a paragraph ends up overflowing
    its box and colliding with the block below it. Normalising here keeps the
    text itself correct as well: nbsp renders at an inconsistent width across
    faces, which is what makes justified Arabic look ragged.
    """
    if not text:
        return text
    out = text.replace(_NBSP, " ")
    for char, replacement in _SPACE_LIKE.items():
        if char in out:
            out = out.replace(char, replacement)
    return out


def normalize_block_text(text: str) -> str:
    """Normalise spacing and strip the ragged indent off every line.

    Source PDFs carry leading and trailing spaces inside their spans - a line
    set as " for immigrants from New" is common, because the producer used a
    space instead of moving the text cursor. Those spaces survive extraction and
    translation, and `insert_textbox` honours them, so the redrawn paragraph
    gets a few points of indent on some lines and none on others: a left edge
    that looks broken even though every line starts at the same box.

    Run-together internal spaces are collapsed for the same reason - two spaces
    between words after a merge render as a visible gap.
    """
    if not text:
        return text
    lines = [
        " ".join(normalize_spaces(line).split())
        for line in normalize_spaces(text).split("\n")
    ]
    return "\n".join(lines)


def contains_arabic(text: str) -> bool:
    return bool(_ARABIC_CHAR_RE.search(text or ""))


def is_already_shaped(text: str) -> bool:
    """True if the string already holds presentation forms - shaping twice
    corrupts the text, so callers guard with this."""
    return bool(_PRESENTATION_RE.search(text or ""))


@lru_cache(maxsize=4096)
def _shape_cached(text: str) -> str:
    reshaped = _RESHAPER.reshape(text)
    return get_display(reshaped)


def shape(text: str) -> str:
    """Return the visual-order, correctly joined form of `text`.

    Safe on non-Arabic input (returned unchanged) and idempotent-guarded against
    double shaping. Never raises: on failure the original text is returned so a
    document still builds, just with worse glyphs.
    """
    if not text or not contains_arabic(text):
        return text
    if is_already_shaped(text):
        return text
    try:
        return _shape_cached(text)
    except Exception:
        return text


def shape_for_render(text: str, direction: str) -> str:
    """Shape only when the output language is Arabic.

    `direction` is the *translation* direction, so en2ar produces Arabic output
    and ar2en produces English output that needs no shaping.
    """
    if direction == "en2ar":
        return shape(text)
    return text


def shape_lines(lines: list[str]) -> str:
    """Shape each visual line separately and join them back with newlines.

    The bidi algorithm reverses a string into visual order, so it must run on
    text that is *already* broken into final lines. Shaping a whole paragraph
    and letting the renderer wrap it afterwards puts the paragraph's tail on
    the first line and its head on the last.
    """
    return "\n".join(shape(line) for line in lines)


def logical_length(text: str) -> int:
    """Character count of the logical string, for overflow estimation."""
    return len(text or "")

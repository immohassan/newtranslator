"""Font resolution and glyph-coverage checks.

Arabic bold must come from a real bold font file. Synthetic bold (stroking the
glyph outline) smears the joins and connecting strokes of Naskh and looks wrong,
so it is never used here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import fitz

from .qa import QAReport

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "fonts")

# Arabic output faces. Bold is a separate file on purpose - see module docstring.
ARABIC_FONTS = {
    "regular": "NotoNaskhArabic-Regular.ttf",
    "bold": "NotoNaskhArabic-Bold.ttf",
    # Naskh has no italic; Arabic typography does not use slanting for emphasis.
    "italic": "NotoNaskhArabic-Regular.ttf",
    "bolditalic": "NotoNaskhArabic-Bold.ttf",
}

LATIN_FONTS = {
    "regular": "DejaVuSans.ttf",
    "bold": "DejaVuSans-Bold.ttf",
    "italic": "DejaVuSans-Oblique.ttf",
    "bolditalic": "DejaVuSans-BoldOblique.ttf",
}

# Arabic reads smaller than Latin at the same point size, so nudge it up.
ARABIC_SIZE_BONUS = 1.0

# The inverse of that bump, for ar2en. An Arabic block's point size was chosen
# for Arabic's visual density; reusing it verbatim for the English replacement
# renders the Latin text noticeably too large. Expressed as a scale rather than
# a flat subtraction so it stays proportional across heading and body sizes -
# taking 1pt off a 25pt heading is invisible, off 11pt body text it is not.
# Tune here after a visual review rather than at the call site.
AR_TO_EN_SIZE_SCALE = 0.87
# Never shrink a heading into body text: below this the result stops looking
# like the original document's hierarchy.
AR_TO_EN_MIN_SIZE = 6.0

SYSTEM_FALLBACK_DIRS = [
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/kacst",
    "/Library/Fonts",
    "C:\\Windows\\Fonts",
]


@dataclass
class ResolvedFont:
    path: str
    name: str          # PyMuPDF fontname handle
    is_arabic: bool
    substituted: bool  # True when the original document font was replaced


def style_key(bold: bool, italic: bool) -> str:
    if bold and italic:
        return "bolditalic"
    if bold:
        return "bold"
    if italic:
        return "italic"
    return "regular"


@lru_cache(maxsize=64)
def _locate(filename: str) -> Optional[str]:
    local = os.path.join(FONT_DIR, filename)
    if os.path.exists(local):
        return local
    for directory in SYSTEM_FALLBACK_DIRS:
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            return candidate
    return None


@lru_cache(maxsize=64)
def _font_has_arabic(path: str) -> bool:
    """A font covers Arabic if it can map a basic Arabic letter."""
    try:
        font = fitz.Font(fontfile=path)
        return font.has_glyph(0x0628)  # beh
    except Exception:
        return False


def font_supports_arabic_name(font_name: str) -> bool:
    """Cheap name-based guess for fonts embedded in the source PDF, which we
    cannot always load directly."""
    lowered = (font_name or "").lower()
    return any(
        k in lowered
        for k in ("arab", "naskh", "kufi", "amiri", "scheherazade", "dubai",
                  "cairo", "tajawal", "almarai", "kacst", "traditional")
    )


def covers(text: str, path: str) -> bool:
    """Whether every non-space character in `text` has a glyph in this font."""
    try:
        font = load_font(path)
    except Exception:
        return False
    return all(font.has_glyph(ord(ch)) for ch in text if not ch.isspace())


def resolve_for_text(
    text: str,
    *,
    bold: bool,
    italic: bool,
    original_font: str = "",
    qa: Optional[QAReport] = None,
) -> "ResolvedFont":
    """Pick a font that can render every character in `text`.

    Noto Naskh carries no Latin glyphs, so a mixed Arabic/Latin string drawn in
    it loses its Latin half to blank boxes. DejaVu covers both scripts and is
    used whenever one face has to serve mixed content.
    """
    from .shape_arabic import contains_arabic

    has_arabic = contains_arabic(text)
    arabic_choice = resolve(target_is_arabic=has_arabic, bold=bold, italic=italic,
                            original_font=original_font, qa=None)
    if covers(text, arabic_choice.path):
        if qa is not None and original_font:
            resolve(target_is_arabic=has_arabic, bold=bold, italic=italic,
                    original_font=original_font, qa=qa)
        return arabic_choice

    # Fall back to the face that covers both scripts.
    key = style_key(bold, italic)
    path = _locate(LATIN_FONTS[key])
    if path and covers(text, path):
        if qa is not None:
            qa.font_substitution(
                os.path.splitext(os.path.basename(arabic_choice.path))[0],
                os.path.splitext(os.path.basename(path))[0],
                "the text mixes scripts and needs a font covering both",
            )
        return ResolvedFont(
            path=path,
            name=os.path.splitext(os.path.basename(path))[0],
            is_arabic=has_arabic,
            substituted=True,
        )
    return arabic_choice


def resolve(
    *,
    target_is_arabic: bool,
    bold: bool,
    italic: bool,
    original_font: str = "",
    qa: Optional[QAReport] = None,
) -> ResolvedFont:
    """Pick a font file for a span, logging any substitution."""
    key = style_key(bold, italic)
    table = ARABIC_FONTS if target_is_arabic else LATIN_FONTS
    filename = table[key]
    path = _locate(filename)

    if path is None:
        # Last resort: any file in the table that does exist.
        for alt in table.values():
            path = _locate(alt)
            if path:
                break

    if path is None:
        raise RuntimeError(
            f"No usable font file found for {'Arabic' if target_is_arabic else 'Latin'} "
            f"output. Expected {filename} in {FONT_DIR}."
        )

    substituted = False
    if original_font:
        base = original_font.split("+")[-1]
        wanted = os.path.splitext(os.path.basename(path))[0]
        if base.lower() not in wanted.lower():
            substituted = True
            if qa is not None and target_is_arabic and not font_supports_arabic_name(base):
                qa.font_substitution(
                    base,
                    wanted,
                    "the original font has no Arabic glyph coverage",
                )
            elif qa is not None:
                qa.font_substitution(
                    base, wanted, "the original font file is not embedded for reuse"
                )

    return ResolvedFont(
        path=path,
        name=os.path.splitext(os.path.basename(path))[0],
        is_arabic=target_is_arabic,
        substituted=substituted,
    )


@lru_cache(maxsize=32)
def load_font(path: str) -> fitz.Font:
    return fitz.Font(fontfile=path)


def line_height(path: str, size: float) -> float:
    """Vertical space one line needs, from the font's own ascender/descender.

    Naskh faces carry much taller metrics than Latin ones - roughly 2.4x the
    point size against 1.4x - so a box sized for English text is too short for
    the Arabic that replaces it.
    """
    try:
        font = load_font(path)
        return (font.ascender - font.descender) * size * 1.15
    except Exception:
        return size * 2.5


def measure(text: str, path: str, size: float) -> float:
    """Width of `text` in points at `size`, using real font metrics."""
    try:
        return load_font(path).text_length(text, fontsize=size)
    except Exception:
        return len(text) * size * 0.5


def adjusted_size(original: float, target_is_arabic: bool,
                  source_is_arabic: bool = False) -> float:
    """Size the output text for the script it is actually being drawn in.

    The two directions are not symmetrical by accident - each corrects for the
    same thing from opposite sides. Latin sized for Latin needs a bump to read
    as Arabic (`ARABIC_SIZE_BONUS`); Arabic sized for Arabic needs a reduction
    to read as Latin (`AR_TO_EN_SIZE_SCALE`).

    `source_is_arabic` distinguishes English that was *translated from* Arabic -
    and so carries an Arabic-tuned size that must be scaled down - from English
    that was always English and should keep the size its author chose.

    Whatever comes back is a starting point only: the caller still runs it
    through the overflow-fit loop, which shrinks further if the particular
    string does not fit its box.
    """
    if target_is_arabic:
        return original + ARABIC_SIZE_BONUS
    if source_is_arabic:
        return max(original * AR_TO_EN_SIZE_SCALE, AR_TO_EN_MIN_SIZE)
    return original


def docx_font_name(target_is_arabic: bool) -> str:
    return "Noto Naskh Arabic" if target_is_arabic else "Calibri"

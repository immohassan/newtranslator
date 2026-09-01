"""Script detection, used to leave already-translated content alone.

A real document mixes languages: an Arabic flyer carries an English
organisation name, an English report quotes an Arabic term. Sending that text
through the translator corrupts it - "Mayor's Office of Immigrant Affairs"
comes back as a garbled re-translation - and re-aligning it flips a Latin
phrase to the wrong side of the page.

So content already in the target language is passed through untouched: not
translated, not reshaped, and not re-aligned.
"""
from __future__ import annotations

import re
from typing import Optional

from .shape_arabic import _ARABIC_CHAR_RE

_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
# Characters that belong to neither script and so cast no vote.
_NEUTRAL_RE = re.compile(r"[\s\d\W_]", re.UNICODE)

# A string counts as "in a script" once that script holds this share of its
# letters. Below it the text is mixed and gets translated normally.
DOMINANCE = 0.6


def script_counts(text: str) -> tuple[int, int]:
    """(arabic_letters, latin_letters) in `text`."""
    arabic = len(_ARABIC_CHAR_RE.findall(text or ""))
    latin = len(_LATIN_CHAR_RE.findall(text or ""))
    return arabic, latin


def dominant_script(text: str) -> str:
    """'arabic', 'latin', 'mixed' or 'neutral' for a string."""
    arabic, latin = script_counts(text)
    total = arabic + latin
    if total == 0:
        return "neutral"
    if arabic / total >= DOMINANCE:
        return "arabic"
    if latin / total >= DOMINANCE:
        return "latin"
    return "mixed"


def target_script(direction: str) -> str:
    return "arabic" if direction == "en2ar" else "latin"


def source_script(direction: str) -> str:
    return "latin" if direction == "en2ar" else "arabic"


def is_purely_target(text: str, direction: str) -> bool:
    """True when `text` contains no source-language letters at all.

    The preservation rule is deliberately strict in the other direction from
    `dominant_script`: a block qualifies only if there is nothing left to
    translate. An Arabic paragraph that merely mentions an English brand name
    still gets translated - the brand name itself survives because the
    translator is told to keep proper nouns.
    """
    if not (text or "").strip():
        return False
    arabic, latin = script_counts(text)
    if target_script(direction) == "arabic":
        return arabic > 0 and latin == 0
    return latin > 0 and arabic == 0


def is_already_target(text: str, direction: str) -> bool:
    """True when `text` is entirely in the language we are translating into.

    Such text is left exactly as it is - same words, same styling, same
    alignment, same position on the page. This is the guard that keeps an
    organisation name, a URL or an English caption from being re-translated,
    re-aligned or moved across the page.
    """
    return is_purely_target(text, direction)


def is_unchanged_by_translation(source: str, translated: Optional[str]) -> bool:
    """True when translating left the text effectively untouched.

    A bilingual label like "Arabic | العربية" is neither dominantly Latin nor
    dominantly Arabic, so the dominance test alone will not protect it - yet
    translating it changes nothing, and moving it across the page would be pure
    damage. Comparing the result to the source catches exactly that case.
    """
    if translated is None:
        return True
    return " ".join(source.split()) == " ".join(translated.split())


def should_preserve(source: str, translated: Optional[str], direction: str) -> bool:
    """Whether a block must be left exactly where and as it is.

    Either it was already written in the target language, or translation made no
    difference to it. In both cases the content is already correct, so it keeps
    its text, its styling, its alignment and its position.
    """
    if not (source or "").strip():
        return False
    return (is_already_target(source, direction)
            or is_unchanged_by_translation(source, translated))


def should_translate(text: str, direction: str) -> bool:
    """Whether a segment should be sent to the translator at all."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if not re.search(r"[^\W\d_]", stripped, re.UNICODE):
        return False  # digits and punctuation only
    return not is_already_target(text, direction)


# A span/block counts as RTL flow once Arabic holds this share of its
# characters. Deliberately lower than DOMINANCE: a heading like
# "مقدمة حول Platform Migration" is mixed by the dominance test yet is still
# Arabic-led text that belongs in the right-to-left flow.
RTL_SHARE = 0.3


def is_arabic(text: str) -> bool:
    """True when `text` is Arabic-led and so part of the right-to-left flow.

    This is a pure *script* test. It is deliberately independent of translation
    direction, unlike `is_already_target`: whether a run of text reads
    right-to-left is a property of the text itself, not of the job being run.
    Mirroring asks this question; the translator asks `should_translate`.
    """
    body = text or ""
    if not body.strip():
        return False
    arabic, latin = script_counts(body)
    if arabic == 0:
        return False
    # Measured against the letters that actually vote, so surrounding digits,
    # punctuation and whitespace cannot dilute a short Arabic label below the
    # threshold ("م 2024" is Arabic, not Latin).
    letters = arabic + latin
    return arabic >= letters * RTL_SHARE


def is_rtl_flow(text: str) -> bool:
    """Whether a span or block participates in right-to-left page flow."""
    return is_arabic(text)

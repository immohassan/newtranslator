"""Per-job QA report: records every place the pipeline had to compromise."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Severity = Literal["info", "warning", "error"]


@dataclass
class QAEntry:
    category: str
    severity: Severity
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


class QAReport:
    """Thread-safe collector. Never raises - QA logging must not break a job."""

    def __init__(self) -> None:
        self._entries: list[QAEntry] = []
        self._lock = threading.Lock()

    def add(self, category: str, severity: Severity, message: str, **detail: Any) -> None:
        with self._lock:
            self._entries.append(QAEntry(category, severity, message, detail))

    # -- convenience shorthands used across the pipeline -------------------
    def font_substitution(self, original: str, replacement: str, reason: str) -> None:
        self.add(
            "font_substitution",
            "info",
            f"Font '{original}' replaced with '{replacement}' ({reason}).",
            original=original,
            replacement=replacement,
            reason=reason,
        )

    def overflow(self, page: int, text: str, original_size: float, new_size: float) -> None:
        self.add(
            "text_overflow",
            "warning",
            f"Translated text on page {page} shrunk from {original_size:.1f}pt "
            f"to {new_size:.1f}pt to fit its box.",
            page=page,
            excerpt=text[:80],
            original_size=round(original_size, 2),
            new_size=round(new_size, 2),
        )

    def clipped(self, page: int, text: str) -> None:
        self.add(
            "text_clipped",
            "error",
            f"Translated text on page {page} could not fit even at the minimum "
            f"font size and may be truncated.",
            page=page,
            excerpt=text[:80],
        )

    def aggressive_shrink(self, page: int, text: str, original_size: float,
                          new_size: float, threshold: float) -> None:
        """A block that lost a large share of its size needs a human eye.

        Shrinking a little is normal - Arabic and English are never the same
        length. Shrinking past this threshold means the translation no longer
        fits the space the designer allowed, which is a layout decision the
        pipeline should not make silently.
        """
        lost = 1.0 - (new_size / original_size) if original_size else 0.0
        self.add(
            "layout_review",
            "warning",
            f"Translated text on page {page} had to shrink {lost * 100:.0f}% "
            f"({original_size:.1f}pt to {new_size:.1f}pt), more than the "
            f"{threshold * 100:.0f}% review threshold - check this block by hand.",
            page=page,
            excerpt=text[:80],
            original_size=round(original_size, 2),
            new_size=round(new_size, 2),
            shrink_ratio=round(lost, 3),
        )

    def block_overlap(self, page: int, text: str, kind: str,
                      element: tuple, other: tuple) -> None:
        """Two placed elements ended up on top of each other.

        Reported rather than silently overprinted: the translated text needed
        more room than the source did, and only a person can decide whether to
        re-flow the page or accept the collision.
        """
        self.add(
            "layout_review",
            "warning",
            f"Translated text on page {page} overlaps a neighbouring {kind} "
            f"after being laid out and needs manual adjustment.",
            page=page,
            excerpt=text[:80],
            element=element,
            other=other,
        )

    def mirror_issue(self, page: int, message: str, **detail: Any) -> None:
        self.add("mirror", "warning", f"Page {page}: {message}", page=page, **detail)

    def translation_failure(self, excerpt: str, error: str) -> None:
        self.add(
            "translation",
            "error",
            f"A text segment could not be translated and was left as-is: {error}",
            excerpt=excerpt[:80],
            error=error,
        )

    # -- serialisation -----------------------------------------------------
    @property
    def entries(self) -> list[QAEntry]:
        with self._lock:
            return list(self._entries)

    def counts(self) -> dict[str, int]:
        c = {"info": 0, "warning": 0, "error": 0}
        for e in self.entries:
            c[e.severity] = c.get(e.severity, 0) + 1
        return c

    def to_dict(self) -> dict[str, Any]:
        entries = self.entries
        by_category: dict[str, int] = {}
        for e in entries:
            by_category[e.category] = by_category.get(e.category, 0) + 1
        return {
            "counts": self.counts(),
            "by_category": by_category,
            "entries": [asdict(e) for e in entries],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

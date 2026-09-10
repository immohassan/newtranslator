#!/usr/bin/env python3
"""Command-line entry point - the same pipeline the web app runs.

    python translate_cli.py input.pdf output.pdf --direction en2ar
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.translate import provider_name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Translate a PDF or DOCX between English and Arabic, "
                    "preserving layout, images and styling."
    )
    parser.add_argument("input", help="source .pdf or .docx file")
    parser.add_argument("output", help="destination file (same format as input)")
    parser.add_argument("--direction", choices=["en2ar", "ar2en"], default="en2ar",
                        help="translation direction (default: en2ar)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="keep element positions and only re-align text, "
                             "instead of mirroring the layout horizontally. "
                             "Mirroring is per-block and automatic: each block "
                             "follows its own script, so English headings and "
                             "footers stay put without this flag.")
    parser.add_argument("--no-html", action="store_true",
                        help="rebuild PDFs by redrawing text at the source "
                             "coordinates instead of re-flowing semantic HTML. "
                             "The coordinate path cannot push content down when "
                             "a translation grows, so dense documents overlap; "
                             "use it only if a browser is unavailable.")
    parser.add_argument("--vision-layout", action="store_true",
                        help="read each page's layout with a vision model "
                             "instead of inferring it from the geometry. The "
                             "geometric reader cannot see a sidebar, so a "
                             "two-column CV reads as one interleaved run. Uses "
                             "whichever API key is set; costs one call per "
                             "page. Also settable with VISION_LAYOUT=true.")
    parser.add_argument("--no-underline", action="store_true",
                        help="drop underlines (uncommon in Arabic typography)")
    parser.add_argument("--no-flip-images", action="store_true",
                        help="do not horizontally flip arrows and other "
                             "directional graphics")
    parser.add_argument("--qa-report", default=None,
                        help="write the QA report here (default: alongside the "
                             "output as qa_report.json)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"error: {args.input} does not exist", file=sys.stderr)
        return 1

    in_ext = os.path.splitext(args.input)[1].lower()
    out_ext = os.path.splitext(args.output)[1].lower()
    if in_ext != out_ext:
        print(f"error: output must keep the input's format ({in_ext}), "
              f"got {out_ext}. Converting between PDF and DOCX destroys the "
              f"layout, so it is not supported.", file=sys.stderr)
        return 1

    options = TranslationOptions(
        direction=args.direction,
        mirror=not args.no_mirror,
        html_engine=not args.no_html,
        underline=not args.no_underline,
        flip_directional_images=not args.no_flip_images,
        # The flag turns it on; without it the environment decides, so a
        # deployment can enable it once rather than per invocation.
        **({"vision_layout": True} if args.vision_layout else {}),
    )

    print(f"Provider: {provider_name()}")

    def progress(stage: str, percent: int, message: str) -> None:
        print(f"[{percent:3d}%] {message}")

    try:
        qa = run_pipeline(args.input, args.output, options, progress=progress)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    qa_path = args.qa_report or os.path.join(
        os.path.dirname(os.path.abspath(args.output)), "qa_report.json"
    )
    qa.save(qa_path)

    counts = qa.counts()
    print(f"\nWrote {args.output}")
    print(f"QA report: {qa_path}")
    print(f"  {counts['error']} need attention, {counts['warning']} worth a look, "
          f"{counts['info']} notes")
    for entry in qa.entries:
        if entry.severity in ("error", "warning"):
            print(f"  ! {entry.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check that the configured providers actually answer, and say why if not.

A bad key does not announce itself: the pipeline is built to degrade rather
than fail, so a rejected request becomes untranslated text and a page read
from its geometry - a quietly worse document rather than an error. This makes
the failure loud, before a job is ever submitted.

Run it after editing .env:  ./check_keys.py
"""
from __future__ import annotations

import os
import sys


def _load_env(path: str = ".env") -> None:
    """Read .env, overriding the shell.

    Overriding is the point: a leftover key exported in the shell is exactly
    what this script exists to catch, so the file must win here even though
    `set -a; . ./.env` lets the shell win.
    """
    if not os.path.exists(path):
        sys.exit("error: no .env file. Copy .env.example to .env first.")
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            os.environ[name.strip()] = value.strip().strip('"').strip("'")


def _explain(error: str) -> str:
    """Turn an API error into the thing to do about it."""
    if "identity-linked" in error:
        return ("This key is identity-linked, so it belongs to a workspace "
                "and cannot be used without one.\n"
                "     Fix: create a standard key at "
                "console.anthropic.com -> Settings -> API Keys,\n"
                "     or set ANTHROPIC_WORKSPACE_ID to its workspace.")
    if "not_found_error" in error and "Workspace" in error:
        return ("ANTHROPIC_WORKSPACE_ID names a workspace this key cannot "
                "reach.\n"
                "     Fix: remove that line if your key is a standard one.")
    if "authentication_error" in error or "invalid x-api-key" in error:
        return "The key was rejected. Check it was copied in full."
    if "No endpoints found" in error or "not a valid model" in error:
        return ("OpenRouter does not have that model id.\n"
                "     Fix: pick one from openrouter.ai/models and set it as "
                "OPENROUTER_MODEL.")
    if "credit balance" in error or "billing" in error.lower():
        return "The account has no credit. Add billing in the console."
    return ""


def main() -> int:
    _load_env()
    from app.core import translate, vision_layout

    failures = 0

    print(f"translation provider : {translate.provider_name()}")
    try:
        out = translate.translate("Beauty Advisor", "en2ar")
        print(f"  OK  -> {out}")
    except Exception as exc:
        failures += 1
        print(f"  FAILED: {exc}")
        hint = _explain(str(exc))
        if hint:
            print(f"     {hint}")

    print(f"layout provider      : {vision_layout.provider_name()}")
    if not vision_layout.is_available():
        print("  not configured - pages will be read from their geometry, "
              "which cannot see a sidebar.")
    else:
        print("  OK  (a real page is only read when a job runs)")

    if failures:
        print("\nThe app still runs with a failing provider: text is left "
              "untranslated\nrather than the job failing. Fix the above "
              "before sending anything to a client.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

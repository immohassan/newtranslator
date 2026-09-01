import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import translate as translate_mod  # noqa: E402
from tests.fake_provider import FakeProvider  # noqa: E402
from tests.make_fixtures import (  # noqa: E402
    make_bulleted_pdf,
    make_docx,
    make_full_page_pdf,
    make_mixed_pdf,
    make_pdf,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


PROVIDER_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_TRANSLATE_API_KEY",
    "DEEPL_API_KEY",
    "TRANSLATION_PROVIDER",
    "TRANSLATION_MODEL",
    "OPENAI_MODEL",
)


@pytest.fixture(autouse=True)
def offline_provider(monkeypatch):
    """Every test runs against the deterministic provider - no network, no keys.

    Real provider keys are stripped from the environment too, so a developer
    with OPENAI_API_KEY exported cannot get different results (or spend money)
    by running the suite.
    """
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    translate_mod.set_provider(FakeProvider())
    yield
    translate_mod.set_provider(FakeProvider())


@pytest.fixture(scope="session")
def sample_pdf():
    os.makedirs(FIXTURES, exist_ok=True)
    path = os.path.join(FIXTURES, "sample.pdf")
    if not os.path.exists(path):
        make_pdf(path)
    return path


@pytest.fixture(scope="session")
def sample_docx():
    os.makedirs(FIXTURES, exist_ok=True)
    path = os.path.join(FIXTURES, "sample.docx")
    if not os.path.exists(path):
        make_docx(path)
    return path


@pytest.fixture(scope="session")
def mixed_pdf():
    """Arabic body text with English headings, plus a tight box for overflow."""
    os.makedirs(FIXTURES, exist_ok=True)
    path = os.path.join(FIXTURES, "mixed.pdf")
    if not os.path.exists(path):
        make_mixed_pdf(path)
    return path


@pytest.fixture(scope="session")
def full_page_pdf():
    """Arabic body with English heading, subtitle and footer - a whole page."""
    os.makedirs(FIXTURES, exist_ok=True)
    path = os.path.join(FIXTURES, "full_page.pdf")
    if not os.path.exists(path):
        make_full_page_pdf(path)
    return path


@pytest.fixture(scope="session")
def bulleted_pdf():
    """Bullet and numbered lists, side icons, and a footer icon row."""
    os.makedirs(FIXTURES, exist_ok=True)
    path = os.path.join(FIXTURES, "bulleted.pdf")
    if not os.path.exists(path):
        make_bulleted_pdf(path)
    return path

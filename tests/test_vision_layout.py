"""The vision layout reader: what it does with a model's answer.

These tests never reach the network. What matters here is not that the model
is clever - it is that a *wrong* answer cannot damage the document, so every
test below feeds a deliberately broken reply and asserts the page survives it.
"""
from __future__ import annotations

import json
import types

import fitz
import pytest

from app.core.extract import extract_pdf
from app.core.qa import QAReport
from app.core import vision_layout
from app.core.vision_structure import read_structure


def _client(reply: dict, refuse: bool = False):
    """A stand-in reader that answers with `reply`, whatever the backend."""

    class Reader(vision_layout.LayoutReader):
        name = "fake"

        def read(self, image_b64, prompt):
            self.image_b64 = image_b64
            self.prompt = prompt
            if refuse:
                raise vision_layout._Refused()
            return json.dumps(reply)

    return Reader()


@pytest.fixture
def page(full_page_pdf):
    qa = QAReport()
    doc = extract_pdf(str(full_page_pdf), qa)
    return doc.pages[0]


@pytest.fixture
def source(full_page_pdf):
    with fitz.open(str(full_page_pdf)) as doc:
        yield doc[0]


def _ids(page):
    return [i for i, b in enumerate(page.blocks) if b.text.strip()]


def _one_column(page, kind="paragraph"):
    return {"columns": [{"role": "main",
                         "blocks": [{"id": i, "kind": kind, "level": 2}
                                    for i in _ids(page)]}]}


def _text_of(blocks) -> str:
    return " ".join(b.text for b in blocks)


def test_every_block_is_kept(page, source):
    """The page's text all arrives, and none of it twice."""
    qa = QAReport()
    blocks = read_structure(page, qa, source, client=_client(_one_column(page)))
    assert len(blocks) == len(_ids(page))


def test_a_block_the_model_forgot_is_not_lost(page, source):
    """Losing text silently is the one failure this must never have."""
    reply = _one_column(page)
    dropped = reply["columns"][0]["blocks"].pop(3)
    missing = page.blocks[dropped["id"]].text.strip()

    qa = QAReport()
    blocks = read_structure(page, qa, source, client=_client(reply))

    assert missing.split()[0] in _text_of(blocks)
    assert any("not placed" in e.message for e in qa.entries)


def test_a_repeated_id_is_emitted_once(page, source):
    """A duplicated id would print the same sentence twice."""
    reply = _one_column(page)
    reply["columns"][0]["blocks"].append(dict(reply["columns"][0]["blocks"][0]))

    qa = QAReport()
    blocks = read_structure(page, qa, source, client=_client(reply))
    assert len(blocks) == len(_ids(page))


def test_an_invented_id_is_ignored(page, source):
    """An id for a block that does not exist must not reach the renderer."""
    reply = _one_column(page)
    reply["columns"][0]["blocks"].append(
        {"id": 9999, "kind": "paragraph", "level": 2})

    qa = QAReport()
    blocks = read_structure(page, qa, source, client=_client(reply))
    assert len(blocks) == len(_ids(page))


def test_the_model_never_supplies_text(page, source):
    """Content comes from the file. A model that writes text is ignored."""
    reply = _one_column(page)
    reply["columns"][0]["blocks"][0]["text"] = "COMPLETELY INVENTED CONTENT"

    qa = QAReport()
    blocks = read_structure(page, qa, source, client=_client(reply))
    assert "INVENTED" not in _text_of(blocks)


def test_two_columns_are_marked_for_the_renderer(page, source):
    """A two-column answer marks each run so the columns are set side by side."""
    ids = _ids(page)
    half = len(ids) // 2
    reply = {"columns": [
        {"role": "sidebar",
         "blocks": [{"id": i, "kind": "paragraph", "level": 2}
                    for i in ids[:half]]},
        {"role": "main",
         "blocks": [{"id": i, "kind": "paragraph", "level": 2}
                    for i in ids[half:]]},
    ]}

    qa = QAReport()
    blocks = read_structure(page, qa, source, client=_client(reply))

    starts = [b.column_start for b in blocks if b.column_start]
    assert len(starts) == 2
    assert sum(1 for b in blocks if b.column_end) == 2


def test_a_single_column_page_is_not_marked(page, source):
    """One column must render exactly as it did before columns existed."""
    qa = QAReport()
    blocks = read_structure(page, qa, source, client=_client(_one_column(page)))
    assert not any(b.column_start or b.column_end for b in blocks)


def test_a_refusal_falls_back_to_the_geometric_reader(page, source):
    qa = QAReport()
    blocks = read_structure(page, qa, source,
                            client=_client({"columns": []}, refuse=True))
    assert blocks  # the geometric reader still produced a page
    assert any("declined" in e.message for e in qa.entries)


def test_malformed_json_falls_back(page, source):
    class Reader(vision_layout.LayoutReader):
        def read(self, image_b64, prompt):
            return "not json"

    qa = QAReport()
    blocks = read_structure(page, qa, source, client=Reader())
    assert blocks
    assert any("could not be read" in e.message for e in qa.entries)


def test_an_api_failure_falls_back(page, source):
    class Reader(vision_layout.LayoutReader):
        def read(self, image_b64, prompt):
            raise RuntimeError("connection reset")

    qa = QAReport()
    blocks = read_structure(page, qa, source, client=Reader())
    assert blocks
    assert any("geometry" in e.message for e in qa.entries)


def test_the_model_is_shown_the_page_and_the_blocks(page, source):
    """The request must carry both halves: the image and the extracted text."""
    client = _client(_one_column(page))
    read_structure(page, QAReport(), source, client=client)

    assert client.image_b64           # the page was rendered for the model
    # Ids are what the model answers with, so they have to be in the prompt.
    assert "[0]" in client.prompt


def test_the_backend_follows_whichever_key_is_set(monkeypatch):
    """Either vendor's key enables the reader - it is not Anthropic-only."""
    monkeypatch.delenv("VISION_LAYOUT_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert vision_layout._reader().name == "openai"

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert vision_layout._reader().name == "anthropic"


def test_the_provider_can_be_named_explicitly(monkeypatch):
    """A key for both set does not have to mean the default wins."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("VISION_LAYOUT_PROVIDER", "anthropic")
    assert vision_layout._reader().name == "anthropic"


def test_no_key_means_no_reader(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VISION_LAYOUT_PROVIDER", raising=False)
    assert not vision_layout.is_available()


def test_a_page_with_too_little_on_it_skips_the_model(page, source, monkeypatch):
    """An API call for a page with no layout to speak of is pure latency."""
    monkeypatch.setattr(vision_layout, "MIN_BLOCKS", 10_000)

    called = False

    class Reader(vision_layout.LayoutReader):
        def read(self, image_b64, prompt):
            nonlocal called
            called = True
            raise AssertionError("the model should not have been asked")

    blocks = read_structure(page, QAReport(), source, client=Reader())
    assert blocks
    assert not called


def test_openrouter_is_selected_by_its_own_key(monkeypatch):
    """An OpenRouter key alone is enough - no other vendor account needed."""
    for name in ("VISION_LAYOUT_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert vision_layout._reader().name == "openrouter"


def test_openrouter_asks_for_plain_json_not_a_strict_schema(monkeypatch):
    """`strict` is an OpenAI feature; a Claude model routed here rejects it."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("VISION_LAYOUT_PROVIDER", "openrouter")
    reader = vision_layout._reader()
    assert reader.RESPONSE_FORMAT == {"type": "json_object"}
    assert vision_layout.OpenAIReader.RESPONSE_FORMAT["type"] == "json_schema"


def test_openrouter_states_the_reply_shape_in_the_prompt(monkeypatch):
    """With no schema on the request the shape must be said in the prompt."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    reader = vision_layout.OpenRouterReader()

    sent = {}

    class Completions:
        def create(self, **kwargs):
            sent.update(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="{}", refusal=None))])

    reader._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=Completions()))
    reader.read("aGk=", "Text blocks:\n[0] Something")

    text = next(part["text"] for part in sent["messages"][1]["content"]
                if part["type"] == "text")
    assert '"columns"' in text
    assert sent["response_format"] == {"type": "json_object"}


def test_openrouter_defaults_to_a_claude_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("VISION_LAYOUT_MODEL", raising=False)
    assert "claude" in vision_layout.OpenRouterReader().model

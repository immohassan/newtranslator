"""Provider abstraction: batching, response parsing, failure handling."""
import json

import pytest

from app.core import translate as T


class Recorder(T.TranslationProvider):
    """Captures what the pipeline asks for, returns a marked translation."""

    name = "recorder"

    def __init__(self):
        self.calls = []

    def translate_batch(self, texts, direction):
        self.calls.append(list(texts))
        return [f"<{t}>" for t in texts]


def test_translate_skips_non_words():
    """Whitespace, digits and punctuation are not worth an API call."""
    provider = Recorder()
    T.set_provider(provider)
    assert T.translate("   ", "en2ar") == "   "
    assert T.translate("42", "en2ar") == "42"
    assert T.translate("...", "en2ar") == "..."
    assert provider.calls == []


def test_translate_batch_preserves_indices():
    """Untranslatable slots keep their place so callers can zip results back."""
    T.set_provider(Recorder())
    out = T.translate_batch(["Hello", "  ", "World"], "en2ar")
    assert out[1] == "  "
    assert out[0] == "<Hello>" and out[2] == "<World>"


def test_batches_are_chunked_by_size():
    provider = Recorder()
    T.set_provider(provider)
    texts = ["word " * 400 for _ in range(8)]  # ~2000 chars each
    T.translate_batch(texts, "en2ar")
    assert len(provider.calls) > 1, "large inputs must be split across requests"
    for call in provider.calls:
        assert sum(len(t) for t in call) <= T.MAX_BATCH_CHARS + 2000


def test_batch_sends_whole_paragraphs():
    """Context matters for idiom, so paragraphs go over whole, not word by word."""
    provider = Recorder()
    T.set_provider(provider)
    paragraph = "The team completed the migration ahead of schedule."
    T.translate_batch([paragraph], "en2ar")
    assert provider.calls == [[paragraph]]


def test_provider_failure_keeps_source_text():
    """A failed batch must not drop content from the document."""

    class Broken(T.TranslationProvider):
        name = "broken"

        def translate_batch(self, texts, direction):
            raise RuntimeError("API is down")

    T.set_provider(Broken())
    errors = []
    out = T.translate_batch(["Hello", "World"], "en2ar",
                            on_error=lambda text, err: errors.append(err))
    assert out == ["Hello", "World"], "source text must survive a failure"
    assert errors and "API is down" in errors[0]


def test_wrong_result_count_is_an_error():
    class Miscounting(T.TranslationProvider):
        name = "miscounting"

        def translate_batch(self, texts, direction):
            return ["only one"]

    T.set_provider(Miscounting())
    errors = []
    out = T.translate_batch(["a", "b"], "en2ar",
                            on_error=lambda text, err: errors.append(err))
    assert out == ["a", "b"]
    assert errors


def test_unknown_direction_rejected():
    with pytest.raises(ValueError):
        T.translate_batch(["hello"], "fr2de")


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _AnthropicResponse:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason


def _anthropic_stub(monkeypatch, response):
    provider = T.AnthropicProvider(api_key="test-key")

    class Messages:
        def create(self, **kwargs):
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(provider, "_client",
                        type("C", (), {"messages": Messages()})())
    return provider


def test_anthropic_parses_json_response(monkeypatch):
    """The provider must survive commentary around the JSON payload."""
    provider = _anthropic_stub(
        monkeypatch,
        _AnthropicResponse('Here you go:\n{"0": "مرحبا", "1": "عالم"}'),
    )
    assert provider.translate_batch(["Hello", "World"], "en2ar") == ["مرحبا", "عالم"]


def test_anthropic_bad_json_raises(monkeypatch):
    provider = _anthropic_stub(monkeypatch, _AnthropicResponse("sorry"))
    with pytest.raises(T.TranslationError):
        provider.translate_batch(["Hello"], "en2ar")


def test_anthropic_refusal_is_reported(monkeypatch):
    """A safety refusal must surface, not be parsed as a failed response."""
    provider = _anthropic_stub(
        monkeypatch, _AnthropicResponse("", stop_reason="refusal")
    )
    with pytest.raises(T.TranslationError, match="declined"):
        provider.translate_batch(["Hello"], "en2ar")


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(T.TranslationError):
        T.AnthropicProvider()


def test_echo_provider_is_the_offline_default(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TRANSLATION_PROVIDER", raising=False)
    T._active = None
    assert T.get_provider().name == "echo"


def test_bad_provider_name_falls_back(monkeypatch):
    monkeypatch.setenv("TRANSLATION_PROVIDER", "nonexistent")
    T._active = None
    assert T.get_provider().name == "echo"


# ---------------------------------------------------------------- OpenAI
class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _openai_stub(monkeypatch, content, capture=None):
    """Replace the provider's client with one returning a canned reply."""
    provider = T.OpenAIProvider(api_key="test-key")

    class Completions:
        def create(self, **kwargs):
            if capture is not None:
                capture.update(kwargs)
            if isinstance(content, Exception):
                raise content
            return _FakeCompletion(content)

    class Chat:
        completions = Completions()

    monkeypatch.setattr(provider, "_client", type("C", (), {"chat": Chat()})())
    return provider


def test_openai_parses_json_response(monkeypatch):
    provider = _openai_stub(monkeypatch, '{"0": "مرحبا", "1": "عالم"}')
    assert provider.translate_batch(["Hello", "World"], "en2ar") == ["مرحبا", "عالم"]


def test_openai_requests_json_mode(monkeypatch):
    """json_object mode is what keeps the reply parseable."""
    captured = {}
    provider = _openai_stub(monkeypatch, '{"0": "مرحبا"}', capture=captured)
    provider.translate_batch(["Hello"], "en2ar")
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["model"] == "gpt-4o"


def test_openai_missing_key_falls_back_to_source(monkeypatch):
    """A dropped key must leave the source text, never lose the paragraph."""
    provider = _openai_stub(monkeypatch, '{"0": "مرحبا"}')
    assert provider.translate_batch(["Hello", "World"], "en2ar") == ["مرحبا", "World"]


def test_openai_bad_json_raises(monkeypatch):
    provider = _openai_stub(monkeypatch, "sorry, I cannot do that")
    with pytest.raises(T.TranslationError):
        provider.translate_batch(["Hello"], "en2ar")


def test_openai_api_error_becomes_translation_error(monkeypatch):
    provider = _openai_stub(monkeypatch, RuntimeError("connection reset"))
    with pytest.raises(T.TranslationError, match="OpenAI request failed"):
        provider.translate_batch(["Hello"], "en2ar")


def test_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(T.TranslationError):
        T.OpenAIProvider()


def test_openai_model_overridable(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    assert T.OpenAIProvider(api_key="k").model == "gpt-4o-mini"


def test_openai_is_default_when_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("TRANSLATION_PROVIDER", raising=False)
    T._active = None
    assert T.get_provider().name == "openai"


def test_anthropic_uses_opus_by_default(monkeypatch):
    monkeypatch.delenv("TRANSLATION_MODEL", raising=False)
    assert T.AnthropicProvider(api_key="k").model == "claude-opus-5"


def test_batch_prompt_keys_every_paragraph():
    """Index keys are what stop a merged reply shifting later paragraphs."""
    prompt = T._batch_prompt(["first", "second", "third"], "en2ar")
    assert '"0"' in prompt and '"1"' in prompt and '"2"' in prompt
    assert "English" in prompt and "Arabic" in prompt

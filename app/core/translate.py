"""Provider-agnostic translation.

Everything downstream calls `translate(text, direction)` or the batched
`translate_batch`. Swapping provider means adding one class and setting
TRANSLATION_PROVIDER - no pipeline code changes.

Text is batched at paragraph level, never word by word: idiom and agreement
need the surrounding sentence to come out right.
"""
from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Optional

log = logging.getLogger(__name__)

MAX_BATCH_CHARS = 8000
MAX_RETRIES = 3  # passed to each SDK client; they back off internally

DIRECTION_LABELS = {
    "en2ar": ("English", "Arabic"),
    "ar2en": ("Arabic", "English"),
}


# An LLM asked for a translation sometimes answers with the *escape sequence*
# rather than the character - a literal backslash-n where a line break belongs.
# Written straight into the PDF it renders as visible "\n" in the middle of a
# sentence, which is what it looked like in the reported Arabic output.
_ESCAPE_FIXES = (
    ("\\r\\n", "\n"),
    ("\\n", "\n"),
    ("\\r", "\n"),
    ("\\t", " "),
)


def clean_translation(text: str) -> str:
    """Turn escape sequences a provider emitted literally into real characters.

    Only the sequences a translator plausibly produces are converted; the text
    is not otherwise unescaped, so a backslash that genuinely belongs to the
    content survives.
    """
    if not text or "\\" not in text:
        return text
    out = text
    for escaped, real in _ESCAPE_FIXES:
        out = out.replace(escaped, real)
    return out


# A heading a template set with letter-spacing extracts as "D E T A I L S":
# the tracking is real spaces between the glyphs, not a style the file records.
# Sent on as-is it is translated letter by letter, and in Arabic the injected
# spaces are worse than cosmetic - they break the cursive join, so the word
# renders as a row of disconnected letterforms. The run is closed up before
# translation, which is the only point where the damage can still be undone.
#
# The test is deliberately narrow. A run must be at least this many
# single-character tokens in a row before it is read as tracking, so ordinary
# prose - "a", "I", initials in "J. R. R. Tolkien" - is never touched.
_TRACKED_MIN_RUN = 4
# Tracking separates the letters of one word by a single space and its words by
# a wider gap, so a run is matched only across single spaces. That keeps the
# word break in "E M P L O Y M E N T  H I S T O R Y" intact - each word closes
# up on its own - instead of fusing the two into one.
_TRACKED_RE = re.compile(
    r"(?<!\S)((?:[^\W\d_] ){%d,}[^\W\d_])(?!\S)" % (_TRACKED_MIN_RUN - 1)
)


def _untrack(text: str) -> str:
    """Close up a run of letter-spaced characters into a word.

    "D E T A I L S" -> "DETAILS". Runs of two or fewer are left alone, and a
    tracked run inside a longer line is closed without disturbing the rest of
    it, so "E M P L O Y M E N T  H I S T O R Y" becomes two words rather than
    one. Text with no such run is returned unchanged.
    """
    if not text or text.count(" ") < _TRACKED_MIN_RUN - 1:
        return text

    def close(match: "re.Match[str]") -> str:
        run = match.group(1)
        letters = run.split()
        # Tracking is set in one case throughout. Mixed case is a genuine run
        # of short words, not a spaced-out word.
        if any(c.isupper() for c in run) and any(c.islower() for c in run):
            return run
        return "".join(letters)

    out = _TRACKED_RE.sub(close, text)
    if out == text:
        return text
    # The wider gap that separated two tracked words is now an ordinary double
    # space between them, so it is closed to one.
    return re.sub(r" {2,}", " ", out)


def _is_translatable(text: str, direction: Optional[str] = None) -> bool:
    """Whether a segment should be sent to the translator.

    Skips whitespace, digits and punctuation-only fragments - sending them
    wastes quota and providers often mangle them - and, when `direction` is
    known, anything already written in the target language. Re-translating an
    organisation name or a URL that is already correct only corrupts it.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if not re.search(r"[^\W\d_]", stripped, re.UNICODE):
        return False
    if direction is not None:
        from .language import is_already_target

        if is_already_target(stripped, direction):
            return False
    return True


class TranslationError(RuntimeError):
    pass


class TranslationProvider(ABC):
    name = "base"

    @abstractmethod
    def translate_batch(self, texts: list[str], direction: str) -> list[str]:
        """Return one translation per input, same order, same length."""

    def translate(self, text: str, direction: str) -> str:
        return self.translate_batch([text], direction)[0]


class EchoProvider(TranslationProvider):
    """Offline default: returns the source text unchanged.

    Keeps the whole pipeline runnable and testable with no API key. Layout,
    shaping and rebuild logic are all exercised; only the words stay English.
    """

    name = "echo"

    def translate_batch(self, texts: list[str], direction: str) -> list[str]:
        return list(texts)


def _batch_prompt(texts: list[str], direction: str) -> str:
    """One prompt for a whole batch, keyed by index.

    Keying by index rather than concatenating with separators means a model that
    merges or splits lines cannot silently shift every later translation onto
    the wrong paragraph.
    """
    src, dst = DIRECTION_LABELS[direction]
    numbered = {str(i): t for i, t in enumerate(texts)}
    return (
        f"Translate each value in this JSON object from {src} to {dst}.\n"
        f"Rules:\n"
        f"- Return ONLY a JSON object with the same keys, no commentary.\n"
        f"- Translate the meaning naturally; do not transliterate.\n"
        f"- Preserve numbers, dates, URLs, emails and proper nouns.\n"
        f"- Keep any leading/trailing whitespace of each value.\n"
        f"- Return {dst} in normal logical character order.\n\n"
        f"{json.dumps(numbered, ensure_ascii=False)}"
    )


def _parse_batch_response(raw: str, texts: list[str]) -> list[str]:
    """Pull the JSON object out of a model reply and map it back to positions.

    Any key the model dropped falls back to the source text, so a partial reply
    degrades to untranslated text rather than losing content.
    """
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        raise TranslationError("provider response was not valid JSON")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise TranslationError(f"provider response was not valid JSON: {exc}")
    return [str(parsed.get(str(i), texts[i])) for i in range(len(texts))]


class OpenAIProvider(TranslationProvider):
    """OpenAI chat completions. Set OPENAI_API_KEY to enable."""

    name = "openai"
    MODEL = "gpt-4o"
    # Left unset here: OpenAI bills what a reply actually uses, so a ceiling
    # only risks truncating a long batch. Subclasses whose host reserves
    # credit up front override it.
    MAX_TOKENS: Optional[int] = None

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", self.MODEL)
        if not self.api_key:
            raise TranslationError("OPENAI_API_KEY is not set.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise TranslationError(f"the openai package is not installed: {exc}")
        # max_retries covers 429s and 5xx with backoff; no hand-rolled retry loop.
        self._client = OpenAI(api_key=self.api_key, timeout=120.0,
                              max_retries=MAX_RETRIES)

    def translate_batch(self, texts: list[str], direction: str) -> list[str]:
        src, dst = DIRECTION_LABELS[direction]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": f"You are a professional {src}-to-{dst} translator. "
                                f"You reply with JSON only."},
                    {"role": "user", "content": _batch_prompt(texts, direction)},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                **({"max_tokens": self.MAX_TOKENS} if self.MAX_TOKENS else {}),
            )
        except Exception as exc:
            # "OpenAI" here names the *protocol*, which is what OpenRouter
            # speaks too; the failure text itself carries the endpoint.
            raise TranslationError(f"OpenAI request failed: {exc}")
        return _parse_batch_response(response.choices[0].message.content or "", texts)


class OpenRouterProvider(OpenAIProvider):
    """Any model OpenRouter hosts - Claude included - over its OpenAI-shaped API.

    OpenRouter speaks the OpenAI chat-completions protocol, so the whole of
    `OpenAIProvider` applies unchanged and only the endpoint, the key and the
    model name differ. Reaching Claude this way avoids the Anthropic account
    setup entirely: an OpenRouter key is never workspace-scoped, which is the
    thing that blocked the direct route.
    """

    name = "openrouter"
    MODEL = "anthropic/claude-opus-4.1"
    BASE_URL = "https://openrouter.ai/api/v1"
    # OpenRouter reserves credit for the *whole* of max_tokens before the
    # request runs, not for what it turns out to use, and with none set the
    # model's own ceiling applies - 32k for Claude, which is refused outright
    # on a small balance. A batch is capped at MAX_BATCH_CHARS of input, so
    # the reply cannot be anywhere near that: this is the real ceiling, and
    # setting it is what keeps the reservation proportionate.
    MAX_TOKENS = 8000

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or os.environ.get("OPENROUTER_MODEL", self.MODEL)
        if not self.api_key:
            raise TranslationError("OPENROUTER_API_KEY is not set.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise TranslationError(f"the openai package is not installed: {exc}")
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=os.environ.get("OPENROUTER_BASE_URL", self.BASE_URL),
            timeout=120.0, max_retries=MAX_RETRIES,
            # OpenRouter attributes requests to an app by these headers. They
            # are optional, and sent only when configured.
            default_headers=_openrouter_headers(),
        )


def _openrouter_headers() -> dict[str, str]:
    """The optional attribution headers OpenRouter reads, if they are set."""
    headers = {}
    referer = os.environ.get("OPENROUTER_SITE_URL", "").strip()
    title = os.environ.get("OPENROUTER_APP_NAME", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


class AnthropicProvider(TranslationProvider):
    """Claude via the official SDK. Set ANTHROPIC_API_KEY to enable."""

    name = "anthropic"
    MODEL = "claude-opus-5"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("TRANSLATION_MODEL", self.MODEL)
        if not self.api_key:
            raise TranslationError("ANTHROPIC_API_KEY is not set.")
        try:
            import anthropic
        except ImportError as exc:
            raise TranslationError(f"the anthropic package is not installed: {exc}")
        # The SDK retries 429/5xx with backoff itself.
        #
        # An identity-linked key is scoped to a workspace and the API rejects a
        # request that does not name one, so the header is sent whenever the
        # id is configured. Ordinary keys need no workspace and ignore it.
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        self._client = anthropic.Anthropic(
            api_key=self.api_key, timeout=120.0, max_retries=MAX_RETRIES,
            default_headers=({"anthropic-workspace-id": workspace}
                             if workspace else None))

    def translate_batch(self, texts: list[str], direction: str) -> list[str]:
        src, dst = DIRECTION_LABELS[direction]
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=f"You are a professional {src}-to-{dst} translator. "
                       f"You reply with JSON only.",
                messages=[{"role": "user",
                           "content": _batch_prompt(texts, direction)}],
            )
        except Exception as exc:
            raise TranslationError(f"Anthropic request failed: {exc}")

        if getattr(response, "stop_reason", None) == "refusal":
            raise TranslationError("the model declined to translate this content")

        raw = "".join(b.text for b in response.content if b.type == "text").strip()
        return _parse_batch_response(raw, texts)


class GoogleProvider(TranslationProvider):
    """Google Cloud Translation v2. Set GOOGLE_TRANSLATE_API_KEY to enable."""

    name = "google"
    ENDPOINT = "https://translation.googleapis.com/language/translate/v2"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_TRANSLATE_API_KEY", "")
        if not self.api_key:
            raise TranslationError("GOOGLE_TRANSLATE_API_KEY is not set.")

    def translate_batch(self, texts: list[str], direction: str) -> list[str]:
        import html

        import httpx

        source, target = ("en", "ar") if direction == "en2ar" else ("ar", "en")
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    self.ENDPOINT,
                    params={"key": self.api_key},
                    json={"q": texts, "source": source, "target": target, "format": "text"},
                )
            resp.raise_for_status()
            items = resp.json()["data"]["translations"]
        except Exception as exc:
            raise TranslationError(f"Google Translate request failed: {exc}")
        return [html.unescape(i["translatedText"]) for i in items]


class DeepLProvider(TranslationProvider):
    """DeepL. Note: DeepL has no Arabic target for some plans - it raises then."""

    name = "deepl"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPL_API_KEY", "")
        if not self.api_key:
            raise TranslationError("DEEPL_API_KEY is not set.")
        self.endpoint = (
            "https://api-free.deepl.com/v2/translate"
            if self.api_key.endswith(":fx")
            else "https://api.deepl.com/v2/translate"
        )

    def translate_batch(self, texts: list[str], direction: str) -> list[str]:
        import httpx

        source, target = ("EN", "AR") if direction == "en2ar" else ("AR", "EN-US")
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    self.endpoint,
                    headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
                    json={"text": texts, "source_lang": source, "target_lang": target},
                )
            resp.raise_for_status()
            return [t["text"] for t in resp.json()["translations"]]
        except Exception as exc:
            raise TranslationError(f"DeepL request failed: {exc}")


_PROVIDERS: dict[str, type[TranslationProvider]] = {
    "echo": EchoProvider,
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "deepl": DeepLProvider,
}

_active: Optional[TranslationProvider] = None


def get_provider() -> TranslationProvider:
    """Resolve the configured provider, falling back to echo if it can't start."""
    global _active
    if _active is not None:
        return _active
    name = os.environ.get("TRANSLATION_PROVIDER", "").strip().lower()
    if not name:
        if os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.environ.get("OPENROUTER_API_KEY"):
            name = "openrouter"
        else:
            name = "echo"
    cls = _PROVIDERS.get(name, EchoProvider)
    try:
        _active = cls()
    except Exception as exc:
        log.warning("Provider %s unavailable (%s); falling back to echo.", name, exc)
        _active = EchoProvider()
    return _active


def set_provider(provider: TranslationProvider) -> None:
    """Injection point for tests and for swapping provider at runtime."""
    global _active
    _active = provider


def provider_name() -> str:
    return get_provider().name


def translate(text: str, direction: str) -> str:
    """Translate one string. Returns the input unchanged if it has no words."""
    if not _is_translatable(text, direction):
        return text
    return get_provider().translate_batch([text], direction)[0]


def translate_batch(
    texts: list[str], direction: str, on_error=None
) -> list[str]:
    """Translate many strings, chunked to stay inside provider limits.

    Untranslatable fragments pass through untouched and keep their slot, so the
    returned list always lines up index-for-index with `texts`.
    """
    if direction not in DIRECTION_LABELS:
        raise ValueError(f"Unknown direction '{direction}'.")

    # Letter-spaced runs are closed up before anything else looks at the text.
    # A tracked heading is otherwise translated character by character, and in
    # Arabic the spaces between those characters break the cursive join.
    texts = [_untrack(t) for t in texts]

    results = list(texts)
    indices = [i for i, t in enumerate(texts) if _is_translatable(t, direction)]
    if not indices:
        return results

    provider = get_provider()
    chunk: list[int] = []
    chunk_chars = 0

    def flush(batch: list[int]) -> None:
        if not batch:
            return
        payload = [texts[i] for i in batch]
        try:
            out = provider.translate_batch(payload, direction)
            if len(out) != len(payload):
                raise TranslationError(
                    f"provider returned {len(out)} translations for "
                    f"{len(payload)} inputs"
                )
            for slot, value in zip(batch, out):
                # Applied once here rather than in each provider, so every
                # backend gets the same treatment.
                results[slot] = clean_translation(value)
        except Exception as exc:
            log.exception("Batch translation failed")
            if on_error is not None:
                on_error(payload[0] if payload else "", str(exc))
            # leave source text in place rather than dropping content

    for i in indices:
        size = len(texts[i])
        if chunk and chunk_chars + size > MAX_BATCH_CHARS:
            flush(chunk)
            chunk, chunk_chars = [], 0
        chunk.append(i)
        chunk_chars += size
    flush(chunk)

    return results

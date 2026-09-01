# Document Translator — English ⇄ Arabic

Translates PDF and DOCX documents between English and Arabic while keeping the
original layout, images and styling. A PDF comes back as a PDF and a DOCX as a
DOCX — there is no lossy conversion step in between.

## What it does

- **Structure-aware extraction** — every text span with its font, size, colour,
  bold/italic/underline flags and bounding box, plus images and vector rules.
- **RTL layout mirroring** — for English → Arabic the whole page is reflected
  horizontally (`new_x0 = W - x1`), so images, text blocks and rules move to the
  side an Arabic reader expects. Toggleable; the inverse applies for Arabic → English.
- **Correct Arabic rendering** — `arabic_reshaper` for joined letterforms and
  `python-bidi` for visual ordering, applied immediately before drawing.
- **Formatting preservation** — real bold font files (never synthetic bold),
  original sizes and colours, and underlines redrawn to the translated text's width.
- **Mixed-language guard rails** — content already written in the target
  language is left completely alone: not re-translated, not re-shaped, and not
  re-aligned. An English organisation name or URL inside an Arabic document
  keeps its exact bytes and its left-to-right alignment, and the same applies to
  Arabic inside an English document. Every preserved segment is listed in the
  QA report.
- **Honest QA reporting** — font substitutions, font-size reductions, untranslated
  segments and elements that could not be cleanly mirrored are all recorded and
  shown to the user rather than silently degraded.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Arabic fonts: NotoNaskhArabic-Regular.ttf and NotoNaskhArabic-Bold.ttf must be
# in fonts/. On Debian/Ubuntu: apt install fonts-noto-core, then copy them from
# /usr/share/fonts/truetype/noto/.

export APP_PASSWORD=password
export OPENAI_API_KEY=sk-...           # optional; see "Translation providers"
.venv/bin/uvicorn app.web.main:app --port 8000
```

Open <http://localhost:8000>, sign in, upload a document.

### Command line

```bash
.venv/bin/python translate_cli.py input.pdf output_ar.pdf --direction en2ar
.venv/bin/python translate_cli.py input.docx output_en.docx --direction ar2en --no-mirror
```

## Configuration

Every setting is an environment variable — see `.env.example`. The important ones:

| Variable | Default | Purpose |
|---|---|---|
| `APP_USERNAME` / `APP_PASSWORD` | `admin` / `changeme` | Login. The password is bcrypt-hashed at startup. |
| `APP_PASSWORD_HASH` | — | Supply a bcrypt hash instead of a plaintext password. Preferred in production. |
| `SECRET_KEY` | random per start | Signs session cookies. Set it, or restarts invalidate sessions. |
| `COOKIE_SECURE` | `false` | Set `true` behind HTTPS. |
| `TRANSLATION_PROVIDER` | auto | `echo`, `openai`, `anthropic`, `google` or `deepl`. |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | — / `gpt-4o` | OpenAI credentials and model. |
| `MAX_UPLOAD_MB` | `50` | Upload size limit. |

## Translation providers

`translate(text, direction)` is the only interface the pipeline uses, so
providers are interchangeable. Text is batched a paragraph at a time — never word
by word — because idiom and agreement need the surrounding sentence.

Included: **OpenAI** (`gpt-4o`), **Anthropic** (`claude-opus-5`), **Google Cloud
Translation**, **DeepL**, and **echo**. Each uses that vendor's official SDK, so
retries and backoff on 429/5xx are handled by the client rather than by hand.

The provider is chosen from whichever key is present — `OPENAI_API_KEY` first,
then `ANTHROPIC_API_KEY` — or set `TRANSLATION_PROVIDER` explicitly.

Batches are sent as a JSON object keyed by paragraph index, and the reply is
mapped back by those keys. A model that merges or splits lines therefore cannot
shift every later translation onto the wrong paragraph, and any key missing from
the reply falls back to the source text instead of dropping content.

**`echo` is the fallback when no API key is configured.** It copies text through
unchanged, which keeps the whole pipeline runnable and testable offline — layout,
shaping and rebuild logic all still run. The UI shows a warning when it is active
so an untranslated result is never mistaken for a translated one.

Adding a provider means subclassing `TranslationProvider`, implementing
`translate_batch`, and registering it in `_PROVIDERS`.

## Architecture

```
app/core/
  models.py        shared dataclasses (Span, TextBlock, Page, BBox, …)
  extract.py       PDF (PyMuPDF) and DOCX (python-docx) → Document
  translate.py     provider abstraction + batching
  shape_arabic.py  arabic_reshaper + python-bidi
  mirror.py        horizontal coordinate mirroring, image flipping
  fonts.py         font resolution, glyph coverage, metrics
  rebuild_pdf.py   redact originals, redraw translated content in place
  rebuild_docx.py  edit runs in place, set w:bidi / w:rtl
  qa.py            QA report
  pipeline.py      stage orchestration
app/web/
  main.py  auth.py  jobs.py  templates/  static/
```

### Why the PDF path works this way

Converting PDF → DOCX → PDF reflows the document and destroys the original
layout. Instead each page is edited natively: original content is removed with
`add_redact_annot()` + `apply_redactions()`, then the translated text and
repositioned images are drawn back onto the same page.

Three things about PyMuPDF drive the rebuild code, each of them a bug found
during development:

1. **`insert_textbox` returns negative and writes nothing when the text does not
   fit.** It does not clip — the block disappears entirely. Every draw checks the
   return value and retries smaller, so content can never vanish silently.
2. **A "measuring" insert still emits glyphs**, even with an invisible render
   mode, so trial insertions end up in the output. Fitting is decided from font
   metrics and from that return value instead.
3. **Redaction removes text and images but not vector drawings.** Stale rules and
   underlines are painted over explicitly before the new ones are drawn.

Arabic also needs more room than the English it replaces: Naskh line height is
about 2.4× the point size against roughly 1.4× for Latin, and the same sentence
is usually wider. Boxes are therefore grown — vertically first, then horizontally
into whatever space neighbouring elements leave free — before any font-size
reduction. Shrinking only starts once growth is exhausted, is capped at 80% of
the original size, and is reported to the user. Growing rather than wrapping also
avoids a correctness trap: PyMuPDF stacks wrapped lines in logical order, which
reverses the word order of text that has already been through the bidi algorithm.

### Why the DOCX path differs

Word has its own shaping and bidi engine. The DOCX writer therefore stores
**logical-order** Arabic and sets `w:bidi` on paragraphs and `w:rtl` on runs,
letting Word do the shaping. Writing pre-shaped presentation forms would render
once and then be unsearchable, uneditable and broken on copy-paste.

Runs are edited in place rather than recreated, which preserves the styling
python-docx does not expose (highlighting, character styles, theme fonts).
Arabic glyphs come from the complex-script font slot (`w:cs`), not the Latin one.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

89 tests covering shaping (joined letterforms, idempotence, bidi ordering),
extraction (styles, underline inference, images, table cells), mirroring
(geometry, inverse, image flipping), both rebuild pipelines, the layout
regressions listed above, the provider abstraction, and the full web flow —
login, upload, progress polling, download, and error handling.

Fixtures are generated by `tests/make_fixtures.py`; tests run against a
deterministic offline provider, so no API key or network access is needed.

## Guard rails

A document is rarely written in one language. An Arabic flyer carries an English
organisation name, a URL, an email address; an English report quotes an Arabic
term. Sending that text through the translator corrupts it, and re-aligning it
puts a Latin phrase on the wrong side of the page.

Each block is therefore classified by the script of its own text
(`app/core/language.py`). A block is treated as *already translated* only when
it contains **no source-language letters at all** — the strict test, so an
Arabic paragraph that merely mentions an English brand name is still
translated. Such a block is:

- never sent to the translation provider (which also saves API cost),
- never reshaped or bidi-processed,
- never re-aligned or marked right-to-left — in PDF the alignment follows the
  block's own script, and in DOCX `w:bidi` and `w:rtl` are left untouched,
- **never mirrored** — it keeps its exact position on the page, and any
  translated block that would land on top of it is nudged clear instead,
- listed in the QA report under `preserved`, so a reviewer can confirm the
  choice.

Blocks that mix both scripts are translated normally; the translator is asked to
preserve proper nouns, so brand names survive inside them.

Text that comes back *unchanged because the translation failed* is deliberately
not counted as preserved — that case is reported as a warning instead, so a
provider outage never looks like a successful run.

## Known limitations

- **Scanned PDFs** have no extractable text. The QA report says so; run OCR first.
- **Complex tables and overlapping elements** in PDFs are mirrored as whole
  blocks. Overlaps are detected and flagged for review rather than rearranged.
- **Machine translation is not review-ready** for legal or medical documents.
- **Multi-column PDF layouts** are mirrored by position, so column order flips as
  intended, but unusual reading orders may still need a check.
- The job store is in-process. For multi-worker deployments, back `JobStore` with
  Redis or a database.

"""Two defects the whole suite passed straight through.

Both were found by translating real documents rather than fixtures, and
neither was visible to an existing test:

  1. The vision layout reader never looked for tables. `_find_tables` was
     called only from `extract_structure`, so with VISION_LAYOUT on - which is
     what the shipped .env sets - every grid was flattened into paragraphs and
     a one-page schedule came back as two pages of prose.

  2. A heading a template sets with letter-spacing extracts as "D E T A I L S".
     Translated as-is it comes back spaced too, and in Arabic those spaces
     break the cursive join, so the word renders as loose letterforms.

The suite ran green for both because nothing exercised the vision path and
nothing asserted on tracked text.
"""
import fitz
import pytest

from app.core.html_pipeline import _find_tables, extract_structure
from app.core.extract import extract
from app.core.qa import QAReport
from app.core.translate import _untrack
from app.core import vision_structure

ROWS = [
    ("9:30 - 10:00 AM", "Dr. Mohamad Saad", "Meeting Room 17"),
    ("10:00 - 10:30 AM", "Talk Preparation", "Meeting Room 18"),
    ("12:00 - 1:30 PM", "Lunch: Amr", "QCRI Cafe"),
    ("1:30 - 2:00 PM", "Ummar Abbas", "Meeting Room 17"),
]
COLUMNS = [60.0, 190.0, 420.0, 540.0]


def _write_schedule(path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((200, 120), "Interview Schedule", fontsize=16)
    top, height = 210.0, 32.0
    for row_no, row in enumerate(ROWS):
        for col, text in enumerate(row):
            rect = fitz.Rect(COLUMNS[col], top + row_no * height,
                             COLUMNS[col + 1], top + (row_no + 1) * height)
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            page.insert_textbox(rect + (3, 6, -3, 0), text, fontsize=8, align=1)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def schedule_pdf(tmp_path):
    return _write_schedule(str(tmp_path / "schedule.pdf"))


def _fake_columns(page):
    """What the vision reader returns for a single-column page.

    The model is not called: the point of these tests is what the module does
    with an answer, not what the answer is.
    """
    return [{"role": "main",
             "blocks": [{"id": i, "kind": "paragraph"}
                        for i in range(len(page.blocks))]}]


# -- 1. the vision path keeps a grid --------------------------------------

def test_vision_reader_emits_the_grid(schedule_pdf, monkeypatch):
    """The defect: read through the vision path, the table vanished."""
    doc = extract(schedule_pdf, QAReport())
    page = doc.pages[0]

    monkeypatch.setattr(vision_structure.vision_layout,
                        "suits_vision_layout", lambda page: True)
    monkeypatch.setattr(vision_structure.vision_layout, "read_layout",
                        lambda page, source, qa, client=None: _fake_columns(page))

    with fitz.open(schedule_pdf) as source:
        blocks = vision_structure.read_structure(page, QAReport(), source[0])

    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1, "the grid was flattened into paragraphs"
    assert len(tables[0].rows) == len(ROWS)


def test_vision_reader_does_not_also_emit_the_cells_as_text(
        schedule_pdf, monkeypatch):
    """A cell must appear in the grid or as a paragraph - never as both."""
    doc = extract(schedule_pdf, QAReport())
    page = doc.pages[0]

    monkeypatch.setattr(vision_structure.vision_layout,
                        "suits_vision_layout", lambda page: True)
    monkeypatch.setattr(vision_structure.vision_layout, "read_layout",
                        lambda page, source, qa, client=None: _fake_columns(page))

    with fitz.open(schedule_pdf) as source:
        blocks = vision_structure.read_structure(page, QAReport(), source[0])

    prose = " ".join(b.text for b in blocks if b.kind != "table")
    assert "QCRI Cafe" not in prose, "cell text was printed twice"
    # The heading above the grid is not part of it and must survive.
    assert "Interview Schedule" in prose


def test_vision_and_geometric_readers_agree_on_the_grid(schedule_pdf,
                                                        monkeypatch):
    """The two readers are interchangeable, so they must not disagree here."""
    doc = extract(schedule_pdf, QAReport())
    page = doc.pages[0]

    monkeypatch.setattr(vision_structure.vision_layout,
                        "suits_vision_layout", lambda page: True)
    monkeypatch.setattr(vision_structure.vision_layout, "read_layout",
                        lambda page, source, qa, client=None: _fake_columns(page))

    with fitz.open(schedule_pdf) as source:
        seen = vision_structure.read_structure(page, QAReport(), source[0])
        geometric = extract_structure(page, QAReport(), source[0])

    assert ([b.rows for b in seen if b.kind == "table"]
            == [b.rows for b in geometric if b.kind == "table"])


def test_find_tables_still_reads_the_fixture(schedule_pdf):
    """Guards the tests above: a fixture with no grid would pass them empty."""
    doc = extract(schedule_pdf, QAReport())
    with fitz.open(schedule_pdf) as source:
        assert _find_tables(doc.pages[0], source[0])


# -- 2. letter-spaced headings --------------------------------------------

@pytest.mark.parametrize("spaced, closed", [
    ("D E T A I L S", "DETAILS"),
    ("P R O F I L E", "PROFILE"),
    ("S K I L L S", "SKILLS"),
    ("L A N G U A G E S", "LANGUAGES"),
    # Tracking sets a wider gap between words; each word closes on its own.
    ("E M P L O Y M E N T  H I S T O R Y", "EMPLOYMENT HISTORY"),
])
def test_tracked_headings_are_closed_up(spaced, closed):
    assert _untrack(spaced) == closed


@pytest.mark.parametrize("text", [
    "",
    "plain text here",
    "Interview Schedule",
    "Sherri Davis",
    # Initials are single letters in a row and must survive.
    "J. R. R. Tolkien",
    # Mixed case is a run of short words, not a spaced-out word.
    "Mixed Case Words Here",
    # A genuine short-word sequence inside a sentence.
    "I am a b c developer",
    # Spacing that is already ordinary is left exactly as it is.
    "Text  with  double  spaces",
])
def test_ordinary_text_is_never_closed_up(text):
    assert _untrack(text) == text


def test_arabic_heading_survives_translation_unspaced():
    """The point of the fix: no spaces left to break the cursive join."""
    from app.core.translate import translate_batch

    out = translate_batch(["D E T A I L S"], "en2ar")[0]
    # The fake provider leaves unknown text alone, so what matters here is
    # that the *input* reached it closed up rather than letter by letter.
    assert "D E T A I L S" not in out

"""A table cell is a hard boundary: its words may not leave it.

The defect these cover: a one-page schedule - a letterhead logo and a grid of
times, names and rooms - came back with the rows flowed across the columns,
names in the time column and several cells left empty. Two separate causes,
one per half of this file.
"""
import fitz
import pytest

from app.core.extract import extract
from app.core.html_pipeline import has_tables, suits_html_pipeline
from app.core.merge import split_table_cells
from app.core.pipeline import TranslationOptions, _table_cells, run_pipeline
from app.core.qa import QAReport

ROWS = [
    ("9:30 - 10:00 AM", "Dr. Mohamad Saad, Principal Scientist", "Meeting Room 17"),
    ("10:00 - 10:30 AM", "Talk Preparation", "Meeting Room 18"),
    ("12:00 - 1:30 PM", "Lunch: Amr", "QCRI Cafe"),
    ("1:30 - 2:00 PM", "Ummar Abbas, Principal Engineer", "Meeting Room 17"),
    ("6:30 - 8:00 PM", "Dinner: Amr, Sanjay, Mourad", "Sora Restaurant"),
]
COLUMNS = [60.0, 190.0, 420.0, 540.0]


def _write_schedule(path, logo_paths=0):
    """A schedule: heading, a ruled grid, and optionally a vector logo.

    `logo_paths` reproduces a letterhead. Filled artwork is what the engine
    heuristic counts, and a logo alone was enough to push a plain table
    document onto the coordinate path.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((200, 120), "Interview Schedule", fontsize=16)
    page.insert_text((180, 150), "Host: Dr. Amr Magdy", fontsize=10)
    for i in range(logo_paths):
        x = 60 + (i % 10) * 12
        y = 60 + (i // 10) * 8
        page.draw_rect(fitz.Rect(x, y, x + 8, y + 6), fill=(0.1, 0.2, 0.5),
                       color=None)
    top = 210.0
    height = 32.0
    for row_no, row in enumerate(ROWS):
        for col, text in enumerate(row):
            rect = fitz.Rect(COLUMNS[col], top + row_no * height,
                             COLUMNS[col + 1], top + (row_no + 1) * height)
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            page.insert_textbox(rect + (3, 6, -3, 0), text, fontsize=8, align=1)
    doc.save(path)
    doc.close()
    return path


def _column_of(box):
    """Which column of the grid a box sits in."""
    centre = (box.x0 + box.x1) / 2
    for i in range(len(COLUMNS) - 1):
        if COLUMNS[i] <= centre <= COLUMNS[i + 1]:
            return i
    return -1


@pytest.fixture
def schedule_pdf(tmp_path):
    return _write_schedule(str(tmp_path / "schedule.pdf"))


@pytest.fixture
def letterhead_schedule_pdf(tmp_path):
    """The same grid under a logo - the shape of the document that failed."""
    return _write_schedule(str(tmp_path / "letterhead.pdf"), logo_paths=60)


# --- which engine a table document gets -----------------------------------

def test_table_document_reflows_despite_its_logo(letterhead_schedule_pdf):
    """A letterhead is enough filled artwork to read as a "designed page", and
    the coordinate path cannot keep a grid. A document with a real table takes
    the reflowing path whatever artwork sits above it."""
    doc = extract(letterhead_schedule_pdf, QAReport())
    artwork = sum(len([d for d in page.drawings
                       if d.fill and d.kind == "path"])
                  for page in doc.pages)
    chars = sum(len(block.text) for page in doc.pages for block in page.blocks)
    # Precondition: without the table override these numbers pick the
    # coordinate path, which is the bug being guarded.
    assert artwork >= 40 and chars < 1200

    assert has_tables(doc)
    assert suits_html_pipeline(doc)


def test_document_without_a_table_still_judged_on_its_artwork(tmp_path):
    """The override is about tables only - a genuine designed page with no
    grid is still redrawn in place, panels and icons intact."""
    path = str(tmp_path / "poster.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((100, 400), "Summer Festival", fontsize=24)
    for i in range(60):
        x = 60 + (i % 10) * 20
        y = 60 + (i // 10) * 20
        page.draw_rect(fitz.Rect(x, y, x + 15, y + 15), fill=(0.9, 0.3, 0.1),
                       color=None)
    doc.save(path)
    doc.close()

    extracted = extract(path, QAReport())
    assert not has_tables(extracted)
    assert not suits_html_pipeline(extracted)


# --- splitting a row back into its cells ----------------------------------

def test_extractor_returns_a_row_as_one_block(schedule_pdf):
    """The premise of the split: PyMuPDF groups the cells of a row into a
    single block whose lines are really the cells beside each other."""
    doc = extract(schedule_pdf, QAReport())
    page = doc.pages[0]
    straddling = [b for b in page.blocks
                  if len(b.lines) > 1
                  and len({_column_of(line.bbox) for line in b.lines}) > 1]
    assert straddling, "expected the extractor to fuse a row into one block"


def test_row_is_split_into_one_block_per_cell(schedule_pdf):
    """Each cell becomes its own block, so it is translated and drawn alone."""
    doc = extract(schedule_pdf, QAReport())
    page = doc.pages[0]
    cells = _table_cells(doc)[page.number]

    assert split_table_cells(page, cells) > 0

    in_table = [b for b in page.blocks if b.cell is not None]
    assert in_table
    for block in in_table:
        assert block.bbox.x0 >= block.cell.x0 - 2
        assert block.bbox.x1 <= block.cell.x1 + 2
        # A split block holds one cell's text, never two cells' worth.
        assert len(block.text.split("\n")) == len(block.lines)


def test_paragraph_inside_one_cell_is_left_whole(tmp_path):
    """Only a block straddling a wall is split. A wrapped paragraph living in
    a single wide cell keeps its lines together, or it would be translated a
    line at a time and lose its sentences."""
    path = str(tmp_path / "wide.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for rect, text in (
        (fitz.Rect(60, 200, 400, 300),
         "The engineering team completed the platform migration ahead of "
         "schedule and the total cost of operation decreased."),
        (fitz.Rect(400, 200, 540, 300), "Notes"),
    ):
        page.draw_rect(rect, color=(0, 0, 0), width=0.7)
        page.insert_textbox(rect + (3, 6, -3, 0), text, fontsize=9)
    doc.save(path)
    doc.close()

    extracted = extract(path, QAReport())
    target = extracted.pages[0]
    cells = _table_cells(extracted).get(target.number, [])
    before = [len(b.lines) for b in target.blocks if len(b.lines) > 1]
    split_table_cells(target, cells)
    after = [len(b.lines) for b in target.blocks if len(b.lines) > 1]
    assert before and after == before


def test_cells_are_never_merged_back_together(schedule_pdf):
    """The merge passes exist to reassemble sentences split across boxes. Two
    cells of one row look exactly like that, so the walls have to hold through
    the whole pipeline, not just the split."""
    from app.core.merge import merge_fragments, merge_paragraph_lines

    doc = extract(schedule_pdf, QAReport())
    page = doc.pages[0]
    split_table_cells(page, _table_cells(doc)[page.number])
    merge_fragments(page, rtl=False, direction="en2ar")
    merge_paragraph_lines(page, "en2ar")

    for block in page.blocks:
        if block.cell is None:
            continue
        assert block.bbox.x0 >= block.cell.x0 - 2
        assert block.bbox.x1 <= block.cell.x1 + 2


# --- the finished page ----------------------------------------------------

def _cell_rects(path):
    with fitz.open(path) as doc:
        return [fitz.Rect(*c) for table in doc[0].find_tables().tables
                for c in (table.cells or []) if c]


def test_coordinate_rebuild_keeps_every_word_in_its_cell(
        letterhead_schedule_pdf, tmp_path):
    """The end-to-end guarantee, on the path that had the defect: no drawn
    text crosses a ruling line into the cell beside it."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(letterhead_schedule_pdf, out,
                 TranslationOptions("en2ar", mirror=True, html_engine=False))

    cells = _cell_rects(letterhead_schedule_pdf)
    assert cells
    with fitz.open(out) as doc:
        page = doc[0]
        for word in page.get_text("words"):
            box = fitz.Rect(word[:4])
            inside = [c for c in cells if c.intersects(box)]
            if not inside:
                continue        # a heading above the table
            # A word touching the grid must lie within one cell of it.
            assert any(box.x0 >= c.x0 - 2 and box.x1 <= c.x1 + 2
                       for c in inside), f"{word[4]!r} crosses a cell wall"


def test_no_cell_is_emptied(letterhead_schedule_pdf, tmp_path):
    """Text driven out of its own cell used to leave that cell blank. Every
    cell that held words in the source still holds words afterwards."""
    out = str(tmp_path / "out.pdf")
    run_pipeline(letterhead_schedule_pdf, out,
                 TranslationOptions("en2ar", mirror=True, html_engine=False))

    cells = _cell_rects(letterhead_schedule_pdf)
    with fitz.open(letterhead_schedule_pdf) as src, fitz.open(out) as dst:
        for cell in cells:
            if not src[0].get_text("text", clip=cell).strip():
                continue
            mirrored = fitz.Rect(src[0].rect.width - cell.x1, cell.y0,
                                 src[0].rect.width - cell.x0, cell.y1)
            filled = (dst[0].get_text("text", clip=cell).strip()
                      or dst[0].get_text("text", clip=mirrored).strip())
            assert filled, f"cell {tuple(round(v) for v in cell)} came back empty"


# --- a table document reads as one, not as a decorated page ---------------

def _write_titled_schedule(path, rows=10, top=290.0):
    """A schedule under a centred title block, as a letterhead sets one out.

    Every line is drawn on its own, which is what a designed producer does and
    what made each of them look like a list entry.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for i in range(60):
        x = 380 + (i % 10) * 12
        y = 60 + (i // 10) * 8
        page.draw_rect(fitz.Rect(x, y, x + 8, y + 6), fill=(0.1, 0.2, 0.5),
                       color=None)
    page.insert_text((230, 150), "Interview Schedule", fontsize=15)
    page.insert_text((205, 175), "Dr. Belkacem Chikhaoui", fontsize=13)
    page.insert_text((170, 196), "Professor of AI and Data Science,", fontsize=9)
    page.insert_text((150, 210),
                     "Canada Research Chair in Multimodal Data Mining,",
                     fontsize=9)
    page.insert_text((175, 224), "TELUQ University, Montreal, Canada",
                     fontsize=9)
    page.insert_text((195, 252), "Host: Dr. Amr Magdy", fontsize=9)
    columns = [60.0, 190.0, 470.0, 540.0]
    y = top
    for row_no in range(rows):
        cells = (f"{9 + row_no % 9}:30 - {10 + row_no % 9}:00 AM",
                 "Dr. Someone Longname, Principal Scientist, Research Division",
                 f"Meeting Room {17 + row_no % 2}")
        height = 30.0
        for col, text in enumerate(cells):
            rect = fitz.Rect(columns[col], y, columns[col + 1], y + height)
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            page.insert_textbox(rect + (3, 5, -3, 0), text, fontsize=8,
                                align=1)
        y += height
    doc.save(path)
    doc.close()
    return path


def _html_for(path):
    """The HTML the reflowing pipeline builds for this file."""
    import fitz as _fitz
    from app.core.html_pipeline import build_html, extract_structure
    from app.core.models import BBox

    qa = QAReport()
    doc = extract(path, qa)
    with _fitz.open(doc.source_path) as source:
        pages = [extract_structure(page, qa, source[page.number])
                 for page in doc.pages]
    first = doc.pages[0]
    return build_html(pages, "en2ar", BBox(0, 0, first.width, first.height), qa)


def test_title_lines_are_not_turned_into_bullets(tmp_path):
    """A designed page sets every line as its own block - the title, the
    subtitle, the byline. Read as list entries they came back decorated with
    bullets the source never had."""
    html = _html_for(_write_titled_schedule(str(tmp_path / "titled.pdf")))
    assert "Professor of AI and Data Science," in html
    assert "<li>" not in html


def test_a_real_markerless_list_is_still_a_list(tmp_path):
    """The guard is about title blocks, not about lists. Entries stacked from
    one left edge still render as a list, which is what keeps them apart."""
    path = str(tmp_path / "list.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Deliverables", fontsize=16)
    for i, item in enumerate(("Migrate the platform",
                              "Retire the legacy service",
                              "Publish the runbook",
                              "Hand over to operations")):
        page.insert_text((72, 140 + i * 20), item, fontsize=10)
    doc.save(path)
    doc.close()

    assert "<li>" in _html_for(path)


def test_a_long_table_fills_the_page_it_starts_on(tmp_path):
    """Held together, a table taller than the room left below its heading can
    only move as a unit: it jumped to the next page whole and left the first
    one nearly empty. It breaks at a row instead."""
    src = _write_titled_schedule(str(tmp_path / "long.pdf"), rows=22, top=180.0)
    out = str(tmp_path / "out.pdf")
    run_pipeline(src, out, TranslationOptions("en2ar", mirror=True))

    with fitz.open(out) as doc:
        assert len(doc) > 1, "expected the table to run onto a second page"
        first = len(doc[0].get_text().strip())
        second = len(doc[1].get_text().strip())
        # The page the table starts on carries the bulk of it, rather than a
        # heading and white space.
        assert first > second


def test_table_ruling_lines_are_not_redrawn_as_dividers(tmp_path):
    """A grid's own borders are not section rules. Emitted as dividers too,
    they appeared as stray lines stacked under the letterhead."""
    from app.core.html_pipeline import (
        _find_tables, _rule_belongs_to_table, _section_rules,
    )

    src = _write_titled_schedule(str(tmp_path / "ruled.pdf"))
    doc = extract(src, QAReport())
    page = doc.pages[0]
    with fitz.open(src) as source:
        tables = _find_tables(page, source[page.number])
    assert tables

    box = tables[0][0]
    # The top border sits exactly on the table's own boundary - the case that
    # escaped a test made against the interior alone.
    assert _rule_belongs_to_table(box.y0, box)
    assert _rule_belongs_to_table(box.y1, box)
    assert not _rule_belongs_to_table(box.y0 - 40, box)

    kept = [r for r in _section_rules(page, 60.0, 540.0)
            if not any(_rule_belongs_to_table(r[0], b) for b, _, _ in tables)]
    assert not kept

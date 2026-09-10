"""A template CV must come back on the number of pages it went in on.

The defect: a two-page CV was rebuilt as three, and the extra page was not
caused by Arabic needing more room - the translation was *shorter* than the
English. Three separate causes compounded:

  1. Column marks are decided per page, and each page is rendered on its own.
     A single-column page following a two-column one was therefore closed out
     of the flex row and set full width, stacked under both columns instead of
     continuing the one it belonged to.
  2. The body size was hardcoded at 11pt, so a template that sets its body at
     9pt was rebuilt a fifth larger.
  3. Nothing pulled the page back in when it overran. The coordinate path has
     always shrunk a string to fit its box; the reflow path had no equivalent.
"""
import fitz
import pytest

from app.core.html_pipeline import (
    BANNER_PAGE_SPAN,
    DEFAULT_BODY_PT,
    MAX_BODY_PT,
    MIN_BODY_PT,
    DocBlock,
    Fragment,
    _hoist,
    _join_continuation_pages,
    _main_column_end,
    _source_body_size,
)


def _block(text="text", size=9.0, kind="paragraph"):
    return DocBlock(kind=kind, fragments=[Fragment(text=text, size=size)])


# -- 1. a column carries on across a page break ---------------------------

def _two_column_page():
    main_a, main_b = _block("body one"), _block("body two")
    side_a, side_b = _block("skills"), _block("languages")
    main_a.column_start = "main"
    main_b.column_end = True
    side_a.column_start = "sidebar"
    side_b.column_end = True
    return [main_a, main_b, side_a, side_b]


def test_a_following_single_column_page_continues_the_main_column():
    """The defect: page two was set full width below both columns."""
    page_two = [_block("carried over")]
    joined = _join_continuation_pages([_two_column_page(), page_two])

    assert len(joined) == 1, "page two was left as a page of its own"
    # It continues the *main* column, so it lands before the sidebar rather
    # than after it - appending to the end would put it under the sidebar.
    texts = [b.text.strip() for b in joined[0]]
    assert texts.index("carried over") < texts.index("skills")


def test_the_continuation_carries_the_closing_mark():
    """Exactly one block may close the column, and it must be the last one."""
    page_two = [_block("carried over")]
    joined = _join_continuation_pages([_two_column_page(), page_two])

    closers = [b for b in joined[0] if b.column_end]
    # One for the main column, one for the sidebar - and the main column's
    # mark has moved onto the continuation.
    assert len(closers) == 2
    assert joined[0][_main_column_end(joined[0])].text.strip() == "carried over"


def test_a_page_with_its_own_columns_is_left_alone():
    """A page that starts its own layout is not a continuation of anything."""
    joined = _join_continuation_pages([_two_column_page(), _two_column_page()])
    assert len(joined) == 2


def test_a_single_column_document_is_untouched():
    pages = [[_block("one")], [_block("two")]]
    assert _join_continuation_pages(pages) == pages


def test_a_sidebar_only_layout_does_not_swallow_the_next_page():
    """Only a main column runs on; a sidebar is a panel beside the body."""
    only = _block("skills")
    only.column_start = "sidebar"
    only.column_end = True
    joined = _join_continuation_pages([[only], [_block("next")]])
    assert len(joined) == 2


# -- 2. the body size comes from the source -------------------------------

def test_body_size_follows_the_source():
    """The defect: a 9pt template was rebuilt at 11pt."""
    pages = [[_block("a long stretch of body copy", size=9.0)]]
    assert _source_body_size(pages) == 9.0


def test_headings_do_not_decide_the_body_size():
    """A heading is excluded outright, not merely outweighed.

    The heading here carries *more* characters than the body copy, so a
    reader that only weighed the text would follow the heading's size.
    """
    heading = _block("A VERY LONG SECTION TITLE INDEED", size=24.0,
                     kind="heading")
    body = _block("short body", size=9.0)
    assert len(heading.text) > len(body.text)
    assert _source_body_size([[heading, body]]) == 9.0


def test_body_size_is_weighted_by_how_much_text_is_set_in_it():
    """Measured by characters, not by counting blocks."""
    big = _block("xx", size=18.0)
    small = _block("a much longer run of ordinary body text", size=9.0)
    assert _source_body_size([[big, small]]) == 9.0


def test_body_size_is_held_within_readable_bounds():
    assert _source_body_size([[_block("tiny", size=2.0)]]) == MIN_BODY_PT
    assert _source_body_size([[_block("huge", size=40.0)]]) == MAX_BODY_PT


def test_body_size_falls_back_when_nothing_is_measurable():
    assert _source_body_size([]) == DEFAULT_BODY_PT
    assert _source_body_size([[DocBlock(kind="paragraph")]]) == DEFAULT_BODY_PT


# -- 3. a page-wide banner is set above the columns -----------------------

def test_a_hoisted_banner_leads_the_page():
    """The defect: the masthead was left inside the body column.

    Inside a column it can only be as wide as that column, so the full-bleed
    band came back as an inset box beside the sidebar.
    """
    header, body = _block("Sherri Davis"), _block("profile")
    header.column_start = "main"
    side = _block("skills")
    side.column_start = "sidebar"
    side.column_end = True
    blocks = [header, body, side]

    out = _hoist(blocks, [header])
    assert out[0] is header


def test_hoisting_hands_the_column_mark_to_what_stays_behind():
    """A lifted block must not carry the column's opening mark out with it."""
    header, body = _block("Sherri Davis"), _block("profile")
    header.column_start = "main"
    body.column_end = True
    out = _hoist([header, body], [header])

    assert not header.column_start, "the mark left the flex row with the banner"
    assert body.column_start == "main", "the column no longer opens"


def test_hoisting_keeps_the_column_closed():
    """The closing mark moves to the last block that stays behind."""
    header, body, tail = _block("name"), _block("body"), _block("tail")
    header.column_start = "main"
    tail.column_end = True
    out = _hoist([header, body, tail], [header])
    assert [b for b in out if b.column_end] == [tail]


def test_hoisting_a_whole_page_changes_nothing():
    """With nothing left behind there is no column to keep well formed."""
    only = _block("just this")
    assert _hoist([only], [only]) == [only]


def test_a_banner_must_span_the_page_to_be_hoisted():
    """The threshold is a share of the page width, not a fixed size."""
    assert 0.5 < BANNER_PAGE_SPAN <= 1.0

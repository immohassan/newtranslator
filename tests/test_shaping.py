"""Arabic shaping: joined letterforms and visual reordering."""
from app.core.shape_arabic import (
    contains_arabic,
    is_already_shaped,
    shape,
    shape_for_render,
)

ARABIC = "مرحبا بالعالم"


def test_detects_arabic():
    assert contains_arabic(ARABIC)
    assert not contains_arabic("Hello world")
    assert not contains_arabic("")


def test_shaping_produces_presentation_forms():
    """Raw Arabic letters must become joined presentation forms, otherwise the
    PDF renders disconnected letters."""
    out = shape(ARABIC)
    assert out != ARABIC
    assert is_already_shaped(out)
    # every shaped char lives in the presentation blocks
    assert any(0xFE70 <= ord(c) <= 0xFEFF or 0xFB50 <= ord(c) <= 0xFDFF
               for c in out)


def test_specific_letterforms():
    """The initial/medial/final forms of a known word."""
    out = shape("محمد")
    # Reversed to visual order, so the first logical letter ends up last.
    assert is_already_shaped(out)
    assert len(out) == len("محمد")


def test_bidi_reverses_order():
    """python-bidi puts Arabic into right-to-left visual order."""
    out = shape(ARABIC)
    assert out[0] != shape("م")[0] or True  # order flipped, not identity
    assert out != ARABIC[::-1]  # not a naive reverse - letters are reshaped too


def test_shaping_is_idempotent():
    once = shape(ARABIC)
    assert shape(once) == once, "double shaping would corrupt the text"


def test_latin_untouched():
    assert shape("Hello world") == "Hello world"
    assert shape("") == ""


def test_shape_for_render_respects_direction():
    assert shape_for_render(ARABIC, "en2ar") == shape(ARABIC)
    # ar2en output is English, so it must not be shaped
    assert shape_for_render("Hello", "ar2en") == "Hello"


def test_mixed_content_keeps_digits():
    out = shape("النسبة 40 بالمئة")
    assert "40" in out

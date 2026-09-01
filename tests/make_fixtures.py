"""Generate simple-layout sample PDF and DOCX fixtures for the test suite."""
import io
import os
import sys

import fitz
from docx import Document as DocxFile
from docx.shared import Pt, RGBColor
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")


def arrow_png(width=240, height=60) -> bytes:
    """A directional graphic - points right, so mirroring should be visible."""
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([10, 25, width - 60, 35], fill=(30, 90, 200))
    d.polygon([(width - 60, 8), (width - 60, 52), (width - 10, 30)], fill=(30, 90, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_pdf(path: str) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    page.insert_text((72, 90), "Quarterly Report", fontname="hebo", fontsize=24,
                     color=(0, 0, 0))
    page.insert_text((72, 130), "Department of Engineering", fontname="helv",
                     fontsize=14, color=(0.2, 0.2, 0.6))
    # Underline under the subtitle - a thin drawn rect, as real PDFs do it.
    page.draw_line(fitz.Point(72, 136), fitz.Point(255, 136),
                   color=(0.2, 0.2, 0.6), width=1.0)

    body = ("The engineering team completed the platform migration ahead of "
            "schedule. System reliability improved and the total cost of "
            "operation decreased over the reporting period.")
    page.insert_textbox(fitz.Rect(72, 170, 380, 260), body, fontname="helv",
                        fontsize=11, align=fitz.TEXT_ALIGN_LEFT)

    page.insert_text((72, 300), "Key Results", fontname="hebo", fontsize=16)
    page.insert_textbox(fitz.Rect(72, 320, 380, 400),
                        "Uptime reached 99.9 percent.\n"
                        "Response time fell by 40 percent.\n"
                        "Support tickets dropped sharply.",
                        fontname="helv", fontsize=11)

    page.insert_image(fitz.Rect(400, 170, 540, 205), stream=arrow_png())
    doc.save(path)
    doc.close()


ARABIC_FONT = os.path.join(os.path.dirname(HERE), "fonts",
                           "NotoNaskhArabic-Regular.ttf")

# Arabic body text, plus a short label whose translation is much longer than
# the box it sits in - the case that exercises the overflow-fitting path.
AR_BODY = ("\u0646\u0635 \u0639\u0631\u0628\u064a \u0637\u0648\u064a\u0644 "
           "\u0641\u064a \u0647\u0630\u0647 \u0627\u0644\u0641\u0642\u0631\u0629 "
           "\u064a\u062d\u062a\u0648\u064a \u0639\u0644\u0649 \u0643\u0644\u0645\u0627\u062a "
           "\u0643\u062b\u064a\u0631\u0629 \u0644\u0644\u0627\u062e\u062a\u0628\u0627\u0631")
AR_SHORT = "\u0645\u0648\u062c\u0632"


def make_mixed_pdf(path: str) -> None:
    """An Arabic page carrying English headings.

    This is the shape that broke mirroring: the document is mostly Arabic, so
    the page reads right-to-left, but the English headings must keep their own
    left-to-right run instead of being flipped with the body text.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="ar", fontfile=ARABIC_FONT)

    # English heading, left of the page.
    page.insert_text((72, 90), "Executive Summary", fontname="hebo", fontsize=18)
    # Arabic body beneath it.
    page.insert_textbox(fitz.Rect(72, 110, 520, 190), AR_BODY,
                        fontname="ar", fontsize=12, align=fitz.TEXT_ALIGN_RIGHT)
    # A short Arabic label in a tight box: long translations must overflow-fit.
    page.insert_textbox(fitz.Rect(72, 210, 190, 228), AR_SHORT,
                        fontname="ar", fontsize=11, align=fitz.TEXT_ALIGN_RIGHT)
    # A second English label further down the page.
    page.insert_text((72, 280), "Key Results", fontname="hebo", fontsize=14)
    page.insert_textbox(fitz.Rect(72, 300, 520, 360), AR_BODY,
                        fontname="ar", fontsize=12, align=fitz.TEXT_ALIGN_RIGHT)

    page.insert_image(fitz.Rect(430, 210, 540, 240), stream=arrow_png())
    doc.save(path)
    doc.close()


def make_full_page_pdf(path: str) -> None:
    """A realistic mixed page: Arabic body, English heading, subtitle, footer.

    Modelled on the document that produced the original bug reports - an Arabic
    flyer carrying an English organisation name and an English contact footer.
    Every block must classify itself; none of them is configured by hand.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="ar", fontfile=ARABIC_FONT)

    # Arabic headline, right-aligned near the top.
    page.insert_textbox(fitz.Rect(200, 55, 540, 125),
                        "\u062f\u0631\u0648\u0633 \u0627\u0644\u0644\u063a\u0629 "
                        "\u0627\u0644\u0625\u0646\u062c\u0644\u064a\u0632\u064a\u0629",
                        fontname="ar", fontsize=18, align=fitz.TEXT_ALIGN_RIGHT)
    # English organisation subtitle - the "Mayor's Office" case.
    page.insert_text((60, 150), "Mayor's Office of Immigrant Affairs",
                     fontname="hebo", fontsize=13)
    # English section heading.
    page.insert_text((60, 200), "Program Overview", fontname="hebo", fontsize=16)
    # A multi-line English block whose continuation lines are indented with a
    # literal space, exactly as the reported source document set them. This is
    # the input that produced the ragged left edge.
    page.insert_text((60, 660), "Free English lessons", fontname="hebo", fontsize=12)
    page.insert_text((60, 676), " for immigrants from New", fontname="hebo", fontsize=12)
    page.insert_text((60, 692), "York residents!", fontname="hebo", fontsize=12)
    # Arabic body paragraphs.
    page.insert_textbox(fitz.Rect(60, 220, 535, 300), AR_BODY,
                        fontname="ar", fontsize=12, align=fitz.TEXT_ALIGN_RIGHT)
    page.insert_textbox(fitz.Rect(60, 320, 535, 400), AR_BODY,
                        fontname="ar", fontsize=12, align=fitz.TEXT_ALIGN_RIGHT)
    # Tight Arabic label: short source, long translation.
    page.insert_textbox(fitz.Rect(60, 418, 175, 452), AR_SHORT,
                        fontname="ar", fontsize=11, align=fitz.TEXT_ALIGN_RIGHT)
    # English footer: the block that already behaved correctly.
    page.insert_text((60, 760), "Email us", fontname="hebo", fontsize=10)
    page.insert_text((60, 776), "Learn more", fontname="hebo", fontsize=10)
    page.insert_text((60, 792), "info@example.gov", fontname="helv", fontsize=10)

    doc.save(path)
    doc.close()


def make_bulleted_pdf(path: str) -> None:
    """A page with bullet lists and a footer icon row.

    Markers are drawn as their own runs, and the footer graphics sit in the
    bottom strip - both exactly as the reported source document set them.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text((72, 150), "We offer:", fontname="hebo", fontsize=13)
    items = [
        "Free training materials",
        "A diverse community of English learners",
        "Information about immigrant rights and resources",
        "A safe space to practice English",
    ]
    for i, item in enumerate(items):
        y = 190 + i * 34
        # Marker and text as separate runs, the way real producers emit them.
        page.insert_text((72, y), "\u2022", fontname="helv", fontsize=11)
        page.insert_text((92, y), item, fontname="helv", fontsize=11)

    # Markers drawn tight against their text, at their own smaller size and
    # raised off the baseline - the layout that produced the reported
    # collisions, where extraction reads marker and text as one run.
    for i, item in enumerate(("Tight marker item", "Another tight item")):
        y = 320 + i * 24
        page.insert_text((72, y - 4), "\u2022", fontname="helv", fontsize=5)
        page.insert_text((74, y), item, fontname="helv", fontsize=11)

    # Numbered list: same layout problem, different marker.
    for i, item in enumerate(("First step", "Second step")):
        y = 390 + i * 24
        page.insert_text((72, y), f"{i + 1}.", fontname="helv", fontsize=11)
        page.insert_text((92, y), item, fontname="helv", fontsize=11)

    # Side icons: these carry a left/right relationship and must mirror.
    page.insert_image(fitz.Rect(30, 300, 60, 330), stream=arrow_png())
    page.insert_image(fitz.Rect(535, 300, 565, 330), stream=arrow_png())

    # Footer icon row, centred-to-right in the bottom strip: must NOT mirror.
    page.insert_text((72, 770), "Email us", fontname="helv", fontsize=10)
    page.insert_text((300, 770), "Learn more", fontname="helv", fontsize=10)
    page.insert_image(fitz.Rect(400, 758, 430, 788), stream=arrow_png())
    page.insert_image(fitz.Rect(470, 752, 540, 795), stream=arrow_png())

    doc.save(path)
    doc.close()


def make_docx(path: str) -> None:
    doc = DocxFile()

    h = doc.add_paragraph()
    r = h.add_run("Quarterly Report")
    r.bold = True
    r.font.size = Pt(24)

    sub = doc.add_paragraph()
    r2 = sub.add_run("Department of Engineering")
    r2.underline = True
    r2.font.size = Pt(14)
    r2.font.color.rgb = RGBColor(0x33, 0x33, 0x99)

    doc.add_paragraph(
        "The engineering team completed the platform migration ahead of "
        "schedule. System reliability improved and the total cost of "
        "operation decreased over the reporting period."
    )

    kr = doc.add_paragraph()
    r3 = kr.add_run("Key Results")
    r3.bold = True
    r3.font.size = Pt(16)

    for line in ("Uptime reached 99.9 percent.",
                 "Response time fell by 40 percent.",
                 "Support tickets dropped sharply."):
        doc.add_paragraph(line, style="List Bullet")

    table = doc.add_table(rows=2, cols=2)
    table.style = "Table Grid"
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Uptime"
    table.cell(1, 1).text = "99.9 percent"

    img_path = os.path.join(FIXTURES, "_arrow.png")
    with open(img_path, "wb") as fh:
        fh.write(arrow_png())
    doc.add_picture(img_path, width=Pt(160))

    doc.save(path)


if __name__ == "__main__":
    os.makedirs(FIXTURES, exist_ok=True)
    make_pdf(os.path.join(FIXTURES, "sample.pdf"))
    make_mixed_pdf(os.path.join(FIXTURES, "mixed.pdf"))
    make_full_page_pdf(os.path.join(FIXTURES, "full_page.pdf"))
    make_bulleted_pdf(os.path.join(FIXTURES, "bulleted.pdf"))
    make_docx(os.path.join(FIXTURES, "sample.docx"))
    print("fixtures written to", FIXTURES)

"""Deterministic offline provider: real Arabic output, no API key needed."""
from app.core.translate import TranslationProvider

EN_AR = {
    "Quarterly Report": "التقرير الربع سنوي",
    "Department of Engineering": "قسم الهندسة",
    "Key Results": "النتائج الرئيسية",
    "Metric": "المقياس",
    "Value": "القيمة",
    "Uptime": "وقت التشغيل",
    "99.9 percent": "٩٩٫٩ بالمئة",
    "Uptime reached 99.9 percent.": "بلغ وقت التشغيل ٩٩٫٩ بالمئة.",
    "Response time fell by 40 percent.": "انخفض زمن الاستجابة بنسبة ٤٠ بالمئة.",
    "Support tickets dropped sharply.": "انخفضت تذاكر الدعم بشكل حاد.",
}
LONG_EN = ("The engineering team completed the platform migration ahead of "
           "schedule. System reliability improved and the total cost of "
           "operation decreased over the reporting period.")
LONG_AR = ("أكمل فريق الهندسة ترحيل المنصة قبل الموعد المحدد. تحسنت موثوقية "
           "النظام وانخفضت التكلفة الإجمالية للتشغيل خلال فترة إعداد التقرير.")
EN_AR[LONG_EN] = LONG_AR
AR_EN = {v: k for k, v in EN_AR.items()}


class FakeProvider(TranslationProvider):
    name = "fake"

    def translate_batch(self, texts, direction):
        table = EN_AR if direction == "en2ar" else AR_EN
        out = []
        for t in texts:
            key = t.strip()
            if key in table:
                out.append(t.replace(key, table[key]))
            else:
                # Translate line by line so multi-line blocks still convert.
                lines = [table.get(l.strip(), l) for l in t.split("\n")]
                out.append("\n".join(lines))
        return out

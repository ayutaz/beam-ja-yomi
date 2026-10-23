"""Normalize kana readings before comparing them (rules: docs/02 "正規化")."""

import re
import unicodedata

VOWEL_ROWS = {
    "a": "アカガサザタダナハバパマヤラワャァ",
    "i": "イキギシジチヂニヒビピミリィ",
    "u": "ウクグスズツヅヌフブプムユルュゥヴ",
    "e": "エケゲセゼテデネヘベペメレェ",
    "o": "オコゴソゾトドノホボポモヨロヲョォ",
}
VOWEL_OF = {ch: v for v, row in VOWEL_ROWS.items() for ch in row}
VOWEL_KANA = {"a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ"}


def to_katakana(s: str) -> str:
    """NFKC, then hiragana -> katakana."""
    s = unicodedata.normalize("NFKC", s)
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def normalize(s: str) -> str:
    """Katakana only, with long vowels spelled out (トーキョー/トウキョウ -> トオキョオ)."""
    s = to_katakana(s)
    s = "".join(c for c in s if "ァ" <= c <= "ヺ" or c == "ー")
    s = s.replace("ヲ", "オ").replace("ヂ", "ジ").replace("ヅ", "ズ")
    out, prev = [], None
    for c in s:
        if c == "ー" and prev:
            c = VOWEL_KANA[prev]
        if c == "ウ" and prev == "o":
            c = "オ"
        elif c == "イ" and prev == "e":
            c = "エ"
        out.append(c)
        prev = VOWEL_OF.get(c)
    return "".join(out)


def has_non_reading(s: str) -> bool:
    """True if the raw text still contains kanji or latin letters/digits."""
    return re.search(r"[一-鿿A-Za-z0-9]", unicodedata.normalize("NFKC", s)) is not None

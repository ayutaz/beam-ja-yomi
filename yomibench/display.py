"""Display helpers for the demo: Hepburn romaji and highlighted differences between readings."""

import difflib
import html

from yomibench.metrics import edit_distance
from yomibench.normalize import normalize

_BASE = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヰ": "i", "ヱ": "e", "ヲ": "o", "ヴ": "vu",
    "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o",
    "ャ": "ya", "ュ": "yu", "ョ": "yo", "ヮ": "wa",
}
_SMALL_Y = {"ャ": "a", "ュ": "u", "ョ": "o"}
_SMALL_V = {"ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o"}
_SPECIAL = {"シ": "sh", "ジ": "j", "チ": "ch", "ヂ": "j"}
_PAIRS = {"ファ": "fa", "フィ": "fi", "フェ": "fe", "フォ": "fo", "ティ": "ti", "ディ": "di",
          "トゥ": "tu", "ドゥ": "du", "ウィ": "wi", "ウェ": "we", "ウォ": "wo", "ヴァ": "va",
          "ヴィ": "vi", "ヴェ": "ve", "ヴォ": "vo", "シェ": "she", "ジェ": "je", "チェ": "che",
          "ツァ": "tsa", "ツィ": "tsi", "ツェ": "tse", "ツォ": "tso", "イェ": "ye"}


def _syllables(kana: str):
    """Yield (kana, romaji or None) chunks; None for ッ, ン, ー and non-kana characters."""
    i = 0
    while i < len(kana):
        two = kana[i:i + 2]
        if two in _PAIRS:
            yield two, _PAIRS[two]
            i += 2
            continue
        if len(two) == 2 and two[1] in _SMALL_Y and two[0] in _BASE and two[0] not in "ャュョァィゥェォ":
            head = _SPECIAL.get(two[0], _BASE[two[0]][:-1] + "y")
            yield two, head + _SMALL_Y[two[1]]
            i += 2
            continue
        if len(two) == 2 and two[1] in _SMALL_V and two[0] in _BASE:
            yield two, _BASE[two[0]][:-1] + _SMALL_V[two[1]]
            i += 2
            continue
        yield kana[i], _BASE.get(kana[i])
        i += 1


def kana_to_romaji(kana: str) -> str:
    chunks = list(_syllables(kana))
    out = []
    for k, (char, roman) in enumerate(chunks):
        nxt = next((r for c, r in chunks[k + 1:k + 2] if r), "")
        if roman is not None:
            out.append(roman)
        elif char == "ッ":
            out.append("t" if nxt.startswith("ch") else (nxt[:1] if nxt[:1] not in "aiueo" else ""))
        elif char == "ン":
            out.append("n'" if nxt[:1] in ("a", "i", "u", "e", "o", "y") and nxt else "n")
        elif char == "ー":
            prev = "".join(out)
            out.append(next((c for c in reversed(prev) if c in "aiueo"), ""))
        else:
            out.append(char)
    return "".join(out)


def diff_html(reference: str, other: str) -> str:
    """`other` with the parts that differ from `reference` wrapped in <mark> (HTML-escaped)."""
    parts = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(None, reference, other, autojunk=False).get_opcodes():
        seg = html.escape(other[j1:j2])
        if not seg:
            continue
        parts.append(seg if op == "equal" else f"<mark>{seg}</mark>")
    return "".join(parts)


def closest_gold(reading: str, alts: list[str], output: str) -> str:
    """The correct reading (main or alternative) nearest to `output` after normalization."""
    pred = normalize(output)
    return min([reading, *alts], key=lambda g: edit_distance(pred, normalize(g)))


def gold_diff_html(reading: str, alts: list[str], output: str, match: bool) -> str:
    """`output` marked against the nearest correct reading; no marks at all when it was scored correct.

    A wrong output is shown in its scored (normalized) form, so that only real differences are marked,
    not ー versus ウ/イ spellings or punctuation.
    """
    if match:
        return html.escape(output)
    return diff_html(normalize(closest_gold(reading, alts, output)), normalize(output))

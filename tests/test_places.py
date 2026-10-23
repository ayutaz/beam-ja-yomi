from pathlib import Path

from yomibench.data import validate_items
from yomibench.places import parse_ken_all, select_towns, to_items

FIXTURE = Path(__file__).parent / "fixtures" / "ken_all_sample.csv"


def test_parse_keeps_only_plain_kanji_towns():
    towns = parse_ken_all(FIXTURE)
    assert sorted({t["town"] for t in towns}) == sorted({"青葉", "青柳", "放出東", "秋築町"})
    osaka = next(t for t in towns if t["town"] == "放出東")
    assert osaka == {"code": "27124", "pref": "大阪府", "city": "大阪市鶴見区", "town": "放出東",
                     "pref_kana": "オオサカフ", "city_kana": "オオサカシツルミク", "town_kana": "ハナテンヒガシ"}


def test_select_is_deterministic_balanced_and_unique():
    towns = parse_ken_all(FIXTURE)
    a = select_towns(towns, n=3, seed=1)
    assert a == select_towns(towns, n=3, seed=1)
    assert len(a) == 3 and len({(t["town"], t["town_kana"]) for t in a}) == 3
    assert len({t["pref"] for t in a}) == 3          # 3つの府県から1件ずつ（偏らない）
    assert len(select_towns(towns, n=50)) == 4       # あるだけ返す（青葉の重複は1件）


def test_to_items_builds_valid_unverified_items():
    towns = [t for t in parse_ken_all(FIXTURE) if t["town"] == "放出東"]
    items = to_items(towns, source="Japan Post postal code data (KEN_ALL), downloaded 2026-10-22")
    assert items[0]["id"] == "place-0001" and items[0]["verified"] is False
    assert items[0]["text"] == "大阪市鶴見区放出東に向かいます"
    assert items[0]["reading"] == "オオサカシツルミクハナテンヒガシニムカイマス"
    assert validate_items(items, require_verified=False) == []

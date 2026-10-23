"""Pick place names from Japan Post's postal code data (KEN_ALL.CSV, 読み仮名データ).

Selection never looks at Beam or OpenJTalk output: rows are filtered by fixed rules, then
sampled per prefecture with a fixed seed and taken round-robin across prefectures.
Japan Post: "郵便番号データに限っては日本郵便株式会社は著作権を主張しません。自由に配布していただいて結構です。"
"""

import csv
import math
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

PLAIN_KANJI_TOWN = re.compile(r"[一-鿿々]{2,6}")
KATAKANA = re.compile(r"[ァ-ヺー]+")


def _kana(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def parse_ken_all(path: Path) -> list[dict]:
    """Rows whose town name is 2-6 plain kanji (drops 一円, parentheses, multi-line records, 丁目 etc.)."""
    towns = []
    with open(path, encoding="cp932", newline="") as f:
        for row in csv.reader(f):
            code, pref_k, city_k, town_k, pref, city, town = row[0], row[3], row[4], row[5], row[6], row[7], row[8]
            if not PLAIN_KANJI_TOWN.fullmatch(town) or town.endswith("一円"):
                continue
            kana = {"pref_kana": _kana(pref_k), "city_kana": _kana(city_k), "town_kana": _kana(town_k)}
            if not all(KATAKANA.fullmatch(v) for v in kana.values()):
                continue
            towns.append({"code": code, "pref": pref, "city": city, "town": town, **kana})
    return towns


def select_towns(towns: list[dict], n: int = 50, seed: int = 20261022) -> list[dict]:
    """Deterministic, prefecture-balanced sample without duplicate (town, reading) pairs."""
    unique, seen = [], set()
    for t in towns:
        key = (t["city"], t["town"], t["town_kana"])
        if key not in seen:
            seen.add(key)
            unique.append(t)
    by_pref = defaultdict(list)
    for t in unique:
        by_pref[t["code"][:2]].append(t)
    rng = random.Random(seed)
    queues = []
    for pref_code in sorted(by_pref):
        pool = by_pref[pref_code]
        rng.shuffle(pool)
        queues.append(pool)
    cap = math.ceil(n / max(len(queues), 1)) + 1
    picked, used_names = [], set()
    for round_ in range(cap):
        for pool in queues:
            if len(picked) >= n:
                return picked
            if round_ >= len(pool):
                continue
            t = pool[round_]
            if (t["town"], t["town_kana"]) not in used_names:  # same name in two cities: keep one
                used_names.add((t["town"], t["town_kana"]))
                picked.append(t)
    return picked


def to_items(towns: list[dict], source: str) -> list[dict]:
    return [{
        "id": f"place-{k:04d}",
        "category": "place",
        "text": f"{t['city']}{t['town']}に向かいます",
        "reading": f"{t['city_kana']}{t['town_kana']}ニムカイマス",
        "reading_alts": [],
        "source": source,
        "note": f"{t['pref']} / local government code {t['code']}",
        "verified": False,
    } for k, t in enumerate(towns, 1)]

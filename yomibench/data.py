"""Dataset schema, loading and validation (docs/02 "正解データの作り方")."""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from yomibench import ROOT
from yomibench.normalize import normalize

CATEGORIES = ("number", "homograph", "place", "person", "medical", "government", "disaster")
TARGET_COUNTS = {"number": 60, "homograph": 60, "place": 50, "person": 30,
                 "medical": 30, "government": 30, "disaster": 40}
MIN_PER_CATEGORY = 30
DATA_PATH = ROOT / "data" / "yomi.jsonl"
REQUIRED_KEYS = ("id", "category", "text", "reading", "reading_alts", "source", "note", "verified")

_KATAKANA_ONLY = re.compile(r"[ァ-ヺー]+")


def load_items(path: Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _reading_errors(label: str, reading) -> list[str]:
    if not isinstance(reading, str) or not _KATAKANA_ONLY.fullmatch(unicodedata.normalize("NFKC", reading)):
        return [f"{label} must be katakana and ー only: {reading!r}"]
    if not normalize(reading):
        return [f"{label} is empty after normalization"]
    return []


def validate_items(items: list[dict], require_verified: bool) -> list[str]:
    """Return a list of error messages (English, each starting with the item id). Empty = valid."""
    errors = []
    for n, item in enumerate(items):
        iid = item.get("id", f"<line {n + 1}>")
        missing = [k for k in REQUIRED_KEYS if k not in item]
        if missing:
            errors.append(f"{iid}: missing keys {missing}")
            continue
        problems = []
        if item["category"] not in CATEGORIES:
            problems.append(f"unknown category {item['category']!r}")
        if not re.fullmatch(rf"{re.escape(str(item['category']))}-\d{{4}}", str(iid)):
            problems.append("id must look like '<category>-0001'")
        if not str(item["text"]).strip():
            problems.append("text is empty")
        problems += _reading_errors("reading", item["reading"])
        if not isinstance(item["reading_alts"], list):
            problems.append("reading_alts must be a list")
        else:
            for alt in item["reading_alts"]:
                problems += _reading_errors("reading_alts entry", alt)
        if not str(item["source"]).strip():
            problems.append("source is empty")
        if not isinstance(item["verified"], bool):
            problems.append("verified must be true or false")
        elif require_verified and not item["verified"]:
            problems.append("not verified by the author")
        errors += [f"{iid}: {p}" for p in problems]

    for key in ("id", "text"):
        for value, count in Counter(i.get(key) for i in items if key in i).items():
            if count > 1:
                errors.append(f"{value}: duplicate {key} ({count} times)")
    return errors

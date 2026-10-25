"""The fresh checked sets for the student (plan 2026-10-24, Task 3): dev 100 and test 100 sentences
from Japanese Wikipedia, ids fresh-dev-0001 / fresh-test-0001, category "dev" or "test".

They use the benchmark's item schema (load_items, score, the review sheet) but not its categories, so
they get their own check here instead of data.validate_items.
"""

import re

from yomibench.data import REQUIRED_KEYS, _reading_errors

_ID = re.compile(r"fresh-(dev|test)-\d{4}")


def validate_fresh(items: list[dict]) -> list[str]:
    errors = []
    for n, item in enumerate(items):
        iid = item.get("id", f"<line {n + 1}>")
        missing = [k for k in REQUIRED_KEYS if k not in item]
        if missing:
            errors.append(f"{iid}: missing keys {missing}")
            continue
        problems = []
        m = _ID.fullmatch(str(iid))
        if not m:
            problems.append("id must look like 'fresh-dev-0001' or 'fresh-test-0001'")
        elif item["category"] != m.group(1):
            problems.append(f"category {item['category']!r} does not match the id")
        problems += _reading_errors("reading", item["reading"])
        for alt in item["reading_alts"]:
            problems += _reading_errors("reading_alts entry", alt)
        if item["verified"] is not True:
            problems.append("not checked (verified is not true)")
        errors += [f"{iid}: {p}" for p in problems]
    return errors

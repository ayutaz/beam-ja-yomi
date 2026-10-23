"""Validate a dataset file.

Usage:
  uv run scripts/check_data.py data/draft/place.jsonl
  uv run scripts/check_data.py data/yomi.jsonl --require-verified
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from yomibench.data import load_items, validate_items


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--require-verified", action="store_true")
    args = ap.parse_args()

    items = load_items(args.path)
    errors = validate_items(items, require_verified=args.require_verified)
    for e in errors:
        print(e)
    if errors:
        sys.exit(f"{len(errors)} error(s) in {args.path}")
    counts = Counter(i["category"] for i in items)
    print(f"OK: {len(items)} items  " + "  ".join(f"{c}={n}" for c, n in sorted(counts.items())))


if __name__ == "__main__":
    main()

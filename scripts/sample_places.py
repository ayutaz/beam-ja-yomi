"""Pick place-name items from KEN_ALL.CSV (Japan Post 読み仮名データ, 全国一括).

Download: https://www.post.japanpost.jp/service/search/zipcode/download/kogaki/zip/ken_all.zip
Unzip it into cache/ken_all/ (git-ignored), then:
  uv run scripts/sample_places.py --ken-all cache/ken_all/KEN_ALL.CSV --downloaded 2026-10-22
"""

import argparse
import json
import sys
from pathlib import Path

from yomibench.data import TARGET_COUNTS, validate_items
from yomibench.places import parse_ken_all, select_towns, to_items


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ken-all", type=Path, required=True)
    ap.add_argument("--downloaded", required=True, help="download date YYYY-MM-DD (goes into source)")
    ap.add_argument("--n", type=int, default=TARGET_COUNTS["place"])
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", type=Path, default=Path("data/draft/place.jsonl"))
    args = ap.parse_args()

    towns = parse_ken_all(args.ken_all)
    picked = select_towns(towns, n=args.n, seed=args.seed)
    items = to_items(picked, source=f"Japan Post postal code data (KEN_ALL), downloaded {args.downloaded}")
    errors = validate_items(items, require_verified=False)
    if errors:
        sys.exit("\n".join(errors))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    print(f"{len(towns)} candidate rows -> {len(items)} items (seed {args.seed}) -> {args.out}")


if __name__ == "__main__":
    main()

"""Apply the author's review sheet and write the dataset.

  uv run scripts/apply_review.py                 # review/review.tsv + data/draft/*.jsonl -> data/yomi.jsonl

Stops without writing if any row is left blank, an id does not match, or a category falls below the
minimum count.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from make_review_sheet import load_all_drafts
from yomibench.data import CATEGORIES, DATA_PATH, MIN_PER_CATEGORY, validate_items
from yomibench.review import apply_sheet, read_sheet


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", type=Path, default=Path("data/draft"))
    ap.add_argument("--sheet", type=Path, default=Path("review/review.tsv"))
    ap.add_argument("--out", type=Path, default=DATA_PATH)
    args = ap.parse_args()

    drafts = load_all_drafts(args.drafts)
    items, problems = apply_sheet(drafts, read_sheet(args.sheet))
    problems += validate_items(items, require_verified=True)
    counts = Counter(i["category"] for i in items)
    problems += [f"{c}: only {counts[c]} verified item(s), need {MIN_PER_CATEGORY}"
                 for c in CATEGORIES if counts[c] < MIN_PER_CATEGORY]
    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} problem(s); {args.out} not written")
    args.out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    print(f"wrote {len(items)} items to {args.out} (dropped {len(drafts) - len(items)})  "
          + "  ".join(f"{c}={counts[c]}" for c in CATEGORIES))


if __name__ == "__main__":
    main()

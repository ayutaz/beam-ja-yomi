"""Write the author's review sheet from the drafts.

  uv run scripts/make_review_sheet.py            # data/draft/*.jsonl -> review/review.tsv

Open review/review.tsv in Excel or a text editor. For each row write y (reading is right) or n (drop
the item) in `ok`, or write the right reading in `fixed_reading` (other right readings in `fixed_alts`,
joined by |). Save it again as UTF-8 tab-separated text.
"""

import argparse
import sys
from pathlib import Path

from yomibench.data import CATEGORIES, load_items, validate_items
from yomibench.review import make_sheet


def load_all_drafts(draft_dir: Path) -> list[dict]:
    items = []
    for cat in CATEGORIES:
        path = draft_dir / f"{cat}.jsonl"
        if path.exists():
            items += load_items(path)
    return items


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", type=Path, default=Path("data/draft"))
    ap.add_argument("--out", type=Path, default=Path("review/review.tsv"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing sheet")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} exists (it may hold the author's answers); pass --force to overwrite")
    items = load_all_drafts(args.drafts)
    errors = validate_items(items, require_verified=False)
    if errors:
        sys.exit("\n".join(errors))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(make_sheet(items), encoding="utf-8", newline="")
    print(f"wrote {len(items)} rows to {args.out}")


if __name__ == "__main__":
    main()

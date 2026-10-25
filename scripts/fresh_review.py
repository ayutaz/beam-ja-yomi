"""Review sheets for the fresh sets (plan 2026-10-24, Task 3): one row at a time, with who checked it kept in
`checked_by` (the published sets were filled in by Claude, cross-checked by a blind second reading).

  uv run scripts/fresh_review.py make                 # data/fresh/draft.jsonl -> review/fresh_dev.tsv, fresh_test.tsv
  uv run scripts/fresh_review.py apply --split dev    # review/fresh_dev.tsv -> data/fresh/dev.jsonl
  uv run scripts/fresh_review.py apply --split dev --checked-by "Claude ..."   # who filled the sheet (default: the author)

`apply` stops without writing if any row is blank, a reading cell was edited, or an item fails
validate_fresh. Rows marked n are dropped (and reported).
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yomibench import ROOT  # noqa: E402
from yomibench.data import load_items  # noqa: E402
from yomibench.fresh import validate_fresh  # noqa: E402
from yomibench.review import ALT_SEP, COLUMNS, apply_sheet, read_sheet  # noqa: E402

DRAFT = ROOT / "data" / "fresh" / "draft.jsonl"
SPLITS = ("dev", "test")


def sheet_path(split: str) -> Path:
    return ROOT / "review" / f"fresh_{split}.tsv"


def make_sheet_with_notes(items: list[dict]) -> str:
    """Like review.make_sheet, but Claude's note (doubts, "suggests n") is shown in the comment column."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\r\n")
    w.writerow(COLUMNS)
    for i in items:
        w.writerow([i["id"], i["category"], i["text"], i["reading"], ALT_SEP.join(i["reading_alts"]),
                    i["source"], "", "", "", f"Claude: {i['note']}" if i["note"] else ""])
    return "﻿" + buf.getvalue()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("make", "apply"))
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--checked-by", default="the author", help="who checked the readings, kept in each item")
    args = ap.parse_args()
    drafts = load_items(DRAFT)

    if args.action == "make":
        empty = [d["id"] for d in drafts if not d["reading"]]
        if empty:
            sys.exit(f"{len(empty)} draft(s) have no reading yet, e.g. {empty[0]}")
        for split in SPLITS:
            if sheet_path(split).exists():
                sys.exit(f"{sheet_path(split)} exists; the author may have started on it")
        for split in SPLITS:
            sheet_path(split).write_text(make_sheet_with_notes([d for d in drafts if d["category"] == split]),
                                         encoding="utf-8", newline="")
            print(f"wrote {sheet_path(split)}")
        return

    if args.split is None:
        sys.exit("apply needs --split dev or --split test")
    part = [d for d in drafts if d["category"] == args.split]
    items, problems = apply_sheet(part, read_sheet(sheet_path(args.split)))
    problems += validate_fresh(items)
    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} problem(s); nothing written")
    items = [{**i, "checked_by": args.checked_by} for i in items]
    out = ROOT / "data" / "fresh" / f"{args.split}.jsonl"
    out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8",
                   newline="\n")
    dropped = sorted({d["id"] for d in part} - {i["id"] for i in items})
    print(f"wrote {len(items)} items to {out}; dropped {len(dropped)}: {' '.join(dropped)}")


if __name__ == "__main__":
    main()

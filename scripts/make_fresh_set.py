"""Pick the fresh sets (dev 100, test 100) from the pool (plan 2026-10-24, Task 3).

The 200 sentences are chosen with random.Random(20261024), the first 100 become dev and the rest test.
They are written to data/fresh/draft.jsonl with an empty reading (Claude drafts it without running
OpenJTalk or Beam; it was planned that the author check every row, but in the end the readings were
cross-checked by a second, blind Claude reading instead: see the pre-registration), and their pool ids go to
data/distill/excluded_ids.txt so that the teacher never sees them.

    uv run scripts/make_fresh_set.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yomibench import ROOT  # noqa: E402
from yomibench.data import load_items  # noqa: E402

SEED, N_DEV, N_TEST = 20261024, 100, 100
POOL = ROOT / "data" / "distill" / "pool.jsonl"
DRAFT = ROOT / "data" / "fresh" / "draft.jsonl"
EXCLUDED = ROOT / "data" / "distill" / "excluded_ids.txt"


def main() -> None:
    if DRAFT.exists():
        sys.exit(f"{DRAFT} exists; refusing to re-pick (delete it on purpose if you must)")
    pool = load_items(POOL)
    picks = random.Random(SEED).sample(pool, N_DEV + N_TEST)
    rows = []
    for n, row in enumerate(picks):
        split, k = ("dev", n + 1) if n < N_DEV else ("test", n - N_DEV + 1)
        rows.append({"id": f"fresh-{split}-{k:04d}", "category": split, "text": row["text"], "reading": "",
                     "reading_alts": [], "pool_id": row["id"],
                     "source": f"Wikipedia ja 20231101: {row['title']} ({row['url']}), CC BY-SA 3.0; "
                               "reading drafted by Claude",
                     "note": "", "verified": False})
    DRAFT.parent.mkdir(parents=True, exist_ok=True)
    with open(DRAFT, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    EXCLUDED.write_text("".join(r["pool_id"] + "\n" for r in rows), encoding="utf-8", newline="\n")
    print(f"dev {N_DEV} + test {N_TEST} -> {DRAFT}")


if __name__ == "__main__":
    main()

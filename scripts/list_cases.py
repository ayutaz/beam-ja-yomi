"""List fixed / broken / unfixed cases of one Beam-fix run as Markdown, to read and label by hand.

  uv run scripts/list_cases.py                                   # results/, beam_fix-low-en-r0
  uv run scripts/list_cases.py --condition beam_fix-high-en-r0 --out analysis/cases-high.md
"""

import argparse
import json
import sys
from pathlib import Path

from yomibench.cases import classify, render_cases


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--condition", default="beam_fix-low-en-r0")
    ap.add_argument("--out", type=Path, help="write here instead of printing")
    args = ap.parse_args()
    md = render_cases(classify(_load(args.results_dir / "ojt.jsonl"),
                               _load(args.results_dir / f"{args.condition}.jsonl")), args.condition)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md + "\n", encoding="utf-8")
        print(f"-> {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()

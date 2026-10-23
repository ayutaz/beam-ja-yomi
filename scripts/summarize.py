"""Aggregate results into summary.json and report.md (in the same directory).

Usage:
  uv run scripts/summarize.py
  uv run scripts/summarize.py --results-dir cache/smoke
"""

import argparse
import json
import sys
from pathlib import Path

from yomibench.summary import render_markdown, summarize


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    summary = summarize(args.results_dir)
    if not summary["conditions"]:
        sys.exit(f"no result files in {args.results_dir}")
    (args.results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    md = render_markdown(summary)
    (args.results_dir / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

"""Run one condition over the dataset and write results/{condition}.jsonl.

Usage:
  uv run scripts/run_eval.py --system ojt
  uv run scripts/run_eval.py --system beam_direct --effort low --runs 3
  uv run scripts/run_eval.py --system beam_direct --effort low --lang ja
  uv run scripts/run_eval.py --system beam_fix --effort low --runs 3

Beam responses are cached (cache/beam/), so re-running never repeats a finished call.
Exit code 2 = the provider's daily limit was reached; re-run the same command after 09:00 JST.
--offline answers from the cache only; exit code 3 = a request is not cached (nothing was sent).
"""

import argparse
import json
import sys
from pathlib import Path

from yomibench.data import DATA_PATH, load_items, validate_items
from yomibench.evaluate import SYSTEMS, condition_name, load_drafts, run_condition
from yomibench.systems.beam import PROVIDERS, BeamClient, CacheMiss


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS, required=True)
    ap.add_argument("--effort", choices=["low", "medium", "high"])
    ap.add_argument("--lang", choices=["en", "ja"], default="en")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--provider", choices=list(PROVIDERS), default="reflection")
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    ap.add_argument("--allow-unverified", action="store_true", help="for smoke tests on drafts only")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--offline", action="store_true", help="never call the API; stop on the first uncached request")
    args = ap.parse_args()

    if (args.allow_unverified or args.limit) and args.results_dir == Path("results"):
        ap.error("--allow-unverified and --limit need a --results-dir other than results/ (the published one)")
    is_beam = args.system.startswith("beam_")
    if is_beam and not args.effort:
        ap.error("--effort is required for Beam systems")
    if args.system == "beam_fix" and args.lang != "en":
        ap.error("beam_fix only has an English prompt")

    items = load_items(args.data)
    errors = validate_items(items, require_verified=not args.allow_unverified)
    if errors:
        sys.exit("dataset is not valid:\n" + "\n".join(errors[:20]))
    if args.limit:
        items = items[: args.limit]

    beam = BeamClient(args.provider, offline=args.offline) if is_beam else None
    drafts = load_drafts(args.results_dir, items) if args.system == "beam_fix" else None
    runs = range(args.runs) if is_beam else range(1)
    for run in runs:
        name = condition_name(args.system, args.effort, args.lang, run)
        try:
            rows, done = run_condition(items, args.system, args.effort, args.lang, run, beam=beam, drafts=drafts)
        except CacheMiss as err:
            print(f"{name}: not in the cache ({err}); nothing was sent. Run without --offline to fill it.")
            sys.exit(3)
        partial = args.results_dir / f"{name}.partial.jsonl"
        if not done:
            write_jsonl(partial, rows)
            print(f"{name}: stopped after {len(rows)}/{len(items)} items.")
            print("Daily limit reached. Re-run after 09:00 JST; cached calls will not be repeated.")
            sys.exit(2)
        write_jsonl(args.results_dir / f"{name}.jsonl", rows)
        partial.unlink(missing_ok=True)
        exact = sum(r["match"] for r in rows)
        truncated = sum(r.get("truncated", False) for r in rows)
        print(f"{name}: exact {exact}/{len(rows)}  truncated {truncated}  -> {args.results_dir / (name + '.jsonl')}")


if __name__ == "__main__":
    main()

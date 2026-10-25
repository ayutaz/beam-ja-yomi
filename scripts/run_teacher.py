"""Ask Beam (via the Hugging Face router, Baseten) to read pool sentences two ways (plan 2026-10-24, Task 4).

Default provider: the HF router (HF_TOKEN), so the official Reflection allowance stays free for the
Space's Try it. With --provider reflection, the run stops itself after --daily-cap new output tokens
(default 700k of the 1M daily allowance, counting what cache/teacher/ already holds from today), leaving
the rest for Try it.
Responses are cached in cache/teacher/; a rerun resumes for free and counts cached usage toward the
budget, so it stops at the same place. Sentences picked for the fresh human-checked sets
(data/distill/excluded_ids.txt), and pool sentences close to them, are skipped. Failed calls (429, 5xx,
timeouts after retries) are left out of the output and retried on the next run. Rows already in --out
are kept and not asked again (so a switch of provider never pays twice); earlier HF rows count toward
the budget.

    uv run scripts/run_teacher.py --limit 20                         # check the path
    uv run scripts/run_teacher.py --limit 20 --effort high --out cache/teacher_high20.jsonl
    uv run scripts/run_teacher.py --limit 12000 --budget-usd 18
    uv run scripts/run_teacher.py --limit 12000 --provider reflection --workers 1
"""

import argparse
import datetime
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yomibench import ROOT  # noqa: E402
from yomibench.corpus import LeakFilter  # noqa: E402
from yomibench.data import load_items  # noqa: E402
from yomibench.systems import openjtalk  # noqa: E402
from yomibench.systems.beam import BeamClient, DailyLimitReached  # noqa: E402
from yomibench.teacher import (Budget, DailyCapReached, TokenMeter, cache_cost_usd, cost_usd, teach_one,  # noqa: E402
                               tokens_since)

DISTILL = ROOT / "data" / "distill"
FRESH_DRAFT = ROOT / "data" / "fresh" / "draft.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--budget-usd", type=float, default=18.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--provider", choices=("hf", "reflection"), default="hf")
    ap.add_argument("--daily-cap", type=int, default=700_000, help="reflection only: new output tokens")
    ap.add_argument("--out", type=Path, default=DISTILL / "teacher.jsonl")
    args = ap.parse_args()

    excluded = set((DISTILL / "excluded_ids.txt").read_text(encoding="utf-8").split())
    near_fresh = LeakFilter([r["text"] for r in load_items(FRESH_DRAFT)])
    pool = [r for r in load_items(DISTILL / "pool.jsonl")
            if r["id"] not in excluded and not near_fresh.leaks(r["text"])]
    prev = {r["id"]: r for r in load_items(args.out)} if args.out.exists() else {}
    rows = [r for r in pool[:args.limit] if r["id"] not in prev]
    print(f"pool {len(pool)} usable, {len(prev)} already done, asking {len(rows)}; "
          f"{args.provider}, effort {args.effort}")

    beam = BeamClient(provider=args.provider, cache_dir=ROOT / "cache" / "teacher")
    meter = None
    if args.provider == "reflection":
        reset = datetime.datetime.now(datetime.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        used = tokens_since(beam.cache_dir, reset.timestamp())  # the allowance resets at 00:00 UTC
        print(f"already used today (since {reset.isoformat()}): {used} output tokens; cap {args.daily_cap}")
        meter = TokenMeter(beam.client.chat.completions.create, cap=args.daily_cap, used=used)
        beam.client = SimpleNamespace(chat=SimpleNamespace(completions=meter))
    budget = Budget(args.budget_usd if args.provider == "hf" else float("inf"))
    for r in prev.values():
        if r.get("provider") == "hf":
            budget.add(r)
    stop = None
    done, failed = dict(prev), 0
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(teach_one, beam, r["text"], openjtalk.read(r["text"]), args.effort): r for r in rows}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                out = fut.result()
            except (DailyCapReached, DailyLimitReached) as err:
                stop = f"{type(err).__name__}: {err}"
                for f in futures:
                    f.cancel()
                break
            except (openai.APIError, openai.APIConnectionError) as err:
                failed += 1
                print(f"{r['id']}: {type(err).__name__}", file=sys.stderr)
                continue
            done[r["id"]] = {**r, **out, "provider": args.provider}
            budget.add(out)
            if len(done) % 200 == 0:
                print(f"{len(done)} done, ${budget.spent:.2f}", flush=True)
            if budget.exceeded:
                print(f"budget reached at {len(done)} rows (${budget.spent:.2f}); cancelling the rest")
                for f in futures:
                    f.cancel()
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for rid in sorted(done):
            f.write(json.dumps(done[rid], ensure_ascii=False) + "\n")
    agreed = sum(r["agree"] for r in done.values())
    n = max(len(done), 1)
    stats = {"rows": len(done), "failed": failed, "agree": agreed, "agree_rate": round(agreed / n, 4),
             "hf_cost_usd": round(cost_usd([r for r in done.values() if r.get("provider") == "hf"]), 4),
             "hf_cost_usd_all_calls": round(cache_cost_usd(beam.cache_dir), 4),  # incl. calls of unfinished rows
             "by_provider": {p: sum(r.get("provider") == p for r in done.values()) for p in ("hf", "reflection")},
             "effort": args.effort,
             "mean_reasoning_tokens": round(sum(r["reasoning_tokens"] for r in done.values()) / n / 2, 1),
             "truncated": sum(r["direct_finish"] == "length" or r["fix_finish"] == "length" for r in done.values())}
    if stop:
        print(f"stopped: {stop}")
    if meter is not None:
        stats["output_tokens_today"] = meter.used
    print(json.dumps(stats))
    if args.out == DISTILL / "teacher.jsonl":
        (DISTILL / "teacher_stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

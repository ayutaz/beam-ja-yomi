"""Choose the student on the 100 dev sentences (plan 2026-10-24, Task 6): train each pre-registered candidate,
predict dev, try every gate in the pre-registered range, pick by the pre-registered rule, check G1.

    uv run scripts/sweep_dev.py                 # train what is missing, predict, write results/distill/dev_summary.md
    uv run scripts/sweep_dev.py --freeze        # ... and write results/distill/frozen.json (then never change it)

Models go to models/<name>/ (git-ignored), dev predictions to results/distill/dev/<name>.jsonl. Nothing here
reads the 300 benchmark sentences or the test set.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yomibench import ROOT  # noqa: E402
from yomibench.data import load_items  # noqa: E402
from yomibench.student import gate  # noqa: E402
from yomibench.distill_eval import (DEFAULT_GATE, GATE_GRID, choose, file_hashes, passes_g1, regate,  # noqa: E402
                                    stale_outputs, summarize_run)

# docs/prereg_yomi_distill.md, "Candidates"; listed smaller model and fewer epochs first (the last tie-break)
CANDIDATES = [
    {"name": "s440-e1", "base": "llm-jp/llm-jp-3-440m", "lora": False, "epochs": 1},
    {"name": "s440-e2", "base": "llm-jp/llm-jp-3-440m", "lora": False, "epochs": 2},
    {"name": "s440-e3", "base": "llm-jp/llm-jp-3-440m", "lora": False, "epochs": 3},
    {"name": "s1.8b-lora-e1", "base": "llm-jp/llm-jp-3-1.8b", "lora": True, "epochs": 1},
    {"name": "s1.8b-lora-e2", "base": "llm-jp/llm-jp-3-1.8b", "lora": True, "epochs": 2},
]
DEV = ROOT / "data" / "fresh" / "dev.jsonl"
OUT = ROOT / "results" / "distill"
MODELS = ROOT / "models"
MAX_NEW_TOKENS = 160


def run(args: list[str]) -> None:
    print("$ " + " ".join(args), flush=True)
    subprocess.run(["uv", "run", "--no-sync", *args], cwd=ROOT, check=True)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="write results/distill/frozen.json for the chosen one")
    ap.add_argument("--only", nargs="*", help="candidate names to run (default: all)")
    args = ap.parse_args()

    items = load_items(DEV)
    if not 90 <= len(items) <= 100 or not all(i.get("verified") is True for i in items):
        sys.exit(f"{DEV} must hold the 90-100 checked dev sentences (run scripts/fresh_review.py apply first)")
    cands = [c for c in CANDIDATES if not args.only or c["name"] in args.only]
    rows_by = {}
    for c in cands:
        model = MODELS / c["name"]
        pred = OUT / "dev" / f"{c['name']}.jsonl"
        stale = stale_outputs(c, model, pred, {i["id"] for i in items})
        if stale:
            sys.exit("will not reuse (delete them to train or predict again):\n" + "\n".join(stale))
        if not (model / "config.json").exists():
            run(["scripts/train_student.py", "--base", c["base"], "--epochs", str(c["epochs"]), "--out", str(model)]
                + (["--lora"] if c["lora"] else []))
        if not pred.exists():
            run(["scripts/predict_student.py", "--model", str(model), "--items", str(DEV), "--out", str(pred),
                 "--max-new-tokens", str(MAX_NEW_TOKENS)])
        rows_by[c["name"]] = load_items(pred)

    ojt = {r["id"]: r["ojt"] for r in next(iter(rows_by.values()))}
    base = summarize_run(items, ojt, ojt)
    tried = []
    for c in cands:
        rows = rows_by[c["name"]]
        for lo, hi in GATE_GRID:
            s = summarize_run(items, ojt, regate(rows, lo, hi))
            fell = sum(gate(r["ojt"], r["raw"], r["truncated"], lo, hi)[1] for r in rows)
            tried.append({**c, "min_ratio": lo, "max_ratio": hi, "fallback": fell,
                          **{k: v for k, v in s.items() if k != "matches"}})
    best = choose(tried)
    g1 = passes_g1(best)

    lines = [f"# Dev sweep ({len(items)} sentences)", "",
             f"OpenJTalk: {pct(base['match'])}. Gate range tried: min {GATE_GRID[0][0]}-{GATE_GRID[-1][0]}, "
             f"max {GATE_GRID[0][1]}-{GATE_GRID[-1][1]}.", "",
             "| candidate | gate | exact match | fixed | broken | break rate | fallback |", "|---|---|---|---|---|---|---|"]
    for c in cands:
        mine = [t for t in tried if t["name"] == c["name"]]
        at_default = next(t for t in mine if (t["min_ratio"], t["max_ratio"]) == DEFAULT_GATE)
        own_best = choose(mine)
        for label, t in (("default", at_default), ("best", own_best)):
            lines.append(f"| {c['name']} | {label} {t['min_ratio']}-{t['max_ratio']} | {pct(t['match'])} | "
                         f"{t['fixed']} | {t['broken']} | {pct(t['break_rate'])} | {t['fallback']} |")
    lines += ["", f"**Chosen:** {best['name']}, gate {best['min_ratio']}-{best['max_ratio']}: {pct(best['match'])} "
              f"({100 * (best['match'] - base['match']):+.1f} points over OpenJTalk), break rate {pct(best['break_rate'])}.",
              f"**G1** (>= +3 points and <= 3% broken): {'passed' if g1 else 'NOT passed'}."]
    text = "\n".join(lines) + "\n"
    (OUT / "dev_summary.md").write_text(text, encoding="utf-8")
    print(text)

    if args.freeze:
        frozen_path = OUT / "frozen.json"
        if frozen_path.exists():
            sys.exit(f"{frozen_path} exists: the student is already frozen")
        model_dir = (MODELS / best["name"]).relative_to(ROOT)
        frozen = {"model_dir": model_dir.as_posix(), "sha256": file_hashes(ROOT / model_dir),
                  "min_ratio": best["min_ratio"], "max_ratio": best["max_ratio"], "max_new_tokens": MAX_NEW_TOKENS,
                  "candidate": {k: best[k] for k in ("name", "base", "lora", "epochs")},
                  "train_config": json.loads((ROOT / model_dir / "train_config.json").read_text(encoding="utf-8")),
                  "dev": {k: best[k] for k in ("n", "match", "ojt_match", "fixed", "broken", "break_rate", "fallback")},
                  "g1": g1}
        frozen_path.write_text(json.dumps(frozen, indent=1) + "\n", encoding="utf-8")
        print(f"frozen -> {frozen_path}")


if __name__ == "__main__":
    main()

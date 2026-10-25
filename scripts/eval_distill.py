"""Tables for the student (plan 2026-10-24, Tasks 6-7).

Each --pred is NAME=FILE, a JSONL with `id` and `output` (student predictions from predict_student.py,
which also carry `ojt` and `fallback`; or benchmark result files such as results/beam_fix-low-en-r0.jsonl).
OpenJTalk's reading is taken from the `ojt` field of the first file that has one, else from --ojt.

    uv run scripts/eval_distill.py --split dev --pred s440-e2=results/distill/dev/s440-e2.jsonl ...
    uv run scripts/eval_distill.py --split test --pred student=... --pred beam_direct=... --pred beam_fix=... --triage
    uv run scripts/eval_distill.py --split eval300 --ojt results/ojt.jsonl --pred student=... \
        --pred beam_fix=results/beam_fix-low-en-r0.jsonl --pred beam_direct=results/beam_direct-low-en-r0.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yomibench import ROOT  # noqa: E402
from yomibench.data import DATA_PATH, load_items  # noqa: E402
from yomibench.distill_eval import bootstrap_ci, fallback_rate, summarize_run  # noqa: E402
from yomibench.metrics import score  # noqa: E402
from yomibench.teacher import agree  # noqa: E402

ITEMS = {"dev": ROOT / "data" / "fresh" / "dev.jsonl", "test": ROOT / "data" / "fresh" / "test.jsonl",
         "eval300": DATA_PATH}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def table(items, ojt, systems: dict[str, list[dict]]) -> list[str]:
    lines = ["| system | n | exact match (95% CI) | CER | fixed | broken | break rate | fallback |",
             "|---|---|---|---|---|---|---|---|"]
    base = summarize_run(items, ojt, ojt)
    lo, hi = bootstrap_ci(base["matches"])
    lines.append(f"| OpenJTalk | {base['n']} | {pct(base['match'])} ({pct(lo)}-{pct(hi)}) | {pct(base['cer'])} | - | - | - | - |")
    for name, rows in systems.items():
        pred = {r["id"]: r["output"] for r in rows}
        s = summarize_run(items, ojt, pred)
        lo, hi = bootstrap_ci(s["matches"])
        fb = fallback_rate(rows, {i["id"] for i in items})
        fallback = "-" if fb is None else pct(fb)
        lines.append(f"| {name} | {s['n']} | {pct(s['match'])} ({pct(lo)}-{pct(hi)}) | {pct(s['cer'])} | {s['fixed']} | "
                     f"{s['broken']} | {pct(s['break_rate'])} | {fallback} |")
    return lines


def triage(items, direct: list[dict], fix: list[dict]) -> list[str]:
    d = {r["id"]: r for r in direct}
    f = {r["id"]: r for r in fix}
    agreed = [i for i in items if agree(d[i["id"]], f[i["id"]])]  # the same rule that picked the teaching pairs
    ok = sum(score(i, f[i["id"]]["output"])["match"] for i in agreed)
    rest = [i for i in items if i not in agreed]
    rest_ok = sum(score(i, f[i["id"]]["output"])["match"] for i in rest)
    return [f"- Agreed (accept Beam's reading): {len(agreed)}/{len(items)} ({pct(len(agreed) / len(items))}), "
            f"exact match {ok}/{len(agreed)} ({pct(ok / max(len(agreed), 1))})",
            f"- Not agreed (send to a person): {len(rest)}; beam_fix would have been right on {rest_ok}"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(ITEMS), required=True)
    ap.add_argument("--pred", action="append", default=[], help="NAME=FILE")
    ap.add_argument("--ojt", type=Path, help="JSONL with id and output, when no --pred has an ojt field")
    ap.add_argument("--triage", action="store_true", help="needs --pred beam_direct=... and beam_fix=...")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    items = load_items(ITEMS[args.split])
    ids = {i["id"] for i in items}
    systems = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        rows = [r for r in load_items(Path(path)) if r["id"] in ids]
        missing = ids - {r["id"] for r in rows}
        if missing:
            sys.exit(f"{name}: {len(missing)} item(s) missing, e.g. {sorted(missing)[0]}")
        systems[name] = rows
    ojt_rows = next((rows for rows in systems.values() if "ojt" in rows[0]), None)
    if ojt_rows is not None:
        ojt = {r["id"]: r["ojt"] for r in ojt_rows}
    elif args.ojt:
        ojt = {r["id"]: r["output"] for r in load_items(args.ojt)}
    else:
        sys.exit("no OpenJTalk readings: pass a student prediction file or --ojt")

    lines = [f"## {args.split} ({len(items)} sentences)", ""] + table(items, ojt, systems)
    if args.split == "eval300":
        no_place = [i for i in items if i["category"] != "place"]
        lines += ["", f"### without place names ({len(no_place)})", ""] + table(no_place, ojt, systems)
        for cat in sorted({i["category"] for i in items}):
            part = [i for i in items if i["category"] == cat]
            lines += ["", f"### {cat} ({len(part)})", ""] + table(part, ojt, systems)
    if args.triage:
        lines += ["", "### triage: beam_direct and beam_fix agree -> accept, else -> a person", ""]
        lines += triage(items, systems["beam_direct"], systems["beam_fix"])
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

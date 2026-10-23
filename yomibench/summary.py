"""Aggregate results/{condition}.jsonl into summary.json and an English Markdown report."""

import json
import re
from collections import defaultdict
from pathlib import Path

CONDITION_FILE = re.compile(r"^(ojt|ojt_vanilla|beam_(?:direct|fix)-[a-z]+-[a-z]+-r\d+)\.jsonl$")


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _stats(rows: list[dict]) -> dict:
    n = len(rows)
    exact = sum(r["match"] for r in rows)
    reasoning = [r["reasoning_tokens"] for r in rows if r.get("reasoning_tokens") is not None]
    completion = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    return {
        "n": n,
        "exact": exact,
        "exact_rate": exact / n if n else 0.0,
        "mean_cer": sum(r["cer"] for r in rows) / n if n else 0.0,
        "format_issues": sum(r["format_issue"] for r in rows),
        "truncated": sum(bool(r.get("truncated")) for r in rows),
        "mean_reasoning_tokens": sum(reasoning) / len(reasoning) if reasoning else None,
        "total_completion_tokens": sum(completion) if completion else None,
        "mean_seconds": (sum(r["seconds"] for r in rows if "seconds" in r) / n) if rows and "seconds" in rows[0] else None,
        "by_category": {c: {"n": len(rs), "exact_rate": sum(r["match"] for r in rs) / len(rs),
                            "mean_cer": sum(r["cer"] for r in rs) / len(rs)}
                        for c, rs in sorted(by_cat.items())},
    }


def check_results(results_dir: Path, items: list[dict]) -> list[str]:
    """Problems that would make results/ disagree with the dataset (empty = they belong together).

    Every condition file must hold exactly the dataset's ids, with the same text, reading and reading_alts.
    """
    results_dir = Path(results_dir)
    files = sorted(p for p in results_dir.glob("*.jsonl") if CONDITION_FILE.match(p.name))
    if not files:
        return [f"no condition files in {results_dir}"]
    by_id = {i["id"]: i for i in items}
    problems = []
    for path in files:
        name = path.name[:-len(".jsonl")]
        rows = {r["id"]: r for r in _load(path)}
        problems += [f"{name}: {i} is in the data but not in the results" for i in by_id if i not in rows]
        problems += [f"{name}: {i} is in the results but not in the data" for i in rows if i not in by_id]
        problems += [f"{name}: {i} has a different text or reading than the data (re-run this condition)"
                     for i, r in rows.items() if i in by_id
                     and any(r.get(k) != by_id[i][k] for k in ("text", "reading", "reading_alts"))]
    return problems


def summarize(results_dir: Path) -> dict:
    results_dir = Path(results_dir)
    data = {p.name[:-len(".jsonl")]: _load(p)
            for p in sorted(results_dir.glob("*.jsonl")) if CONDITION_FILE.match(p.name)}
    summary = {"conditions": {name: _stats(rows) for name, rows in data.items()}, "fix_vs_ojt": {}, "run_spread": {}}

    ojt = {r["id"]: r["match"] for r in data.get("ojt", [])}
    for name, rows in data.items():
        if not name.startswith("beam_fix-") or not ojt:
            continue
        counts = {"fixed": 0, "broken": 0, "kept_correct": 0, "still_wrong": 0}
        for r in rows:
            if r["id"] not in ojt:  # check_results reports this; do not crash the demo
                continue
            before, after = ojt[r["id"]], r["match"]
            key = ("kept_correct" if after else "broken") if before else ("fixed" if after else "still_wrong")
            counts[key] += 1
        summary["fix_vs_ojt"][name] = counts

    groups = defaultdict(list)
    for name, stats in summary["conditions"].items():
        m = re.match(r"^(.*)-r\d+$", name)
        if m:
            groups[m.group(1)].append(stats["exact_rate"])
    summary["run_spread"] = {g: {"runs": len(v), "min": min(v), "max": max(v)}
                             for g, v in sorted(groups.items()) if len(v) >= 2}
    return summary


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_markdown(summary: dict) -> str:
    conds = summary["conditions"]
    out = ["# Results", "", "## Overall", "",
           "| condition | n | exact match | mean CER | format issues | truncated | mean reasoning tokens |",
           "|---|---|---|---|---|---|---|"]
    for name, s in conds.items():
        rt = "" if s["mean_reasoning_tokens"] is None else f"{s['mean_reasoning_tokens']:.0f}"
        out.append(f"| {name} | {s['n']} | {_pct(s['exact_rate'])} | {s['mean_cer']:.3f} | "
                   f"{s['format_issues']} | {s['truncated']} | {rt} |")

    cats = sorted({c for s in conds.values() for c in s["by_category"]})
    out += ["", "## By category (exact match)", "", "| category | " + " | ".join(conds) + " |",
            "|---|" + "---|" * len(conds)]
    for c in cats:
        cells = [_pct(s["by_category"][c]["exact_rate"]) if c in s["by_category"] else "" for s in conds.values()]
        out.append(f"| {c} | " + " | ".join(cells) + " |")

    out += ["", "## Beam fix vs OpenJTalk", "",
            "fixed = OpenJTalk wrong → Beam right; broken = OpenJTalk right → Beam wrong.", "",
            "| condition | fixed | broken | kept correct | still wrong |", "|---|---|---|---|---|"]
    for name, c in summary["fix_vs_ojt"].items():
        out.append(f"| {name} | {c['fixed']} | {c['broken']} | {c['kept_correct']} | {c['still_wrong']} |")

    if summary["run_spread"]:
        out += ["", "## Run-to-run spread (exact match)", "", "| condition | runs | min | max |", "|---|---|---|---|"]
        for g, s in summary["run_spread"].items():
            out.append(f"| {g} | {s['runs']} | {_pct(s['min'])} | {_pct(s['max'])} |")
    return "\n".join(out) + "\n"

"""Scoring for the student (plan 2026-10-24, Tasks 6-7): exact match against human-checked readings,
fixed / broken counts against OpenJTalk, and bootstrap intervals."""

import random

from yomibench.metrics import score
from yomibench.student import gate

DEFAULT_GATE = (0.8, 1.25)
# the pre-registered range for choosing the gate on dev (docs/prereg_yomi_distill.md)
GATE_GRID = [(lo, hi) for lo in (0.7, 0.75, 0.8, 0.85, 0.9) for hi in (1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4)]


def summarize_run(items: list[dict], ojt: dict[str, str], pred: dict[str, str]) -> dict:
    """items: human-checked items; ojt / pred: id -> reading. break_rate = broken / OJT-correct."""
    ojt_ok = [score(i, ojt[i["id"]])["match"] for i in items]
    scored = [score(i, pred[i["id"]]) for i in items]
    pred_ok = [x["match"] for x in scored]
    n = len(items)
    fixed = sum(p and not o for o, p in zip(ojt_ok, pred_ok))
    broken = sum(o and not p for o, p in zip(ojt_ok, pred_ok))
    return {"n": n, "match": sum(pred_ok) / n, "ojt_match": sum(ojt_ok) / n, "fixed": fixed, "broken": broken,
            "break_rate": broken / max(sum(ojt_ok), 1), "cer": sum(x["cer"] for x in scored) / n, "matches": pred_ok}


def fallback_rate(rows: list[dict], ids: set[str]) -> float | None:
    """Share of the given items where the gate kept OpenJTalk's draft; None for rows without `fallback`."""
    fb = [r["fallback"] for r in rows if r["id"] in ids and "fallback" in r]
    return sum(fb) / len(fb) if fb else None


def bootstrap_ci(matches: list[bool], n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of the exact-match rate."""
    rng = random.Random(seed)
    k = len(matches)
    rates = sorted(sum(rng.choices(matches, k=k)) / k for _ in range(n))
    return rates[int(0.025 * n)], rates[int(0.975 * n) - 1]


def regate(rows: list[dict], min_ratio: float, max_ratio: float) -> dict[str, str]:
    """id -> reading, re-applying the gate to predict_student rows (which keep `raw`, `truncated`, `ojt`)."""
    return {r["id"]: gate(r["ojt"], r["raw"], r["truncated"], min_ratio, max_ratio)[0] for r in rows}


def choose(cands: list[dict]) -> dict:
    """The pre-registered rule: highest exact match, then fewer broken. Further ties: the default gate, then the
    earlier candidate in the list (smaller model, fewer epochs)."""
    def key(i_c):
        i, c = i_c
        return (-c["match"], c["broken"], (c["min_ratio"], c["max_ratio"]) != DEFAULT_GATE, i)
    return min(enumerate(cands), key=key)[1]


def passes_g1(s: dict) -> bool:
    """At least 3 points over OpenJTalk on dev, and at most 3% of OpenJTalk's correct readings broken."""
    return s["match"] - s["ojt_match"] >= 0.03 - 1e-9 and s["break_rate"] <= 0.03 + 1e-9


def file_hashes(model_dir) -> dict[str, str]:
    """sha256 of every file directly in model_dir, for results/distill/frozen.json."""
    import hashlib
    from pathlib import Path

    out = {}
    for p in sorted(Path(model_dir).iterdir()):
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            out[p.name] = h.hexdigest()
    return out


def stale_outputs(cand: dict, model_dir, pred_path, dev_ids: set[str]) -> list[str]:
    """Why an existing model or dev prediction file must not be reused for `cand` (empty if fine or absent):
    the model's train_config.json names other settings, or the predictions cover other sentences."""
    import json
    from pathlib import Path

    from yomibench.data import load_items

    model_dir, pred_path = Path(model_dir), Path(pred_path)
    problems = []
    config = model_dir / "train_config.json"
    if (model_dir / "config.json").exists():
        trained = json.loads(config.read_text(encoding="utf-8")) if config.exists() else {}
        for key in ("base", "lora", "epochs"):
            if trained.get(key) != cand[key]:
                problems.append(f"{model_dir}: {key} is {trained.get(key)!r}, the candidate needs {cand[key]!r}")
    if pred_path.exists():
        got = {r["id"] for r in load_items(pred_path)}
        if got != dev_ids:
            problems.append(f"{pred_path}: predictions for other sentences ({len(got - dev_ids)} not in dev, "
                            f"{len(dev_ids - got)} missing)")
    return problems

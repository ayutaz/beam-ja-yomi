"""Run one condition (system x effort x prompt language x run) over all items."""

import json
from pathlib import Path

from yomibench.metrics import score
from yomibench.systems import openjtalk
from yomibench.systems.beam import BeamClient, DailyLimitReached, build_messages, parse_response

SYSTEMS = ("ojt", "ojt_vanilla", "beam_direct", "beam_fix")
ITEM_KEYS = ("id", "category", "text", "reading", "reading_alts")


def condition_name(system: str, effort: str | None, lang: str, run: int) -> str:
    if system in openjtalk.MODES:
        return system
    return f"{system}-{effort}-{lang}-r{run}"


def load_drafts(results_dir: Path, items: list[dict]) -> dict[str, str]:
    """OpenJTalk readings (the drafts beam_fix corrects), checked against the current data."""
    path = Path(results_dir) / "ojt.jsonl"
    if not path.exists():
        raise ValueError(f"{path} not found: run --system ojt first (beam_fix corrects its readings)")
    rows = {r["id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}
    stale = [i["id"] for i in items if i["id"] not in rows or rows[i["id"]]["text"] != i["text"]]
    if stale:
        raise ValueError(f"{path} is stale for {len(stale)} item(s) (e.g. {stale[0]}): re-run --system ojt")
    return {i["id"]: rows[i["id"]]["output"] for i in items}


def run_condition(items, system, effort=None, lang="en", run=0, beam: BeamClient | None = None,
                  drafts: dict[str, str] | None = None) -> tuple[list[dict], bool]:
    """Return (rows, finished). Stops early (finished=False) when the daily limit is reached."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}")
    rows = []
    for item in items:
        base = {k: item[k] for k in ITEM_KEYS}
        if system in openjtalk.MODES:
            output = openjtalk.read(item["text"], system)
            rows.append({**base, "output": output, **score(item, output)})
            continue
        if system == "beam_fix":
            if drafts is None:
                raise ValueError("beam_fix needs the OpenJTalk drafts (load_drafts)")
            messages = build_messages("fix", lang, item["text"], drafts[item["id"]])
        else:
            messages = build_messages("direct", lang, item["text"])
        try:
            parsed = parse_response(beam.complete(messages, effort, run))
        except DailyLimitReached:
            return rows, False
        row = {**base, **parsed, **score(item, parsed["output"], parsed["truncated"]), "provider": beam.provider}
        if system == "beam_fix":
            row["draft"] = drafts[item["id"]]
        rows.append(row)
    return rows, True

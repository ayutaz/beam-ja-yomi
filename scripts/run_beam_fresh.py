"""OpenJTalk and Beam on the fresh test set, for the comparison in the student's report (plan 2026-10-24,
Task 7). Same settings as the benchmark's main condition: official Reflection API, `low`, English
prompts, run 0. Responses go to the benchmark's cache (cache/beam/).

    uv run scripts/run_beam_fresh.py            # data/fresh/test.jsonl -> results/distill/test/{ojt,beam_direct,beam_fix}.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yomibench import ROOT  # noqa: E402
from yomibench.data import load_items  # noqa: E402
from yomibench.evaluate import run_condition  # noqa: E402
from yomibench.systems.beam import BeamClient  # noqa: E402

OUT = ROOT / "results" / "distill" / "test"


def write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                       encoding="utf-8", newline="\n")


def main() -> None:
    items = load_items(ROOT / "data" / "fresh" / "test.jsonl")
    ojt, _ = run_condition(items, "ojt")
    write("ojt", ojt)
    beam = BeamClient(provider="reflection")
    drafts = {r["id"]: r["output"] for r in ojt}
    for system in ("beam_direct", "beam_fix"):
        rows, finished = run_condition(items, system, "low", "en", 0, beam=beam, drafts=drafts)
        if not finished:
            sys.exit(f"{system}: daily limit reached after {len(rows)} rows; run again after 09:00 JST")
        write(system, rows)
        print(f"{system}: {sum(r['match'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()

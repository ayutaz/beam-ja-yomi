"""Sort Beam-fix outputs against OpenJTalk into fixed / broken / unfixed / kept, and list them for reading."""

import difflib

from yomibench.display import closest_gold
from yomibench.normalize import normalize

KINDS = ("broken", "unfixed", "fixed", "kept")  # the order they are read in; "kept" (both right) is not listed
_NOTES = {
    "broken": "OpenJTalk was right, Beam made it wrong.",
    "unfixed": "OpenJTalk was wrong and Beam's output is still wrong.",
    "fixed": "OpenJTalk was wrong, Beam made it right.",
}


def classify(ojt_rows: list[dict], fix_rows: list[dict]) -> list[dict]:
    ojt = {r["id"]: r for r in ojt_rows}
    out = []
    for f in fix_rows:
        o = ojt.get(f["id"])
        if o is None:
            continue
        kind = {(False, True): "fixed", (True, False): "broken", (False, False): "unfixed"}.get(
            (o["match"], f["match"]), "kept")
        out.append({"id": f["id"], "category": f["category"], "text": f["text"], "reading": f["reading"],
                    "reading_alts": f.get("reading_alts", []), "ojt": o["output"], "fix": f["output"],
                    "ojt_match": o["match"], "fix_match": f["match"], "kind": kind,
                    "kept_draft": f["output"] == o["output"]})
    return out


def mark_diff(reference: str, other: str) -> str:
    """`other` with the parts that differ from `reference` in 【】."""
    parts = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(None, reference, other, autojunk=False).get_opcodes():
        seg = other[j1:j2]
        if seg:
            parts.append(seg if op == "equal" else f"【{seg}】")
    return "".join(parts)


def _marked(case: dict, which: str) -> str:
    out = case[which]
    if case[f"{which}_match"]:
        return out
    return mark_diff(normalize(closest_gold(case["reading"], case["reading_alts"], out)), normalize(out))


def _cell(s: str) -> str:
    return s.replace("|", r"\|").replace("\n", " ")


def render_cases(cases: list[dict], condition: str) -> str:
    lines = [f"# Cases: {condition}", "",
             "Wrong readings are shown in their scored (normalized) form; "
             "【】 = differs from the nearest correct reading. Fill in `type` while reading.", ""]
    for kind in KINDS[:3]:
        rows = sorted((c for c in cases if c["kind"] == kind), key=lambda c: (c["category"], c["id"]))
        lines += [f"## {kind} ({len(rows)})", "", _NOTES[kind], ""]
        if not rows:
            lines += ["(none)", ""]
            continue
        lines += ["| id | text | correct | OpenJTalk | Beam | kept draft | type |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {c['id']} | {_cell(c['text'])} | {c['reading']} | {_cell(_marked(c, 'ojt'))} | "
                  f"{_cell(_marked(c, 'fix'))} | {'yes' if c['kept_draft'] else 'no'} |  |" for c in rows]
        lines.append("")
    return "\n".join(lines)

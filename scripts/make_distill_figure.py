"""The Part 2 figure for the cards: exact match by category on the 300 sentences, OpenJTalk, the frozen student and
Beam correcting OpenJTalk (`beam_fix`, low, run 0). Built from results/ only; no numbers are typed in.

    uv run scripts/make_distill_figure.py    # -> hf/figures/distill.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yomibench import ROOT  # noqa: E402
from yomibench.data import DATA_PATH, load_items  # noqa: E402
from yomibench.metrics import score  # noqa: E402

ORDER = ("homograph", "disaster", "number", "place", "medical", "government", "person")
# categorical slots 2, 3, 1 (dataviz validator, light: all pass; slot 3 is below 3:1 contrast, so every bar is labelled)
SERIES = (("OpenJTalk", "#eb6834"), ("student (this work)", "#1baf7a"), ("Beam correcting OpenJTalk", "#2a78d6"))
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def matches() -> dict[str, dict[str, bool]]:
    """system -> id -> exact match, all scored the same way as the report."""
    items = {i["id"]: i for i in load_items(DATA_PATH)}
    student = load_items(ROOT / "results" / "distill" / "eval300" / "student.jsonl")
    return {
        "OpenJTalk": {r["id"]: r["match"] for r in load_items(ROOT / "results" / "ojt.jsonl")},
        "student (this work)": {r["id"]: score(items[r["id"]], r["output"])["match"] for r in student},
        "Beam correcting OpenJTalk": {r["id"]: r["match"]
                                      for r in load_items(ROOT / "results" / "beam_fix-low-en-r0.jsonl")},
    }


def main() -> None:
    items = load_items(DATA_PATH)
    m = matches()
    groups = [("all", [i["id"] for i in items])] + [(c, [i["id"] for i in items if i["category"] == c]) for c in ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    height, step = 0.26, 0.29
    for g, (label, ids) in enumerate(reversed(groups)):
        for k, (name, color) in enumerate(SERIES):
            y = g + (1 - k) * step
            rate = 100 * sum(m[name][i] for i in ids) / len(ids)
            ax.barh(y, rate, height=height, color=color, edgecolor=SURFACE, linewidth=1)
            ax.text(rate + 1, y, f"{rate:.0f}", ha="left", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{label} ({len(ids)})" for label, ids in reversed(groups)], fontsize=9, color=INK)
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("exact match (%)", fontsize=9, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(handles=[Patch(color=color, label=name) for name, color in SERIES], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8, labelcolor=INK,
              handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
    ax.set_title("Part 1's 300 human-approved sentences, scored once after freezing the student", loc="left",
                 fontsize=10, color=INK, pad=22)
    fig.tight_layout()
    out = ROOT / "hf" / "figures" / "distill.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()

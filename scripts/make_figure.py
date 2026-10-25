"""The results figure for the cards and the Space: exact match by category, OpenJTalk -> Beam fixing
OpenJTalk's reading (`beam_fix`, low, run 0). Built from results/ only; no numbers are typed in.

    uv run scripts/make_figure.py            # -> hf/figures/results.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yomibench import ROOT  # noqa: E402
from yomibench.data import load_items  # noqa: E402

ORDER = ("homograph", "disaster", "number", "place", "medical", "government", "person")
OJT_COLOR, BEAM_COLOR = "#eb6834", "#2a78d6"  # categorical slots 2 and 1 (validated: dataviz validator, light)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def rates(name: str) -> dict[str, tuple[float, int]]:
    rows = load_items(ROOT / "results" / f"{name}.jsonl")
    out = {"all": (sum(r["match"] for r in rows) / len(rows), len(rows))}
    for cat in ORDER:
        part = [r for r in rows if r["category"] == cat]
        out[cat] = (sum(r["match"] for r in part) / len(part), len(part))
    return out


def main() -> None:
    ojt, beam = rates("ojt"), rates("beam_fix-low-en-r0")
    labels = ["all"] + list(ORDER)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for y, key in enumerate(reversed(labels)):
        a, b = 100 * ojt[key][0], 100 * beam[key][0]
        ax.plot([a, b], [y, y], color=GRID, linewidth=2, zorder=1, solid_capstyle="round")
        ax.scatter([a], [y], s=70, color=OJT_COLOR, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([b], [y], s=70, color=BEAM_COLOR, edgecolor=SURFACE, linewidth=2, zorder=3)
        if round(a) == round(b):
            ax.text(b + 2, y, f"{b:.0f} (both)", ha="left", va="center", fontsize=8, color=INK)
            continue
        left, right = (a, b) if a <= b else (b, a)
        ax.text(left - 2, y, f"{left:.0f}", ha="right", va="center", fontsize=8, color=MUTED)
        ax.text(right + 2, y, f"{right:.0f}", ha="left", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{k} ({ojt[k][1]})" for k in reversed(labels)], fontsize=9, color=INK)
    ax.set_xlim(20, 112)
    ax.set_xticks(range(20, 101, 20))
    ax.set_xlabel("exact match (%)", fontsize=9, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.scatter([], [], s=50, color=OJT_COLOR, label="OpenJTalk (piper-plus front end)")
    ax.scatter([], [], s=50, color=BEAM_COLOR, label="Beam correcting OpenJTalk (low, run 0)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8, labelcolor=INK,
              handletextpad=0.3, columnspacing=1.2)
    ax.set_title("Readings of 300 human-approved Japanese sentences", loc="left", fontsize=10, color=INK, pad=22)
    fig.tight_layout()
    out = ROOT / "hf" / "figures" / "results.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()

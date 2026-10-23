"""Score one output against the gold reading and its alternatives."""

from yomibench.normalize import has_non_reading, normalize


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(item: dict, output: str, truncated: bool = False) -> dict:
    """Exact match (after normalization), CER against the closest gold, and format issue.

    An empty output that was truncated counts as truncated only (docs/02), not also as a format issue.
    """
    pred = normalize(output)
    golds = [normalize(g) for g in [item["reading"], *item["reading_alts"]]]
    return {
        "match": pred in golds,
        "cer": min(edit_distance(pred, g) / len(g) for g in golds),
        "format_issue": has_non_reading(output) or (not output.strip() and not truncated),
    }

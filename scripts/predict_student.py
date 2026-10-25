"""Run a trained student on a set of items and apply the gate (plan 2026-10-24, Task 5).

Each item needs `id` and `text`; the OpenJTalk draft is computed here (the same `ojt` mode as the
benchmark). Greedy decoding; an output that hits --max-new-tokens without EOS is marked truncated.

    uv run --group train scripts/predict_student.py --model models/s440-e2 --items data/fresh/dev.jsonl \
        --out results/distill/dev/s440-e2.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yomibench.data import load_items  # noqa: E402
from yomibench.student import format_prompt, gate, generate  # noqa: E402
from yomibench.systems import openjtalk  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-ratio", type=float, default=0.8)
    ap.add_argument("--max-ratio", type=float, default=1.25)
    args = ap.parse_args()

    items = load_items(args.items)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for i in range(0, len(items), args.batch):
            chunk = items[i:i + args.batch]
            drafts = [openjtalk.read(it["text"]) for it in chunk]
            gens = generate(model, tok, [format_prompt(it["text"], d) for it, d in zip(chunk, drafts)],
                            args.max_new_tokens)
            for it, draft, (raw, truncated) in zip(chunk, drafts, gens):
                output, fallback = gate(draft, raw, truncated, args.min_ratio, args.max_ratio)
                f.write(json.dumps({"id": it["id"], "ojt": draft, "raw": raw, "output": output,
                                    "fallback": fallback, "truncated": truncated}, ensure_ascii=False) + "\n")
    print(f"{len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()

"""Train the student on the teacher pairs where Beam's two readings agree (plan 2026-10-24, Task 5).

Input: data/distill/teacher.jsonl (rows with agree == true). Prompt: yomibench.student.format_prompt(text,
ojt); target: clean_target(fix) + EOS. The loss covers the target only. Full fine-tuning, or LoRA with
--lora (merged into the base weights before saving, so inference needs no peft).

    uv run --group train scripts/train_student.py --base llm-jp/llm-jp-3-440m --epochs 2 --out models/s440-e2
    uv run --group train scripts/train_student.py --base llm-jp/llm-jp-3-1.8b --lora --epochs 1 --out models/s18l-e1
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yomibench import ROOT  # noqa: E402
from yomibench.data import load_items  # noqa: E402
from yomibench.student import clean_target, format_prompt  # noqa: E402

TEACHER = ROOT / "data" / "distill" / "teacher.jsonl"
MAX_LEN = 256


def encode(tok, row: dict) -> tuple[list[int], list[int]]:
    prompt = tok(format_prompt(row["text"], row["ojt"]), add_special_tokens=False)["input_ids"]
    target = tok(clean_target(row["fix"]), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = ([tok.bos_token_id] if tok.bos_token_id is not None else []) + prompt + target
    labels = [-100] * (len(ids) - len(target)) + target
    return ids[:MAX_LEN], labels[:MAX_LEN]


def batches(examples, batch_size: int, pad_id: int, rng: random.Random):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [examples[j] for j in order[i:i + batch_size]]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.tensor([x + [pad_id] * (width - len(x)) for x, _ in chunk])
        labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in chunk])
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in chunk])
        yield ids, labels, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="llm-jp/llm-jp-3-440m")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None, help="default 2e-5 full, 2e-4 LoRA")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", type=Path, default=TEACHER)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lr = args.lr or (2e-4 if args.lora else 2e-5)

    torch.manual_seed(args.seed)
    rows = [r for r in load_items(args.teacher) if r["agree"]]
    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    examples = [encode(tok, r) for r in rows]
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda")
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = math.ceil(len(examples) / args.batch / args.accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, max(1, steps // 20), steps)
    rng = random.Random(args.seed)
    model.train()
    t0, step, losses = time.time(), 0, []
    for epoch in range(args.epochs):
        for k, (ids, labels, mask) in enumerate(batches(examples, args.batch, pad_id, rng), 1):
            out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda())
            (out.loss / args.accum).backward()
            losses.append(out.loss.item())
            if k % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step}/{steps} loss {sum(losses[-50:]) / len(losses[-50:]):.4f}",
                          flush=True)
    if args.lora:
        model = model.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    config = {"base": args.base, "lora": args.lora, "epochs": args.epochs, "lr": lr, "batch": args.batch,
              "accum": args.accum, "seed": args.seed, "examples": len(examples), "steps": steps,
              "final_loss": round(sum(losses[-50:]) / max(len(losses[-50:]), 1), 4),
              "minutes": round((time.time() - t0) / 60, 1), "teacher_rows": args.teacher.resolve().relative_to(ROOT).as_posix()}
    (args.out / "train_config.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(config))


if __name__ == "__main__":
    main()

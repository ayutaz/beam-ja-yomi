"""G0 check: can this machine fine-tune the student on the GPU? (plan 2026-10-24, Task 1)

Loads the base model in bf16, takes 20 optimizer steps on one sentence, and prints whether the loss
went down and the peak GPU memory. Nothing is saved.

    uv run --group train scripts/check_gpu_train.py
    uv run --group train scripts/check_gpu_train.py --base llm-jp/llm-jp-3-1.8b --lora
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SAMPLE = "文: このカレーは辛い\n下書き: コノカレーワツライ\n読み: コノカレーワカライ"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="llm-jp/llm-jp-3-440m")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda")
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4 if args.lora else 1e-5)
    batch = tok([SAMPLE] * 8, return_tensors="pt").to("cuda")
    losses = []
    for _ in range(args.steps):
        out = model(**batch, labels=batch["input_ids"])
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(out.loss.item())
    print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}", "down" if losses[-1] < losses[0] else "NOT down")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()

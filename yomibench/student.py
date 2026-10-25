"""The student: a small causal LM that reads (text, OpenJTalk draft) and writes the corrected reading.

One prompt format, shared by training and inference. The gate falls back to the OpenJTalk draft when
the student's output cannot be trusted (not kana only, empty, cut off, or far longer/shorter than the
draft), so a bad generation never makes the reading worse than the front end's.
"""

import re

from yomibench.normalize import normalize, to_katakana

PROMPT = "文: {text}\n下書き: {draft}\n読み: "
_KANA_ONLY = re.compile(r"[ァ-ヺー]+")


def format_prompt(text: str, draft: str) -> str:
    return PROMPT.format(text=text, draft=draft)


def clean_target(reading: str) -> str:
    """Katakana and ー only (teacher outputs sometimes keep 。 or 、)."""
    return "".join(c for c in to_katakana(reading) if "ァ" <= c <= "ヺ" or c == "ー")


def gate(draft: str, output: str, truncated: bool, min_ratio: float = 0.8,
         max_ratio: float = 1.25) -> tuple[str, bool]:
    """Return (reading to use, fell back to the draft?)."""
    out = output.strip()
    if truncated or not _KANA_ONLY.fullmatch(out):
        return draft, True
    ratio = len(normalize(out)) / max(len(normalize(draft)), 1)
    if not min_ratio <= ratio <= max_ratio:
        return draft, True
    return out, False


def generate(model, tok, prompts: list[str], max_new_tokens: int) -> list[tuple[str, bool]]:
    """Greedy decoding of a batch, left-padded. Returns (text, truncated) per prompt; truncated = no eos."""
    import torch  # only the training / inference environments have it

    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    seqs = [bos + tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    width = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs]).to(model.device)
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs]).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=pad, eos_token_id=tok.eos_token_id)
    results = []
    for row in out[:, width:].tolist():
        truncated = tok.eos_token_id not in row
        if not truncated:
            row = row[:row.index(tok.eos_token_id)]
        results.append((tok.decode(row, skip_special_tokens=True), truncated))
    return results

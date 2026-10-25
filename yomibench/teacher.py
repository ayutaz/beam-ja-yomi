"""Beam as the teacher: read each pool sentence two ways and keep the pairs where the two agree.

Beam's readings here are training material, not gold. Only human-checked readings are used to score
(docs/05, CLAUDE.md). Prices are the Hugging Face router's for Beam via Baseten, checked 2026-10-24:
$0.40 per million input tokens and $1.20 per million output tokens.
"""

import json
import threading
from pathlib import Path

from yomibench.normalize import has_non_reading, normalize
from yomibench.systems.beam import build_messages, parse_response

USD_PER_M_INPUT, USD_PER_M_OUTPUT = 0.40, 1.20


def agree(direct: dict, fix: dict) -> bool:
    """Both parsed responses complete, kana only, non-empty, and equal after normalization."""
    def ok(p):
        return not p["truncated"] and p["output"].strip() and not has_non_reading(p["output"])
    return bool(ok(direct) and ok(fix) and normalize(direct["output"]) == normalize(fix["output"])
                and normalize(direct["output"]))


def cost_usd(usage_rows: list[dict]) -> float:
    return sum(r["prompt_tokens"] * USD_PER_M_INPUT + r["completion_tokens"] * USD_PER_M_OUTPUT
               for r in usage_rows) / 1_000_000


def teach_one(beam, text: str, draft: str, effort: str = "low") -> dict:
    raw = {kind: beam.complete(build_messages(kind, "en", text, draft if kind == "fix" else None), effort, run=0)
           for kind in ("direct", "fix")}
    parsed = {kind: parse_response(data) for kind, data in raw.items()}
    usage = [data.get("usage") or {} for data in raw.values()]
    return {
        "ojt": draft,
        "direct": parsed["direct"]["output"],
        "fix": parsed["fix"]["output"],
        "direct_finish": parsed["direct"]["finish"],
        "fix_finish": parsed["fix"]["finish"],
        "agree": agree(parsed["direct"], parsed["fix"]),
        "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
        "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
        "reasoning_tokens": sum(p["reasoning_tokens"] or 0 for p in parsed.values()),
    }


class Budget:
    """Running cost of teach_one rows, cached or not, so a resumed run stops at the same cap."""

    def __init__(self, limit_usd: float):
        self.limit_usd = limit_usd
        self.spent = 0.0

    def add(self, row: dict) -> None:
        self.spent += cost_usd([row])

    @property
    def exceeded(self) -> bool:
        return self.spent >= self.limit_usd


class DailyCapReached(Exception):
    """This run's own cap on new output tokens is spent (the rest of the day's allowance is for the Space)."""


class TokenMeter:
    """Wraps chat.completions.create: counts output tokens of calls that really went out (cache hits never
    reach it) and refuses new calls once `cap` is reached. Thread-safe."""

    def __init__(self, create, cap: int, used: int = 0):
        self._create, self.cap, self.used = create, cap, used
        self._lock = threading.Lock()

    def create(self, **request):
        with self._lock:
            if self.used >= self.cap:
                raise DailyCapReached(f"{self.used} output tokens used (cap {self.cap})")
        resp = self._create(**request)
        usage = resp["usage"] if isinstance(resp, dict) else resp.usage.model_dump()
        with self._lock:
            self.used += usage.get("completion_tokens") or 0
        return resp


def tokens_since(cache_dir: Path, since: float) -> int:
    """Output tokens of cached responses written at or after `since` (a timestamp): what this machine
    already spent of the provider's daily allowance, for a run restarted the same day."""
    total = 0
    for path in Path(cache_dir).glob("*.json"):
        if path.stat().st_mtime >= since:
            usage = json.loads(path.read_text(encoding="utf-8")).get("usage") or {}
            total += usage.get("completion_tokens") or 0
    return total


HF_MODEL = "reflection/Beam-1.0"  # the `model` field of responses that came through HF (Baseten)


def cache_cost_usd(cache_dir: Path, model: str = HF_MODEL) -> float:
    """Cost of every cached response from `model`, including calls whose row was never finished (the other call
    of the pair failed, or the run stopped at the budget): what was really paid, unlike Budget."""
    total = 0.0
    for path in Path(cache_dir).glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model") == model:
            total += cost_usd([{"prompt_tokens": 0, "completion_tokens": 0,
                                **{k: v or 0 for k, v in (data.get("usage") or {}).items()
                                   if k in ("prompt_tokens", "completion_tokens")}}])
    return total

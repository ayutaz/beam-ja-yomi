"""Beam readings via the Reflection API (OpenAI-compatible), with an on-disk cache.

Prompts are copied verbatim from docs/02 ("プロンプト"). Every response is cached under
cache/beam/ keyed by (base_url, request, run), so the same call is never made twice and an
interrupted run resumes for free. Keys come from .env and are never printed.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

from yomibench import ROOT

PROVIDERS = {
    "reflection": {
        "base_url": "https://api.reflection.ai/openai/v1",
        "model": "Beam-1.0-501B-A23B",
        "key_env": "REFLECTION_API_KEY",
    },
    "hf": {
        "base_url": "https://router.huggingface.co/v1",
        "model": "reflection/Beam-1.0-501B-A23B:baseten",
        "key_env": "HF_TOKEN",
    },
}

TEMPERATURE = 0.7
TOP_P = 0.99
MAX_COMPLETION_TOKENS = 16000

PROMPTS = {
    ("direct", "en"): (
        "Convert the user's Japanese text into its reading, written only in katakana, exactly as it is "
        "pronounced aloud. Write particles as pronounced (は→ワ, へ→エ, を→オ). Spell out numbers, units and "
        "symbols as they are read. Output only the katakana reading, with no explanation, punctuation, or spaces."
    ),
    ("fix", "en"): (
        "You will receive Japanese text and a draft katakana reading produced by a dictionary-based converter "
        "(OpenJTalk) for text-to-speech. The draft may contain misreadings, especially for numbers, units, "
        "symbols, words whose reading depends on context, and proper nouns. Output the full corrected reading "
        "of the text, written only in katakana, exactly as it is pronounced aloud. Keep the draft wherever it "
        "is already correct. Write particles as pronounced (は→ワ, へ→エ, を→オ). Output only the katakana "
        "reading, with no explanation, punctuation, or spaces."
    ),
    ("direct", "ja"): (
        "ユーザーの日本語の文を、声に出して読むとおりにカタカナだけで書いてください。助詞は発音どおりに書きます"
        "（は→ワ、へ→エ、を→オ）。数字・単位・記号も読み方で書きます。読み以外の説明・句読点・空白は書かないでください。"
    ),
}


class DailyLimitReached(Exception):
    """The provider's daily allowance is spent (429 daily_allowance_exhausted). Do not retry today."""


def build_messages(kind: str, lang: str, text: str, draft: str | None = None) -> list[dict]:
    system = PROMPTS[(kind, lang)]
    if kind == "fix":
        if draft is None:
            raise ValueError("a 'fix' prompt needs the OpenJTalk draft reading")
        user = f"Text: {text}\nDraft reading: {draft}"
    else:
        user = text
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _error_code(err: openai.APIStatusError) -> str | None:
    code = getattr(err, "code", None)
    body = err.body if isinstance(err.body, dict) else {}
    return code or body.get("code") or (body.get("error") or {}).get("code")


MEMORY_CACHE_SIZE = 500


class CacheMiss(Exception):
    """offline=True and the request is not in the cache. Nothing was sent."""


class BeamClient:
    def __init__(self, provider: str = "reflection", client=None, cache_dir: Path | None = ROOT / "cache" / "beam",
                 offline: bool = False, timeout: float = 600, max_retries: int = 5):
        """cache_dir=None keeps responses in memory only (the Space: visitors' text never touches the disk)."""
        self.provider = provider
        self.cfg = PROVIDERS[provider]
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._memory: dict[str, dict] = {}
        self.offline = offline
        if client is None and not offline:
            load_dotenv(ROOT / ".env")
            api_key = os.environ.get(self.cfg["key_env"])
            if not api_key:
                raise RuntimeError(f"{self.cfg['key_env']} is not set in .env")
            client = OpenAI(base_url=self.cfg["base_url"], api_key=api_key, max_retries=max_retries,
                            timeout=timeout)
        self.client = client

    def complete(self, messages: list[dict], effort: str, run: int) -> dict:
        request = {
            "model": self.cfg["model"],
            "messages": messages,
            "reasoning_effort": effort,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        }
        key_src = json.dumps([self.cfg["base_url"], request, run], ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
        path = self.cache_dir / f"{key}.json" if self.cache_dir is not None else None
        if path is None and key in self._memory:
            return self._memory[key]
        if path is not None and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.offline:
            raise CacheMiss(f"{key}.json")
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(**request)
        except openai.RateLimitError as err:
            if _error_code(err) == "daily_allowance_exhausted":
                raise DailyLimitReached(str(err)) from err
            raise
        data = resp.model_dump()
        data["_elapsed"] = time.time() - t0
        if path is None:
            if len(self._memory) >= MEMORY_CACHE_SIZE:
                self._memory.pop(next(iter(self._memory)))
            self._memory[key] = data
            return data
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")  # write then rename, so an interrupted run never leaves a broken cache file
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
        return data


def parse_response(data: dict) -> dict:
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    finish = choice.get("finish_reason")
    return {
        "output": (choice["message"].get("content") or "").strip(),
        "finish": finish,
        "truncated": finish == "length",
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": round(float(data.get("_elapsed", 0.0)), 2),
    }

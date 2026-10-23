"""Test doubles for the OpenAI client (no network)."""

import httpx2
import openai
import pytest


class _Resp:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return dict(self._data)


def _ok_response(req):
    return {
        "id": "fake",
        "model": req["model"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "ハナテン", "reasoning_content": "..."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22,
                  "completion_tokens_details": {"reasoning_tokens": 8}},
    }


def daily_limit_error(body):
    resp = httpx2.Response(429, request=httpx2.Request("POST", "https://api.reflection.ai/openai/v1/chat/completions"))
    return openai.RateLimitError("Daily allowance exhausted", response=resp, body=body)


class FakeOpenAI:
    """Mimics `client.chat.completions.create(**req)`; can raise the daily limit after N calls."""

    def __init__(self, limit_after=None, error_body=None):
        self.calls = 0
        self.last_request = None
        self.limit_after = limit_after
        self.error_body = error_body or {"error": {"code": "daily_allowance_exhausted"}}
        self.chat = self
        self.completions = self

    def reset_limit(self):
        self.limit_after = None

    def create(self, **req):
        if self.limit_after is not None and self.calls >= self.limit_after:
            raise daily_limit_error(self.error_body)
        self.calls += 1
        self.last_request = req
        return _Resp(_ok_response(req))


@pytest.fixture
def fake_openai():
    return FakeOpenAI()


@pytest.fixture
def fake_openai_daily_limit():
    return FakeOpenAI(limit_after=0)


@pytest.fixture
def fake_openai_limit_after_1():
    return FakeOpenAI(limit_after=1)


@pytest.fixture
def items():
    base = {"reading_alts": [], "source": "test", "note": "", "verified": True}
    return [
        {**base, "id": "number-0001", "category": "number", "text": "1.5mの高さ", "reading": "イッテンゴメートルノタカサ",
         "reading_alts": ["イチテンゴメートルノタカサ"]},
        {**base, "id": "homograph-0001", "category": "homograph", "text": "今日は市場に行く", "reading": "キョウワイチバニイク"},
        {**base, "id": "place-0001", "category": "place", "text": "放出駅で降りる", "reading": "ハナテンエキデオリル"},
    ]

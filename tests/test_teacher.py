import pytest

from yomibench.teacher import Budget, agree, cost_usd, teach_one


def _parsed(output, finish="stop"):
    return {"output": output, "finish": finish, "truncated": finish == "length"}


def test_agree_true_when_normalized_equal():
    assert agree(_parsed("トーキョー"), _parsed("トウキョウ"))


def test_agree_false_when_different():
    assert not agree(_parsed("カライ"), _parsed("ツライ"))


def test_agree_false_when_truncated():
    assert not agree(_parsed("カライ", "length"), _parsed("カライ"))


def test_agree_false_when_kanji():
    assert not agree(_parsed("辛イ"), _parsed("辛イ"))


def test_agree_false_when_empty():
    assert not agree(_parsed(""), _parsed(""))


def test_cost_usd():
    assert cost_usd([{"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}]) == pytest.approx(1.60)


class FakeBeam:
    """Stands in for BeamClient: answers カライ to both prompts with fixed usage; counts real calls."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, effort, run):
        self.calls += 1
        return {"choices": [{"message": {"content": "カライ"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 400,
                          "completion_tokens_details": {"reasoning_tokens": 300}}}


def test_teach_one():
    row = teach_one(FakeBeam(), "辛い", "ツライ")
    assert row["ojt"] == "ツライ" and row["direct"] == "カライ" and row["fix"] == "カライ"
    assert row["agree"] is True
    assert row["prompt_tokens"] == 200 and row["completion_tokens"] == 800 and row["reasoning_tokens"] == 600


def test_budget_counts_cached_usage():
    """A resumed run gets cached responses back; their usage still counts, so the cap is never exceeded."""
    budget = Budget(limit_usd=0.003)
    row = teach_one(FakeBeam(), "辛い", "ツライ")  # 200 in + 800 out = 0.00104 USD
    budget.add(row)
    assert not budget.exceeded
    budget.add(row)
    assert not budget.exceeded
    budget.add(row)
    assert budget.exceeded
    assert budget.spent == pytest.approx(3 * 0.00104)


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        return {"usage": {"completion_tokens": 400}}


def test_token_meter_counts_only_real_calls_and_stops_at_cap():
    from yomibench.teacher import DailyCapReached, TokenMeter
    inner = FakeCompletions()
    meter = TokenMeter(inner.create, cap=1000)
    meter.create(model="m")
    meter.create(model="m")
    assert meter.used == 800
    meter.create(model="m")  # 1200: this call already went out, so it is kept
    with pytest.raises(DailyCapReached):
        meter.create(model="m")
    assert inner.calls == 3


def test_tokens_since_counts_only_recent_cache_files(tmp_path):
    import json
    import os
    from yomibench.teacher import tokens_since
    old, new = tmp_path / "a.json", tmp_path / "b.json"
    for p in (old, new):
        p.write_text(json.dumps({"usage": {"completion_tokens": 300}}), encoding="utf-8")
    os.utime(old, (1000, 1000))
    os.utime(new, (5000, 5000))
    assert tokens_since(tmp_path, since=2000) == 300


def test_token_meter_starts_from_tokens_already_used():
    from yomibench.teacher import DailyCapReached, TokenMeter
    meter = TokenMeter(FakeCompletions().create, cap=1000, used=1000)
    with pytest.raises(DailyCapReached):
        meter.create(model="m")


def test_cache_cost_counts_every_cached_response_of_the_model(tmp_path):
    import json

    from yomibench.teacher import USD_PER_M_INPUT, USD_PER_M_OUTPUT, cache_cost_usd

    for name, model, usage in [("a", "hf-model", {"prompt_tokens": 100, "completion_tokens": 10}),
                               ("b", "hf-model", {"prompt_tokens": 200, "completion_tokens": 20}),
                               ("c", "other", {"prompt_tokens": 999, "completion_tokens": 999})]:
        (tmp_path / f"{name}.json").write_text(json.dumps({"model": model, "usage": usage}), encoding="utf-8")
    assert cache_cost_usd(tmp_path, "hf-model") == (300 * USD_PER_M_INPUT + 30 * USD_PER_M_OUTPUT) / 1_000_000

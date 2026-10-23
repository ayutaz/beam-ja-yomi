import pytest

from tests.conftest import FakeOpenAI
from yomibench.systems.beam import BeamClient, DailyLimitReached, build_messages, parse_response


def test_fix_messages():
    m = build_messages("fix", "en", "市場に行く", "シジョーニイク")
    assert m[0]["role"] == "system" and "OpenJTalk" in m[0]["content"]
    assert m[1] == {"role": "user", "content": "Text: 市場に行く\nDraft reading: シジョーニイク"}


def test_direct_messages_in_japanese():
    m = build_messages("direct", "ja", "放出駅")
    assert m[0]["content"].startswith("ユーザーの日本語の文を") and m[1]["content"] == "放出駅"


def test_cache_hit_skips_api_and_run_index_is_part_of_key(tmp_path, fake_openai):
    bc = BeamClient(client=fake_openai, cache_dir=tmp_path)
    msgs = build_messages("direct", "en", "放出駅")
    bc.complete(msgs, "low", run=0)
    bc.complete(msgs, "low", run=0)
    bc.complete(msgs, "low", run=1)
    assert fake_openai.calls == 2


def test_request_parameters(tmp_path, fake_openai):
    BeamClient(client=fake_openai, cache_dir=tmp_path).complete(build_messages("direct", "en", "x"), "high", 0)
    req = fake_openai.last_request
    assert (req["model"], req["reasoning_effort"], req["temperature"], req["top_p"], req["max_completion_tokens"]) \
        == ("Beam-1.0-501B-A23B", "high", 0.7, 0.99, 16000)


def test_parse_normal_response(tmp_path, fake_openai):
    data = BeamClient(client=fake_openai, cache_dir=tmp_path).complete(build_messages("direct", "en", "x"), "low", 0)
    r = parse_response(data)
    assert r["output"] == "ハナテン" and r["truncated"] is False and r["reasoning_tokens"] == 8
    assert r["completion_tokens"] == 12 and r["seconds"] >= 0


def test_truncated_response_with_null_content():
    data = {"choices": [{"message": {"content": None}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 16000, "completion_tokens_details": {"reasoning_tokens": 15999}},
            "_elapsed": 3.0}
    r = parse_response(data)
    assert r["output"] == "" and r["truncated"] is True and r["reasoning_tokens"] == 15999


def test_daily_limit_raises(tmp_path, fake_openai_daily_limit):
    with pytest.raises(DailyLimitReached):
        BeamClient(client=fake_openai_daily_limit, cache_dir=tmp_path).complete(
            build_messages("direct", "en", "x"), "low", 0)


def test_daily_limit_with_unwrapped_error_body(tmp_path):
    fake = FakeOpenAI(limit_after=0, error_body={"code": "daily_allowance_exhausted"})
    with pytest.raises(DailyLimitReached):
        BeamClient(client=fake, cache_dir=tmp_path).complete(build_messages("direct", "en", "x"), "low", 0)


def test_missing_key_message_does_not_leak(tmp_path, monkeypatch):
    monkeypatch.setattr("yomibench.systems.beam.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("REFLECTION_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="REFLECTION_API_KEY is not set"):
        BeamClient(cache_dir=tmp_path)


def test_prompts_match_docs_verbatim():
    from yomibench import ROOT
    from yomibench.systems.beam import PROMPTS
    design = ROOT / "docs" / "02_ベンチマークの設計.md"
    if not design.exists():
        pytest.skip("the design notes are not part of the public repository")
    section = design.read_text(encoding="utf-8").split("## プロンプト")[1]
    blocks = [b.strip() for b in section.split("```")[1::2]]
    assert all(p in blocks for p in PROMPTS.values())


def test_offline_uses_cache_and_never_calls_api(tmp_path, fake_openai):
    msgs = build_messages("direct", "en", "x")
    first = BeamClient(client=fake_openai, cache_dir=tmp_path).complete(msgs, "low", 0)
    offline = BeamClient(cache_dir=tmp_path, offline=True)
    assert offline.complete(msgs, "low", 0) == first
    assert fake_openai.calls == 1


def test_offline_cache_miss_raises_without_calling(tmp_path):
    from yomibench.systems.beam import CacheMiss
    with pytest.raises(CacheMiss):
        BeamClient(cache_dir=tmp_path, offline=True).complete(build_messages("direct", "en", "x"), "low", 0)


def test_offline_needs_no_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("REFLECTION_API_KEY", raising=False)
    monkeypatch.setattr("yomibench.systems.beam.load_dotenv", lambda *a, **k: None)
    assert BeamClient(cache_dir=tmp_path, offline=True).client is None


def test_interrupted_cache_write_leaves_no_broken_file(tmp_path, fake_openai, monkeypatch):
    import os
    def boom(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        BeamClient(client=fake_openai, cache_dir=tmp_path).complete(build_messages("direct", "en", "x"), "low", 0)
    assert list(tmp_path.glob("*.json")) == []


def test_memory_cache_when_no_cache_dir(tmp_path, fake_openai, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bc = BeamClient(client=fake_openai, cache_dir=None)
    msgs = build_messages("direct", "en", "x")
    assert bc.complete(msgs, "low", 0) == bc.complete(msgs, "low", 0)
    assert fake_openai.calls == 1 and list(tmp_path.rglob("*")) == []

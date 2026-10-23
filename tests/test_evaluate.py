import json

import pytest

from yomibench.evaluate import condition_name, load_drafts, run_condition
from yomibench.systems.beam import BeamClient


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_condition_names():
    assert condition_name("ojt", None, "en", 0) == "ojt"
    assert condition_name("ojt_vanilla", None, "en", 0) == "ojt_vanilla"
    assert condition_name("beam_fix", "low", "en", 2) == "beam_fix-low-en-r2"
    assert condition_name("beam_direct", "high", "ja", 0) == "beam_direct-high-ja-r0"


def test_ojt_rows_are_scored(items):
    rows, done = run_condition(items, "ojt")
    assert done is True and [r["id"] for r in rows] == [i["id"] for i in items]
    homograph = rows[1]
    assert homograph["output"] == "キョーワシジョーニイク" and homograph["match"] is False
    assert {"cer", "format_issue", "reading", "text", "category"} <= homograph.keys()


def test_beam_fix_requires_matching_ojt_results(tmp_path, items):
    with pytest.raises(ValueError, match="run --system ojt first"):
        load_drafts(tmp_path, items)
    write_jsonl(tmp_path / "ojt.jsonl", [{"id": i["id"], "text": i["text"], "output": "x"} for i in items])
    assert load_drafts(tmp_path, items) == {i["id"]: "x" for i in items}
    write_jsonl(tmp_path / "ojt.jsonl", [{"id": "number-0001", "text": "別の文", "output": "x"}])
    with pytest.raises(ValueError, match="stale"):
        load_drafts(tmp_path, items)


def test_beam_fix_sends_draft_and_records_it(tmp_path, items, fake_openai):
    drafts = {i["id"]: f"DRAFT{n}" for n, i in enumerate(items)}
    bc = BeamClient(client=fake_openai, cache_dir=tmp_path)
    rows, done = run_condition(items, "beam_fix", "low", beam=bc, drafts=drafts)
    assert done and rows[2]["draft"] == "DRAFT2"
    assert fake_openai.last_request["messages"][1]["content"].endswith("Draft reading: DRAFT2")
    assert rows[0]["provider"] == "reflection" and rows[0]["truncated"] is False


def test_daily_limit_mid_run_keeps_partial_and_resume_skips_done(tmp_path, items, fake_openai_limit_after_1):
    bc = BeamClient(client=fake_openai_limit_after_1, cache_dir=tmp_path)
    rows, done = run_condition(items, "beam_direct", "low", beam=bc)
    assert done is False and len(rows) == 1
    fake_openai_limit_after_1.reset_limit()
    rows, done = run_condition(items, "beam_direct", "low", beam=bc)
    assert done is True and len(rows) == len(items)
    assert fake_openai_limit_after_1.calls == len(items)  # 1件目は二度呼ばない


def test_offline_run_stops_on_first_cache_miss(tmp_path, items):
    from yomibench.systems.beam import CacheMiss
    beam = BeamClient(cache_dir=tmp_path, offline=True)
    with pytest.raises(CacheMiss):
        run_condition(items, "beam_direct", "low", "en", 0, beam=beam)

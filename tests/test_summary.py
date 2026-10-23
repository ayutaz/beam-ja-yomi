import json

from yomibench.summary import render_markdown, summarize


def write(tmp_path, name, rows):
    (tmp_path / f"{name}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def base(i, match):
    return {"id": f"number-000{i}", "category": "number", "match": match, "cer": 0.0 if match else 0.5,
            "format_issue": False}


def beam(row, truncated=False, reasoning=100):
    return {**row, "truncated": truncated, "reasoning_tokens": reasoning, "completion_tokens": reasoning + 10,
            "seconds": 1.0}


def test_fixed_and_broken_counts(tmp_path):
    write(tmp_path, "ojt", [base(1, False), base(2, True), base(3, False)])
    write(tmp_path, "beam_fix-low-en-r0", [beam(base(1, True)), beam(base(2, False)),
                                           beam(base(3, False), truncated=True, reasoning=16000)])
    s = summarize(tmp_path)
    assert s["fix_vs_ojt"]["beam_fix-low-en-r0"] == {"fixed": 1, "broken": 1, "kept_correct": 0, "still_wrong": 1}
    assert s["conditions"]["beam_fix-low-en-r0"]["truncated"] == 1
    assert s["conditions"]["ojt"]["exact_rate"] == 1 / 3
    assert s["conditions"]["beam_fix-low-en-r0"]["by_category"]["number"]["n"] == 3


def test_run_spread_and_ignored_files(tmp_path):
    write(tmp_path, "ojt", [base(1, True)])
    for run, m in enumerate([True, False, True]):
        write(tmp_path, f"beam_direct-low-en-r{run}", [beam(base(1, m))])
    write(tmp_path, "beam_direct-high-en-r0.partial", [beam(base(1, True))])   # 途中で止まった条件
    write(tmp_path, "probe", [{"id": "x"}])                                     # データのファイル
    s = summarize(tmp_path)
    assert set(s["conditions"]) == {"ojt", "beam_direct-low-en-r0", "beam_direct-low-en-r1", "beam_direct-low-en-r2"}
    assert s["run_spread"]["beam_direct-low-en"] == {"runs": 3, "min": 0.0, "max": 1.0}


def test_markdown_has_the_three_tables(tmp_path):
    write(tmp_path, "ojt", [base(1, False)])
    write(tmp_path, "beam_fix-low-en-r0", [beam(base(1, True))])
    md = render_markdown(summarize(tmp_path))
    assert "## Overall" in md and "## By category" in md and "## Beam fix vs OpenJTalk" in md
    assert "| beam_fix-low-en-r0 |" in md


def _item(i, reading="イチ"):
    return {"id": f"number-000{i}", "category": "number", "text": f"文{i}", "reading": reading,
            "reading_alts": [], "source": "s", "note": "", "verified": True}


def _row(item, match=True):
    return {**{k: item[k] for k in ("id", "category", "text", "reading", "reading_alts")},
            "output": item["reading"], "match": match, "cer": 0.0, "format_issue": False}


def test_check_results_accepts_results_that_match_the_data(tmp_path):
    from yomibench.summary import check_results
    items = [_item(1), _item(2)]
    write(tmp_path, "ojt", [_row(i) for i in items])
    assert check_results(tmp_path, items) == []


def test_check_results_flags_missing_extra_and_stale_rows(tmp_path):
    from yomibench.summary import check_results
    items = [_item(1), _item(2)]
    write(tmp_path, "ojt", [_row(items[0])])                                     # number-0002 missing
    write(tmp_path, "beam_fix-low-en-r0", [_row(items[0]), _row(_item(2, "ニ")), _row(_item(3))])  # stale + extra
    problems = check_results(tmp_path, items)
    assert any("ojt" in p and "number-0002" in p for p in problems)
    assert any("beam_fix-low-en-r0" in p and "number-0002" in p for p in problems)
    assert any("beam_fix-low-en-r0" in p and "number-0003" in p for p in problems)


def test_check_results_needs_at_least_one_condition(tmp_path):
    from yomibench.summary import check_results
    assert check_results(tmp_path, [_item(1)])


def test_summarize_does_not_crash_when_a_fix_row_has_no_ojt_row(tmp_path):
    write(tmp_path, "ojt", [base(1, False)])
    write(tmp_path, "beam_fix-low-en-r0", [beam(base(1, True)), beam(base(2, True))])
    assert summarize(tmp_path)["fix_vs_ojt"]["beam_fix-low-en-r0"]["fixed"] == 1

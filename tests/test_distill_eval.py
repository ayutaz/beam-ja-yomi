from yomibench.distill_eval import bootstrap_ci, summarize_run


def _item(iid, reading):
    return {"id": iid, "reading": reading, "reading_alts": []}


def test_summarize_run_counts_fixed_and_broken():
    items = [_item("a", "カライ"), _item("b", "シモテ"), _item("c", "ウエ")]
    ojt = {"a": "ツライ", "b": "シモテ", "c": "ウエ"}       # OJT: a wrong, b and c right
    pred = {"a": "カライ", "b": "ヘタ", "c": "ウエ"}        # student fixes a, breaks b
    s = summarize_run(items, ojt, pred)
    assert s["n"] == 3
    assert s["ojt_match"] == 2 / 3 and s["match"] == 2 / 3
    assert s["fixed"] == 1 and s["broken"] == 1
    assert s["break_rate"] == 1 / 2


def test_summarize_run_normalizes_long_vowels():
    s = summarize_run([_item("a", "トウキョウ")], {"a": "トーキョー"}, {"a": "トーキョー"})
    assert s["match"] == 1.0 and s["fixed"] == 0 and s["broken"] == 0


def test_bootstrap_ci_contains_mean():
    lo, hi = bootstrap_ci([True] * 50 + [False] * 50)
    assert lo < 0.5 < hi
    assert 0.15 <= hi - lo <= 0.25


def test_bootstrap_ci_is_deterministic():
    m = [True] * 30 + [False] * 10
    assert bootstrap_ci(m) == bootstrap_ci(m)


def test_regate_applies_other_ratios_to_the_raw_outputs():
    from yomibench.distill_eval import regate

    rows = [{"id": "a", "ojt": "アイウエ", "raw": "アイウエオ", "truncated": False},  # 1.25 times the draft
            {"id": "b", "ojt": "カキ", "raw": "漢字", "truncated": False}]
    assert regate(rows, 0.8, 1.25) == {"a": "アイウエオ", "b": "カキ"}
    assert regate(rows, 0.8, 1.2) == {"a": "アイウエ", "b": "カキ"}


def test_choose_takes_match_then_fewer_broken_then_the_default_gate_then_list_order():
    from yomibench.distill_eval import choose

    cands = [{"name": "a", "min_ratio": 0.8, "max_ratio": 1.25, "match": 0.80, "broken": 3},
             {"name": "b", "min_ratio": 0.7, "max_ratio": 1.4, "match": 0.82, "broken": 5},
             {"name": "c", "min_ratio": 0.75, "max_ratio": 1.3, "match": 0.82, "broken": 4},
             {"name": "d", "min_ratio": 0.8, "max_ratio": 1.25, "match": 0.82, "broken": 4},
             {"name": "e", "min_ratio": 0.8, "max_ratio": 1.25, "match": 0.82, "broken": 4}]
    assert choose(cands)["name"] == "d"


def test_g1_needs_three_points_over_openjtalk_and_at_most_three_percent_broken():
    from yomibench.distill_eval import passes_g1

    assert passes_g1({"match": 0.78, "ojt_match": 0.75, "break_rate": 0.03})
    assert not passes_g1({"match": 0.779, "ojt_match": 0.75, "break_rate": 0.0})
    assert not passes_g1({"match": 0.90, "ojt_match": 0.75, "break_rate": 0.031})


def test_file_hashes_cover_every_file_in_the_model_dir(tmp_path):
    import hashlib

    from yomibench.distill_eval import file_hashes

    (tmp_path / "config.json").write_text("a", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"b")
    (tmp_path / "sub").mkdir()
    assert file_hashes(tmp_path) == {"config.json": hashlib.sha256(b"a").hexdigest(),
                                     "model.safetensors": hashlib.sha256(b"b").hexdigest()}


def test_summarize_run_gives_mean_cer():
    items = [_item("a", "カライ"), _item("b", "ウエ")]
    s = summarize_run(items, {"a": "ツライ", "b": "ウエ"}, {"a": "ツライ", "b": "ウエ"})
    assert s["cer"] == (1 / 3 + 0) / 2


def test_fallback_rate_counts_only_the_given_items():
    from yomibench.distill_eval import fallback_rate

    rows = [{"id": "a", "fallback": True}, {"id": "b", "fallback": False}, {"id": "c", "fallback": False}]
    assert fallback_rate(rows, {"a", "b"}) == 1 / 2
    assert fallback_rate([{"id": "a", "output": "ア"}], {"a"}) is None

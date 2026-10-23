from yomibench.cases import classify, mark_diff, render_cases


def _r(id_, cat, reading, output, match, text=None):
    return {"id": id_, "category": cat, "text": text or f"文{id_}", "reading": reading, "output": output, "match": match}


OJT = [_r("n-1", "number", "イチ", "ニ", False), _r("n-2", "number", "サン", "サン", True),
       _r("h-1", "homograph", "カタ", "ホウ", False), _r("h-2", "homograph", "ジョウズ", "ジョウズ", True)]
FIX = [_r("n-1", "number", "イチ", "イチ", True), _r("n-2", "number", "サン", "シ", False),
       _r("h-1", "homograph", "カタ", "ホウ", False), _r("h-2", "homograph", "ジョウズ", "ジョウズ", True)]


def test_classify_kinds():
    got = {c["id"]: c["kind"] for c in classify(OJT, FIX)}
    assert got == {"n-1": "fixed", "n-2": "broken", "h-1": "unfixed", "h-2": "kept"}


def test_unfixed_records_whether_beam_kept_the_draft():
    case = next(c for c in classify(OJT, FIX) if c["id"] == "h-1")
    assert case["kept_draft"] is True and case["ojt"] == "ホウ" and case["fix"] == "ホウ"


def test_mark_diff_brackets_only_the_differing_part():
    assert mark_diff("トウキョウ", "トウキョオ") == "トウキョ【オ】"
    assert mark_diff("アイ", "アイ") == "アイ"


def test_render_lists_broken_and_unfixed_with_a_type_column_but_not_kept():
    md = render_cases(classify(OJT, FIX), "beam_fix-low-en-r0")
    assert "## broken" in md and "## unfixed" in md and "## fixed" in md
    assert "n-2" in md and "h-1" in md and "h-2" not in md
    assert "| type |" in md
    assert md.index("## broken") < md.index("## unfixed") < md.index("## fixed")
    assert "| text |" in md and "文n-2" in md and "シ" in md


def test_render_escapes_pipes_in_text():
    case = classify([_r("x-1", "number", "ア", "イ", False, text="a|b")], [_r("x-1", "number", "ア", "ア", True, text="a|b")])
    assert r"a\|b" in render_cases(case, "c")


def test_correct_outputs_are_not_bracketed():
    md = render_cases(classify(OJT, FIX), "c")
    row = next(line for line in md.splitlines() if line.startswith("| n-1 |"))
    assert row.split(" | ")[4] == "イチ"  # Beam's correct reading, no 【】


def test_case_marks_ignore_spellings_that_scoring_treats_as_equal():
    case = classify([_r("x-1", "n", "コテイ", "コーテー。", False)], [_r("x-1", "n", "コテイ", "コテー", True)])
    md = render_cases(case, "c")
    row = next(line for line in md.splitlines() if line.startswith("| x-1 |"))
    assert row.split(" | ")[3].count("【") == 1 and "。" not in row.split(" | ")[3]

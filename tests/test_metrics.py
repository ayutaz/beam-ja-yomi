from yomibench.metrics import edit_distance, score

ITEM = {"reading": "イッテンゴメートル", "reading_alts": ["イチテンゴメートル"]}


def test_alt_reading_counts_as_match():
    assert score(ITEM, "イチテンゴメートル") == {"match": True, "cer": 0.0, "format_issue": False}


def test_cer_uses_closest_gold_and_empty_output():
    assert edit_distance("abc", "axc") == 1
    s = score(ITEM, "")
    assert s["match"] is False and s["cer"] == 1.0 and s["format_issue"] is True


def test_truncated_empty_output_is_not_also_a_format_issue():
    s = score(ITEM, "", truncated=True)
    assert s["match"] is False and s["format_issue"] is False
    assert score(ITEM, "漢字", truncated=True)["format_issue"] is True

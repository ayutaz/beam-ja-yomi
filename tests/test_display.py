from yomibench.display import diff_html, kana_to_romaji


def test_romaji():
    assert kana_to_romaji("ブタイノカミテ") == "butainokamite"
    assert kana_to_romaji("ジョーズ") == "joozu"
    assert kana_to_romaji("イッテンゴ") == "ittengo"
    assert kana_to_romaji("キンエン") == "kin'en"
    assert kana_to_romaji("マッチ") == "matchi"


def test_romaji_youon_n_before_ya_and_non_kana():
    assert kana_to_romaji("シャシン") == "shashin"
    assert kana_to_romaji("キンヨウビ") == "kin'youbi"
    assert kana_to_romaji("ホンヤ") == "hon'ya"
    assert kana_to_romaji("ニュース") == "nyuusu"
    assert kana_to_romaji("ツヅク") == "tsuzuku"
    assert kana_to_romaji("ファイル") == "fairu"
    assert kana_to_romaji("ハッピョウ") == "happyou"
    assert kana_to_romaji("A1、ア") == "A1、a"


def test_diff_marks_only_changed_part():
    assert diff_html("ブタイノカミテ", "ブタイノジョーズ") == "ブタイノ<mark>ジョーズ</mark>"
    assert "<script>" not in diff_html("ア", "<script>")
    assert diff_html("アイウ", "アイウ") == "アイウ"


def test_gold_diff_has_no_marks_when_the_output_is_correct():
    from yomibench.display import closest_gold, gold_diff_html
    assert gold_diff_html("トウキョウ", [], "トーキョー", True) == "トーキョー"
    assert gold_diff_html("イッテン", ["イチテン"], "イチテン", True) == "イチテン"
    assert closest_gold("イッテンゴ", ["イチテンゴ"], "イチテンロク") == "イチテンゴ"
    assert "<mark>" in gold_diff_html("イッテンゴ", ["イチテンゴ"], "イチテンロク", False)
    assert gold_diff_html("イッテンゴ", ["イチテンゴ"], "イチテンロク", False).startswith("イチテン<mark>")


def test_wrong_reading_is_marked_only_where_the_scored_forms_differ():
    from yomibench.display import gold_diff_html
    # 固定 コテイ written コーテー: only the extra vowel counts, not the ー/イ spelling or the punctuation
    got = gold_diff_html("カグノコテイモ", [], "カグノコーテーモ。", False)
    assert got.count("<mark>") == 1 and "。" not in got

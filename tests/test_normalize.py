from yomibench.normalize import has_non_reading, normalize


def test_long_vowel_variants_match():
    assert normalize("トーキョー") == normalize("トウキョウ") == "トオキョオ"
    assert normalize("センセー") == normalize("センセイ")
    assert normalize("ジューゴフン") == normalize("ジュウゴフン")


def test_hiragana_halfwidth_and_symbols():
    assert normalize("はなてん、えき。") == "ハナテンエキ"
    assert normalize("ｷｮｳ") == normalize("キョウ")
    assert normalize("ヲ") == "オ"


def test_wrapped_output_is_stripped_but_flagged():
    assert normalize("「ハナテン」\n") == "ハナテン"
    assert has_non_reading("Reading: ハナテン")
    assert has_non_reading("イッテンゴｍノタカサ")  # 全角 m（OpenJTalk が読まなかった）
    assert not has_non_reading("ハナテン、エキ")

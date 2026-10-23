import pytest

from yomibench.systems.openjtalk import read


def test_piper_plus_mode_matches_probe():
    assert read("舞台の上手から登場する") == "ブタイノジョーズカラトージョースル"
    assert read("高齢者等避難を発令しました", "ojt") == "コーレーシャトウヒナンヲハツレーシマシタ"


def test_vanilla_mode_differs():
    assert read("高齢者等避難を発令しました", "ojt_vanilla") == "コーレーシャナドヒナンヲハツレーシマシタ"


def test_unknown_mode():
    with pytest.raises(ValueError):
        read("今日", "mecab")

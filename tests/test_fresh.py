from yomibench.fresh import validate_fresh


def _item(**kw):
    item = {"id": "fresh-dev-0001", "category": "dev", "text": "辛いカレー。", "reading": "カライカレー",
            "reading_alts": [], "source": "Wikipedia", "note": "", "verified": True}
    return {**item, **kw}


def test_valid_items_have_no_errors():
    assert validate_fresh([_item(), _item(id="fresh-test-0001", category="test", text="甘いカレー。")]) == []


def test_bad_id():
    assert len(validate_fresh([_item(id="dev-0001")])) == 1


def test_bad_category():
    assert len(validate_fresh([_item(category="number")])) == 1


def test_id_and_category_must_match():
    assert len(validate_fresh([_item(id="fresh-test-0001")])) == 1


def test_non_kana_reading():
    assert len(validate_fresh([_item(reading="辛いカレー")])) == 1


def test_not_verified():
    assert len(validate_fresh([_item(verified=False)])) == 1

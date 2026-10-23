from yomibench.data import validate_items

OK = {"id": "number-0001", "category": "number", "text": "1.5mの高さ",
      "reading": "イッテンゴメートルノタカサ", "reading_alts": [], "source": "x",
      "note": "", "verified": True}


def test_valid_item_passes():
    assert validate_items([OK], require_verified=True) == []


def test_rejects_bad_fields():
    bad = [
        {**OK, "id": "num-1"},                                    # id の形
        {**OK, "id": "number-0002", "category": "food"},          # カテゴリ
        {**OK, "id": "number-0003", "reading": "いってんご"},      # ひらがなは不可
        {**OK, "id": "number-0004", "reading": "、。"},            # 正規化すると空
        {**OK, "id": "number-0005", "source": ""},                # 出典なし
        {**OK, "id": "number-0006", "verified": False},           # 確認前
        {**OK, "id": "number-0007", "reading_alts": ["ABC"]},     # 別解もカタカナだけ
    ]
    bad = [{**b, "text": f"{b['text']}{i}"} for i, b in enumerate(bad)]  # text の重複は別のテストで見る
    errors = validate_items(bad, require_verified=True)
    for item in bad:
        assert any(e.startswith(item["id"]) for e in errors), item["id"]


def test_missing_key():
    item = {k: v for k, v in OK.items() if k != "note"}
    assert any("missing" in e for e in validate_items([item], require_verified=False))


def test_duplicate_id_and_text():
    errors = validate_items([OK, {**OK}], require_verified=False)
    assert any("duplicate id" in e for e in errors)
    assert any("duplicate text" in e for e in errors)

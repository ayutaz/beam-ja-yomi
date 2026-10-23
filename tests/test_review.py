from yomibench.review import COLUMNS, apply_sheet, make_sheet


def _item(n, reading="イチ", alts=()):
    return {"id": f"number-{n:04d}", "category": "number", "text": f"文{n}", "reading": reading,
            "reading_alts": list(alts), "source": "Claude draft", "note": "", "verified": False}


ITEMS = [_item(1, alts=["ヒトツ", "ワン"]), _item(2), _item(3), _item(4)]


def _fill(sheet, answers):
    """answers: {id: (ok, fixed_reading, fixed_alts, comment)}"""
    lines = sheet.lstrip("\ufeff").splitlines()
    out = [lines[0]]
    for line in lines[1:]:
        cells = line.split("\t")
        cells[6:10] = answers[cells[0]]
        out.append("\t".join(cells))
    return "\ufeff" + "\r\n".join(out) + "\r\n"


def test_make_sheet_columns_and_alts():
    sheet = make_sheet(ITEMS)
    assert sheet.startswith("\ufeff")
    rows = [line.split("\t") for line in sheet.lstrip("\ufeff").splitlines()]
    assert rows[0] == list(COLUMNS)
    assert COLUMNS == ("id", "category", "text", "reading", "reading_alts", "source",
                       "ok", "fixed_reading", "fixed_alts", "comment")
    assert rows[1][:6] == ["number-0001", "number", "文1", "イチ", "ヒトツ|ワン", "Claude draft"]
    assert rows[1][6:] == ["", "", "", ""]
    assert len(rows) == 5


def test_apply_sheet_ok_fixed_rejected_and_blank():
    tsv = _fill(make_sheet(ITEMS), {
        "number-0001": ("y", "", "", ""),
        "number-0002": ("", "ニ", "フタツ|ツー", "reading was wrong"),
        "number-0003": ("n", "", "", "bad sentence"),
        "number-0004": ("", "", "", ""),
    })
    items, problems = apply_sheet(ITEMS, tsv)
    by_id = {i["id"]: i for i in items}
    assert by_id["number-0001"]["verified"] is True
    assert by_id["number-0001"]["reading_alts"] == ["ヒトツ", "ワン"]
    assert by_id["number-0002"] == {**ITEMS[1], "reading": "ニ", "reading_alts": ["フタツ", "ツー"], "verified": True}
    assert "number-0003" not in by_id
    assert "number-0004" not in by_id
    assert any(p.startswith("number-0004") for p in problems)
    assert len(problems) == 1


def test_apply_sheet_ok_is_case_and_space_insensitive_and_reports_unknown_ids():
    tsv = _fill(make_sheet(ITEMS[:1]), {"number-0001": (" Y ", "", "", "")})
    tsv += "number-9999\tnumber\tx\tイチ\t\tx\ty\t\t\t\r\n"
    items, problems = apply_sheet(ITEMS, tsv)
    assert [i["id"] for i in items] == ["number-0001"]
    assert any(p.startswith("number-9999") for p in problems)
    assert any(p.startswith("number-0002") for p in problems)  # in the data but not in the sheet


def test_read_sheet_accepts_utf8_bom_utf16_and_cp932(tmp_path):
    from yomibench.review import read_sheet
    text = "id\tok\r\nnumber-0001\ty\r\n文\tイチ\r\n"
    for enc in ("utf-8-sig", "utf-16", "cp932"):
        p = tmp_path / f"{enc}.tsv"
        p.write_bytes(text.encode(enc))
        assert read_sheet(p) == text, enc


def _set(tsv, rid, column, value):
    """Change one cell of a filled sheet (as an author editing it in Excel would)."""
    col = COLUMNS.index(column)
    lines = tsv.splitlines()
    for k, line in enumerate(lines):
        cells = line.split("\t")
        if cells[0] == rid:
            cells[col] = value
            lines[k] = "\t".join(cells)
    return "\r\n".join(lines) + "\r\n"


ALL_Y = {f"number-{n:04d}": ("y", "", "", "") for n in range(1, 5)}


def test_reading_edited_in_the_sheet_is_a_problem_not_silently_ignored():
    tsv = _set(_fill(make_sheet(ITEMS), ALL_Y), "number-0002", "reading", "ニ")
    items, problems = apply_sheet(ITEMS, tsv)
    assert "number-0002" not in {i["id"] for i in items}
    assert any(p.startswith("number-0002") and "fixed_reading" in p for p in problems)


def test_alts_edited_in_the_sheet_is_a_problem():
    tsv = _set(_fill(make_sheet(ITEMS), ALL_Y), "number-0001", "reading_alts", "ヒトツ")
    _, problems = apply_sheet(ITEMS, tsv)
    assert any(p.startswith("number-0001") for p in problems)


def test_draft_changed_after_the_sheet_was_made_is_a_problem():
    tsv = _fill(make_sheet(ITEMS), ALL_Y)
    changed = [ITEMS[0], _item(2, reading="サン"), ITEMS[2], ITEMS[3]]
    items, problems = apply_sheet(changed, tsv)
    assert "number-0002" not in {i["id"] for i in items}
    assert any(p.startswith("number-0002") for p in problems)


def test_ok_y_with_fixed_alts_adds_the_alternatives():
    tsv = _fill(make_sheet(ITEMS), {**ALL_Y, "number-0001": ("y", "", "イッ|ワン", "")})
    items, problems = apply_sheet(ITEMS, tsv)
    assert problems == []
    assert items[0]["reading"] == "イチ" and items[0]["reading_alts"] == ["ヒトツ", "ワン", "イッ"]


def test_ok_n_with_a_fixed_reading_is_a_problem():
    tsv = _fill(make_sheet(ITEMS), {**ALL_Y, "number-0003": ("n", "サン", "", "")})
    items, problems = apply_sheet(ITEMS, tsv)
    assert "number-0003" not in {i["id"] for i in items}
    assert any(p.startswith("number-0003") for p in problems)


def test_duplicate_rows_are_a_problem():
    tsv = _fill(make_sheet(ITEMS), ALL_Y)
    tsv += tsv.splitlines()[2] + "\r\n"  # number-0002 twice
    items, problems = apply_sheet(ITEMS, tsv)
    assert "number-0002" not in {i["id"] for i in items}
    assert any(p.startswith("number-0002") and "twice" in p for p in problems)


def test_fullwidth_and_halfwidth_kana_in_an_untouched_sheet_still_match():
    item = _item(5, reading="ナンバー", alts=["ナンバ"])
    tsv = _fill(make_sheet([item]), {"number-0005": ("y", "", "", "")})
    tsv = _set(tsv, "number-0005", "reading", "ﾅﾝﾊﾞｰ")  # an editor that converted to half width
    items, problems = apply_sheet([item], tsv)
    assert problems == [] and items[0]["reading"] == "ナンバー"

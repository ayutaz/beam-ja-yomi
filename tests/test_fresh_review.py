import json
import sys

import scripts.fresh_review as fr


def test_apply_records_who_checked_the_readings(tmp_path, monkeypatch):
    draft = {"id": "fresh-dev-0001", "category": "dev", "text": "辛い", "reading": "カライ", "reading_alts": [],
             "pool_id": "wiki-000001", "source": "s", "note": "", "verified": False}
    (tmp_path / "data" / "fresh").mkdir(parents=True)
    (tmp_path / "review").mkdir()
    monkeypatch.setattr(fr, "ROOT", tmp_path)
    monkeypatch.setattr(fr, "DRAFT", tmp_path / "data" / "fresh" / "draft.jsonl")
    fr.DRAFT.write_text(json.dumps(draft, ensure_ascii=False) + "\n", encoding="utf-8")
    sheet = fr.make_sheet_with_notes([draft]).replace("\ts\t\t\t\t", "\ts\ty\t\t\t", 1)
    (tmp_path / "review" / "fresh_dev.tsv").write_text(sheet, encoding="utf-8", newline="")
    monkeypatch.setattr(sys, "argv", ["fresh_review.py", "apply", "--split", "dev", "--checked-by", "Claude"])
    fr.main()
    row = json.loads((tmp_path / "data" / "fresh" / "dev.jsonl").read_text(encoding="utf-8"))
    assert row["verified"] is True and row["checked_by"] == "Claude"

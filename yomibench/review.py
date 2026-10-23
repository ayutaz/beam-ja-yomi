"""The author's review sheet: one TSV row per draft item (docs/02 "正解データの作り方").

The author writes y / n in `ok`, or a corrected reading in `fixed_reading` (and `fixed_alts`).
"""

import csv
import io
import unicodedata
from pathlib import Path

COLUMNS = ("id", "category", "text", "reading", "reading_alts", "source",
           "ok", "fixed_reading", "fixed_alts", "comment")
ALT_SEP = "|"


def make_sheet(items: list[dict]) -> str:
    """TSV with a UTF-8 BOM so that Excel opens it as UTF-8."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\r\n")
    w.writerow(COLUMNS)
    for i in items:
        w.writerow([i["id"], i["category"], i["text"], i["reading"], ALT_SEP.join(i["reading_alts"]),
                    i["source"], "", "", "", ""])
    return "\ufeff" + buf.getvalue()


def read_sheet(path: Path) -> str:
    """Excel may save the sheet as UTF-16 ("Unicode テキスト") or Shift_JIS ("テキスト (タブ区切り)")."""
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp932")


def _alts(cell: str) -> list[str]:
    return [a.strip() for a in cell.split(ALT_SEP) if a.strip()]


def _same(cell: str, value: str) -> bool:
    return unicodedata.normalize("NFKC", (cell or "").strip()) == unicodedata.normalize("NFKC", value)


def apply_sheet(items: list[dict], tsv: str) -> tuple[list[dict], list[str]]:
    """Return (verified items in data order, problems). Rows with a problem are not applied.

    A row's reading and reading_alts cells must still equal the draft's (NFKC), so that an answer is never
    applied to a reading the author did not see: an edited cell, or a draft changed after the sheet was made.
    The text is not compared, because a Shift_JIS save can mangle rare kanji (readings are kana and survive).
    """
    rows, seen, problems = {}, set(), []
    for row in csv.DictReader(io.StringIO(tsv.lstrip("\ufeff")), delimiter="\t"):
        rid = (row.get("id") or "").strip()
        if rid in rows:
            seen.add(rid)
        rows[rid] = row
    problems += [f"{rid}: appears twice in the sheet" for rid in sorted(seen)]
    known = {i["id"] for i in items}
    problems += [f"{rid}: in the sheet but not in the data" for rid in rows if rid and rid not in known]

    out = []
    for item in items:
        row = rows.get(item["id"])
        if row is None:
            problems.append(f"{item['id']}: not in the sheet")
            continue
        if item["id"] in seen:
            continue
        if not (_same(row.get("reading"), item["reading"])
                and _same(row.get("reading_alts"), ALT_SEP.join(item["reading_alts"]))):
            problems.append(f"{item['id']}: the reading or reading_alts cell differs from the draft "
                            "(write a correction in fixed_reading / fixed_alts and leave those cells as they were)")
            continue
        ok = (row.get("ok") or "").strip().lower()
        fixed = (row.get("fixed_reading") or "").strip()
        extra = _alts(row.get("fixed_alts") or "")
        if ok == "n" and (fixed or extra):
            problems.append(f"{item['id']}: ok is n but fixed_reading / fixed_alts are filled (which one?)")
        elif fixed:
            out.append({**item, "reading": fixed, "reading_alts": extra, "verified": True})
        elif ok == "y":
            alts = item["reading_alts"] + [a for a in extra if a not in item["reading_alts"]]
            out.append({**item, "reading_alts": alts, "verified": True})
        elif ok == "n":
            continue
        else:
            problems.append(f"{item['id']}: ok is {ok!r} (write y or n, or a fixed_reading)")
    return out, problems

import json
from pathlib import Path

import pytest

from scripts.publish import files_for


def _touch(root: Path, *paths: str):
    for p in paths:
        f = root / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    _touch(tmp_path, "hf/dataset_card.md", "data/yomi.jsonl", "data/draft/number.jsonl",
           "results/ojt.jsonl", "results/beam_fix-low-en-r0.jsonl", "results/summary.json", "results/report.md",
           "results/ojt.partial.jsonl", "space/app.py", "space/README.md", "space/requirements.txt",
           "space/.beam_cache/a.json", "space/results/old.jsonl", "space/yomibench/old.py", "space/x.wav",
           "yomibench/__init__.py", "yomibench/systems/beam.py", "yomibench/__pycache__/a.pyc",
           ".env", "cache/beam/a.json")
    return tmp_path


def test_dataset_files(repo):
    got = dict((dst, src.relative_to(repo).as_posix()) for src, dst in files_for("dataset", repo))
    assert got == {"README.md": "hf/dataset_card.md", "data/yomi.jsonl": "data/yomi.jsonl",
                   "results/ojt.jsonl": "results/ojt.jsonl",
                   "results/beam_fix-low-en-r0.jsonl": "results/beam_fix-low-en-r0.jsonl",
                   "results/summary.json": "results/summary.json", "results/report.md": "results/report.md"}


def test_space_files_exclude_secrets_cache_audio_and_stale_copies(repo):
    dsts = sorted(dst for _, dst in files_for("space", repo))
    assert dsts == ["README.md", "app.py", "requirements.txt",
                    "results/beam_fix-low-en-r0.jsonl", "results/ojt.jsonl", "results/report.md",
                    "results/summary.json", "yomibench/__init__.py", "yomibench/systems/beam.py"]


def test_missing_dataset_file_is_an_error(tmp_path):
    _touch(tmp_path, "hf/dataset_card.md")
    with pytest.raises(FileNotFoundError):
        files_for("dataset", tmp_path)


def test_results_dir_can_point_elsewhere(repo):
    _touch(repo, "cache/smoke/ojt.jsonl", "cache/smoke/probe.jsonl")
    got = {dst: src.relative_to(repo).as_posix() for src, dst in files_for("space", repo, repo / "cache" / "smoke")}
    assert got["results/ojt.jsonl"] == "cache/smoke/ojt.jsonl"
    assert "results/probe.jsonl" not in got and "results/beam_fix-low-en-r0.jsonl" not in got


class _FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def create_commit(self, repo_id, **kw):
        self.calls.append(("create_commit", repo_id, kw))
        return type("Info", (), {"commit_url": "https://hf.co/x"})()


def test_upload_creates_private_space_when_asked(repo):
    from scripts.publish import upload
    api = _FakeApi()
    upload(api, "space", "me/demo", files_for("space", repo), private=True)
    (_, repo_id, kw), (_, _, commit) = api.calls
    assert repo_id == "me/demo" and kw["private"] is True and kw["repo_type"] == "space" and kw["space_sdk"] == "gradio"
    assert sorted(op.path_in_repo for op in commit["operations"])[0] == "README.md"


def test_upload_never_changes_visibility_unless_asked(repo):
    from scripts.publish import upload
    api = _FakeApi()
    upload(api, "dataset", "me/data", files_for("dataset", repo), private=False)
    assert api.calls[0][2]["private"] is None  # None keeps an existing repo's visibility


def _real_release(root, reading_in_results="イチ", verified=True):
    item = {"id": "number-0001", "category": "number", "text": "文1", "reading": "イチ", "reading_alts": [],
            "source": "s", "note": "", "verified": verified}
    row = {**{k: item[k] for k in ("id", "category", "text", "reading_alts")}, "reading": reading_in_results,
           "output": "イチ", "match": True, "cer": 0.0, "format_issue": False}
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "yomi.jsonl").write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "ojt.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_check_release_passes_for_matching_verified_data(tmp_path):
    from scripts.publish import check_release
    _real_release(tmp_path)
    assert check_release(tmp_path) == []


def test_check_release_blocks_stale_results_and_unverified_data(tmp_path):
    from scripts.publish import check_release
    _real_release(tmp_path, reading_in_results="ニ")
    assert check_release(tmp_path)
    _real_release(tmp_path, verified=False)
    assert check_release(tmp_path)


def test_unfilled_placeholders_are_found_but_field_patterns_are_not():
    from scripts.publish import unfilled_placeholders
    text = "- Code: {GitHub URL}\n{N} sentences, {n} each\n{results table}\nid is `{category}-{number}`, `{city}{town}`"
    assert unfilled_placeholders(text) == ["{GitHub URL}", "{N}", "{n}", "{results table}"]


def test_the_dataset_card_has_no_placeholder_left():
    from scripts.publish import unfilled_placeholders
    path = Path(__file__).resolve().parent.parent / "hf" / "dataset_card.md"
    if not path.is_file():
        pytest.skip("the dataset card is not part of the public code repository")
    card = path.read_text(encoding="utf-8")
    assert unfilled_placeholders(card) == []


def test_space_requirements_pin_the_reading_stack_as_locked():
    import tomllib
    lock = {p["name"]: p["version"] for p in tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))["package"]}
    reqs = Path("space/requirements.txt").read_text(encoding="utf-8").split()
    for name in ("sudachidict-core", "sudachipy"):
        assert f"{name}=={lock[name]}" in reqs, name


def test_space_is_set_up_for_zerogpu():
    readme = Path("space/README.md").read_text(encoding="utf-8")
    assert 'python_version: "3.12"' in readme  # ZeroGPU runs Python 3.12.12 or 3.10.13 (checked 2026-10-23)
    assert "spaces" in Path("space/requirements.txt").read_text(encoding="utf-8").split()


def test_upload_asks_for_zerogpu_hardware_for_a_new_space(repo):
    from scripts.publish import upload
    api = _FakeApi()
    upload(api, "space", "me/demo", files_for("space", repo), private=True)
    assert api.calls[0][2]["space_hardware"] == "zero-a10g"

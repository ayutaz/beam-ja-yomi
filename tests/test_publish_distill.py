import hashlib
import json
from pathlib import Path

import pytest

from scripts.publish_distill import files_for, upload


def _touch(root: Path, *paths: str):
    for p in paths:
        f = root / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    _touch(tmp_path, "models/s440-e2/config.json", "models/s440-e2/model.safetensors",
           "models/s440-e2/tokenizer.json", "models/s440-e2/train_config.json", "models/other/model.safetensors",
           "hf/distill_model_card.md", "hf/distill_dataset_card.md", "hf/figures/distill.png",
           "data/distill/teacher.jsonl", "data/distill/pool_stats.json", "data/distill/pool.jsonl",
           "data/fresh/dev.jsonl", "data/fresh/test.jsonl", "data/fresh/draft.jsonl",
           "data/fresh/dev_blind.jsonl", "data/fresh/test_blind.jsonl",
           "docs/prereg_yomi_distill.md", "results/distill/report.md",
           "results/distill/test/student.jsonl", "results/distill/test/beam_fix.jsonl",
           "results/distill/dev/s440-e1.jsonl", "results/distill/dev_summary.md",
           "results/distill/eval300/student.jsonl",
           "space_distill/app.py", "space_distill/README.md", "space_distill/requirements.txt",
           "yomibench/__init__.py", "yomibench/student.py", "yomibench/__pycache__/a.pyc", ".env", "cache/teacher/a.json")
    sha = {n: hashlib.sha256(b"x").hexdigest() for n in ("config.json", "model.safetensors", "tokenizer.json", "train_config.json")}
    frozen = {"model_dir": "models/s440-e2", "sha256": sha, "min_ratio": 0.8, "max_ratio": 1.25,
              "max_new_tokens": 160}
    (tmp_path / "results/distill/frozen.json").write_text(json.dumps(frozen), encoding="utf-8")
    return tmp_path


def _dsts(target, root):
    return {dst: src.relative_to(root).as_posix() for src, dst in files_for(target, root)}


def test_model_files_are_the_frozen_model_and_its_card(repo):
    assert _dsts("model", repo) == {
        "README.md": "hf/distill_model_card.md", "config.json": "models/s440-e2/config.json",
        "model.safetensors": "models/s440-e2/model.safetensors", "tokenizer.json": "models/s440-e2/tokenizer.json",
        "train_config.json": "models/s440-e2/train_config.json", "frozen.json": "results/distill/frozen.json"}


def test_model_upload_stops_when_a_file_changed_after_freezing(repo):
    (repo / "models/s440-e2/model.safetensors").write_text("retrained", encoding="utf-8")
    with pytest.raises(ValueError, match="model.safetensors"):
        files_for("model", repo)


def test_data_files_hold_teacher_rows_fresh_sets_prereg_and_results(repo):
    assert _dsts("data", repo) == {
        "README.md": "hf/distill_dataset_card.md", "data/teacher.jsonl": "data/distill/teacher.jsonl",
        "data/fresh_dev.jsonl": "data/fresh/dev.jsonl", "data/fresh_test.jsonl": "data/fresh/test.jsonl",
        "data/fresh_dev_blind.jsonl": "data/fresh/dev_blind.jsonl",
        "data/fresh_test_blind.jsonl": "data/fresh/test_blind.jsonl",
        "prereg.md": "docs/prereg_yomi_distill.md", "leak_filter_stats.json": "data/distill/pool_stats.json",
        "results/frozen.json": "results/distill/frozen.json", "results/report.md": "results/distill/report.md",
        "results/test/beam_fix.jsonl": "results/distill/test/beam_fix.jsonl",
        "results/test/student.jsonl": "results/distill/test/student.jsonl",
        "results/eval300/student.jsonl": "results/distill/eval300/student.jsonl",
        "results/dev_summary.md": "results/distill/dev_summary.md",
        "results/dev/s440-e1.jsonl": "results/distill/dev/s440-e1.jsonl",
        "figures/distill.png": "hf/figures/distill.png"}


def test_space_files_exclude_secrets_caches_and_bytecode(repo):
    assert _dsts("space", repo) == {
        "app.py": "space_distill/app.py", "README.md": "space_distill/README.md",
        "requirements.txt": "space_distill/requirements.txt", "frozen.json": "results/distill/frozen.json",
        "report.md": "results/distill/report.md", "yomibench/__init__.py": "yomibench/__init__.py",
        "yomibench/student.py": "yomibench/student.py"}


def test_missing_file_is_an_error(repo):
    (repo / "data/fresh/test.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        files_for("data", repo)


class _FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def create_commit(self, repo_id, **kw):
        self.calls.append(("create_commit", repo_id, kw))
        return type("Info", (), {"commit_url": "https://hf.co/x"})()

    def update_repo_settings(self, repo_id, **kw):
        self.calls.append(("update_repo_settings", repo_id, kw))


def test_upload_uses_the_right_repo_type_and_zero_gpu_for_the_space(repo):
    for target, repo_type in (("model", "model"), ("data", "dataset"), ("space", "space")):
        api = _FakeApi()
        upload(api, target, "me/x", files_for(target, repo), private=True)
        kw = api.calls[0][2]
        assert kw["repo_type"] == repo_type and kw["private"] is True
        assert (kw.get("space_hardware") == "zero-a10g") == (target == "space")


def test_upload_keeps_visibility_unless_asked(repo):
    api = _FakeApi()
    upload(api, "model", "me/x", files_for("model", repo), private=False)
    assert api.calls[0][2]["private"] is None


def test_check_release_blocks_unchecked_fresh_sets_and_placeholders(repo):
    from scripts.publish_distill import check_release

    for split in ("dev", "test"):
        (repo / f"data/fresh/{split}.jsonl").write_text("", encoding="utf-8")
    (repo / "hf/distill_model_card.md").write_text("Exact match {N}%", encoding="utf-8")
    problems = check_release(repo)
    assert any("distill_model_card" in p for p in problems)
    assert any(p.startswith("dev:") for p in problems) and any(p.startswith("test:") for p in problems)


def test_model_upload_stops_when_a_file_appeared_after_freezing(repo):
    (repo / "models/s440-e2/extra.bin").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="extra.bin"):
        files_for("model", repo)


def test_upload_makes_the_repo_public_after_the_commit_only_when_asked(repo):
    api = _FakeApi()
    upload(api, "data", "me/x", files_for("data", repo), make_public=True)
    assert [c[0] for c in api.calls] == ["create_repo", "create_commit", "update_repo_settings"]
    assert api.calls[2][2] == {"private": False, "repo_type": "dataset"}
    api = _FakeApi()
    upload(api, "data", "me/x", files_for("data", repo))
    assert "update_repo_settings" not in [c[0] for c in api.calls]


def test_a_private_upload_is_checked_too(repo, monkeypatch):
    import scripts.publish_distill as pd

    for split in ("dev", "test"):
        (repo / f"data/fresh/{split}.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(pd, "ROOT", repo)
    monkeypatch.setattr("sys.argv", ["publish_distill.py", "--target", "data", "--repo", "me/x", "--private"])
    with pytest.raises(SystemExit, match="not ready"):
        pd.main()

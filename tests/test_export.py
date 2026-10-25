from pathlib import Path

from scripts.export_public import public_files


def _touch(root: Path, *paths: str):
    for p in paths:
        f = root / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")


def test_public_files_keep_code_data_results_and_drop_private_notes(tmp_path):
    _touch(tmp_path, "yomibench/__init__.py", "yomibench/systems/beam.py", "scripts/run_eval.py", "scripts/run_all.sh",
           "scripts/make_smoke_set.py", "tests/test_beam.py", "tests/fixtures/ken_all_sample.csv",
           "space/app.py", "space/README.md", "space/requirements.txt", "space/.beam_cache/a.json",
           "data/yomi.jsonl", "data/draft/number.jsonl", "results/ojt.jsonl", "results/ojt.partial.jsonl",
           "hf/README_github.md", "hf/dataset_card.md", "pyproject.toml", "uv.lock", ".gitignore",
           ".gitattributes", ".python-version", "LICENSE",
           "README.md", "CLAUDE.md", "docs/00_概要.md", "review/review.tsv", "probe/a.jsonl", ".env",
           "cache/beam/a.json", ".superpowers/sdd/x.md", "yomibench/__pycache__/a.pyc")
    got = {dst: src.relative_to(tmp_path).as_posix() for src, dst in public_files(tmp_path)}
    assert got["README.md"] == "hf/README_github.md"
    assert got["data/yomi.jsonl"] == "data/yomi.jsonl" and got["results/ojt.jsonl"] == "results/ojt.jsonl"
    assert {"yomibench/systems/beam.py", "scripts/run_all.sh", "tests/fixtures/ken_all_sample.csv",
            "space/app.py", "LICENSE", "uv.lock"} <= set(got)
    for private in ("CLAUDE.md", "docs/00_概要.md", "review/review.tsv", "probe/a.jsonl", ".env",
                    "cache/beam/a.json", ".superpowers/sdd/x.md", "data/draft/number.jsonl",
                    "results/ojt.partial.jsonl", "space/.beam_cache/a.json", "scripts/make_smoke_set.py",
                    "yomibench/__pycache__/a.pyc", "hf/dataset_card.md"):
        assert private not in got and private not in got.values(), private


def test_public_files_include_the_distill_space_but_not_models_or_teacher_data(tmp_path):
    _touch(tmp_path, "hf/README_github.md", "data/yomi.jsonl", "space/app.py", "space/README.md",
           "space/requirements.txt", "space_distill/app.py", "space_distill/README.md",
           "space_distill/requirements.txt", "space_distill/__pycache__/app.pyc", "scripts/train_student.py",
           "models/s1.8b-lora-e2/model.safetensors", "data/distill/teacher.jsonl", "data/fresh/draft.jsonl",
           "results/distill/report.md", "hf/figures/results.png")
    got = {dst: src.relative_to(tmp_path).as_posix() for src, dst in public_files(tmp_path)}
    assert {"hf/figures/results.png", "space_distill/app.py", "space_distill/README.md", "space_distill/requirements.txt",
            "scripts/train_student.py"} <= set(got)
    for private in ("space_distill/__pycache__/app.pyc", "models/s1.8b-lora-e2/model.safetensors",
                    "data/distill/teacher.jsonl", "data/fresh/draft.jsonl", "results/distill/report.md"):
        assert private not in got, private

"""Publish Yomi-Distill to Hugging Face: the student model, the teaching data, or the Space. Without --yes it
only lists the files.

  uv run scripts/publish_distill.py --target model --repo ayousanz/yomi-distill-ja                  # list only
  uv run scripts/publish_distill.py --target data --repo ayousanz/beam-yomi-teacher --private --yes   # private
  uv run scripts/publish_distill.py --target space --repo ayousanz/yomi-distill --private --yes

The model is the one named in results/distill/frozen.json, and every file must still match the sha256 written
there. Needs a write-enabled HF_TOKEN in .env (only with --yes). Exactly the listed files are uploaded.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.publish import unfilled_placeholders  # noqa: E402
from yomibench.distill_eval import file_hashes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO_TYPE = {"model": "model", "data": "dataset", "space": "space"}
CARD = {"model": "hf/distill_model_card.md", "data": "hf/distill_dataset_card.md", "space": "space_distill/README.md"}
SPACE_FILES = ("app.py", "README.md", "requirements.txt")
# per split: 100 drawn, minus the rows dropped as unusable (docs/prereg_yomi_distill.md)
FRESH_SIZES = range(90, 101)


def frozen_model(root: Path) -> tuple[Path, dict]:
    """(model dir, frozen.json). Raises ValueError if a file changed after freezing."""
    frozen = json.loads((root / "results" / "distill" / "frozen.json").read_text(encoding="utf-8"))
    model_dir = root / frozen["model_dir"]
    now = file_hashes(model_dir)
    changed = sorted(name for name in set(now) | set(frozen["sha256"]) if now.get(name) != frozen["sha256"].get(name))
    if changed:
        raise ValueError(f"changed after freezing: {changed}")
    return model_dir, frozen


def files_for(target: str, root: Path = ROOT) -> list[tuple[Path, str]]:
    """(local file, path in the repo). Never includes .env, cache/, the pool or drafts."""
    root = Path(root)
    distill = root / "results" / "distill"
    if target == "model":
        model_dir, _ = frozen_model(root)
        out = [(root / CARD["model"], "README.md")]
        out += [(p, p.name) for p in sorted(model_dir.iterdir()) if p.is_file()]
        out += [(distill / "frozen.json", "frozen.json")]
    elif target == "data":
        out = [(root / CARD["data"], "README.md"),
               (root / "data" / "distill" / "teacher.jsonl", "data/teacher.jsonl"),
               (root / "data" / "fresh" / "dev.jsonl", "data/fresh_dev.jsonl"),
               (root / "data" / "fresh" / "test.jsonl", "data/fresh_test.jsonl"),
               (root / "data" / "fresh" / "dev_blind.jsonl", "data/fresh_dev_blind.jsonl"),
               (root / "data" / "fresh" / "test_blind.jsonl", "data/fresh_test_blind.jsonl"),
               (root / "docs" / "prereg_yomi_distill.md", "prereg.md"),
               (root / "data" / "distill" / "pool_stats.json", "leak_filter_stats.json"),
               (distill / "frozen.json", "results/frozen.json"),
               (distill / "report.md", "results/report.md"),
               (distill / "dev_summary.md", "results/dev_summary.md")]
        for split in ("test", "eval300", "dev"):
            out += [(p, f"results/{split}/{p.name}") for p in sorted((distill / split).glob("*.jsonl"))]
        figure = root / "hf" / "figures" / "distill.png"
        if figure.is_file():
            out.append((figure, "figures/distill.png"))
    elif target == "space":
        out = [(root / "space_distill" / name, name) for name in SPACE_FILES]
        out += [(distill / "frozen.json", "frozen.json"), (distill / "report.md", "report.md")]
        out += [(p, p.relative_to(root).as_posix()) for p in sorted((root / "yomibench").rglob("*.py"))]
    else:
        raise ValueError(f"unknown target {target!r}")
    missing = [str(p) for p, _ in out if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing: {missing}")
    return out


def check_release(root: Path = ROOT) -> list[str]:
    """Before a public upload: the fresh sets are human-checked, and no card has a placeholder left."""
    from yomibench.data import load_items
    from yomibench.fresh import validate_fresh

    root = Path(root)
    problems = []
    for split in ("dev", "test"):
        items = load_items(root / "data" / "fresh" / f"{split}.jsonl")
        if len(items) not in FRESH_SIZES:
            problems.append(f"{split}: {len(items)} sentences, expected 90-100 (100 drawn, a few dropped)")
        problems += [f"{split}: {p}" for p in validate_fresh(items)]
    for card in CARD.values():
        left = unfilled_placeholders((root / card).read_text(encoding="utf-8"))
        if left:
            problems.append(f"{card} still has placeholders: {left}")
    return problems


def upload(api, target: str, repo: str, files: list[tuple[Path, str]], private: bool = False) -> str:
    """Create the repo if needed and upload exactly `files` in one commit. Returns the commit URL."""
    from huggingface_hub import CommitOperationAdd

    repo_type = REPO_TYPE[target]
    # A free account can host Gradio Spaces only on ZeroGPU (HF docs, checked 2026-10-25); the student needs the GPU.
    space = {"space_sdk": "gradio", "space_hardware": "zero-a10g"} if repo_type == "space" else {}
    api.create_repo(repo, repo_type=repo_type, exist_ok=True, private=True if private else None, **space)
    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)) for src, dst in files]
    info = api.create_commit(repo, operations=ops, repo_type=repo_type, commit_message="Upload from beam-ja-yomi-bench")
    return info.commit_url


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(REPO_TYPE), required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--yes", action="store_true", help="really upload (otherwise list only)")
    ap.add_argument("--private", action="store_true", help="create the repo as private (a new repo only)")
    args = ap.parse_args()

    if not args.private:
        problems = check_release(ROOT)
        if problems:
            sys.exit("not ready for a public upload:\n" + "\n".join(problems[:20]))
    files = files_for(args.target, ROOT)
    for src, dst in files:
        print(f"{dst}  <-  {src.relative_to(ROOT).as_posix()}  ({src.stat().st_size} bytes)")
    print(f"{len(files)} files for {args.target} {args.repo}" + (" (private)" if args.private else ""))
    if not args.yes:
        print("dry run: nothing uploaded (pass --yes to upload)")
        return

    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set in .env")
    print(f"uploaded: {upload(HfApi(token=token), args.target, args.repo, files, args.private)}")


if __name__ == "__main__":
    main()

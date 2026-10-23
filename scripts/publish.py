"""Publish the dataset or the Space to Hugging Face. Without --yes it only lists the files.

  uv run scripts/publish.py --target dataset --repo ayousanz/beam-ja-yomi-bench          # list only
  uv run scripts/publish.py --target space --repo ayousanz/beam-ja-yomi-fixer --yes       # upload
  uv run scripts/publish.py --target space --repo ayousanz/beam-ja-yomi-fixer --private --results-dir cache/smoke --yes
                                                                                          # private trial build

Needs a write-enabled HF_TOKEN in .env (only with --yes). Exactly the listed files are uploaded.
"""

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_FILE = re.compile(r"^(ojt|ojt_vanilla|beam_(?:direct|fix)-[a-z]+-[a-z]+-r\d+)\.jsonl$|^summary\.json$|^report\.md$")
SPACE_FILES = ("app.py", "README.md", "requirements.txt")
# {GitHub URL}, {N}, {n}, {results table} in a draft; not field patterns such as {category}-{number}
PLACEHOLDER = re.compile(r"\{(?:[A-Z][^{}\n]*|n|results table)\}")


def _results(results_dir: Path) -> list[tuple[Path, str]]:
    files = sorted(p for p in results_dir.glob("*") if RESULT_FILE.match(p.name))
    if not files:
        raise FileNotFoundError(f"no result files in {results_dir}")
    return [(p, f"results/{p.name}") for p in files]


def files_for(target: str, root: Path = ROOT, results_dir: Path | None = None) -> list[tuple[Path, str]]:
    """(local file, path in the repo). Never includes .env, cache/, drafts, audio or partial runs.

    results_dir (default root/results) is only for a private trial build with the smoke results.
    """
    root = Path(root)
    results_dir = Path(results_dir) if results_dir else root / "results"
    if target == "dataset":
        out = [(root / "hf" / "dataset_card.md", "README.md"), (root / "data" / "yomi.jsonl", "data/yomi.jsonl")]
        out += _results(results_dir)
    elif target == "space":
        out = [(root / "space" / name, name) for name in SPACE_FILES]
        out += [(p, p.relative_to(root).as_posix()) for p in sorted((root / "yomibench").rglob("*.py"))]
        out += _results(results_dir)
    else:
        raise ValueError(f"unknown target {target!r}")
    missing = [str(p) for p, _ in out if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing: {missing}")
    return out


def check_release(root: Path = ROOT) -> list[str]:
    """Before a real publish: data/yomi.jsonl is fully verified and results/ was made from exactly that file."""
    from yomibench.data import load_items, validate_items
    from yomibench.summary import check_results

    root = Path(root)
    items = load_items(root / "data" / "yomi.jsonl")
    return validate_items(items, require_verified=True) + check_results(root / "results", items)


def unfilled_placeholders(text: str) -> list[str]:
    return PLACEHOLDER.findall(text)


def upload(api, target: str, repo: str, files: list[tuple[Path, str]], private: bool = False) -> str:
    """Create the repo if needed and upload exactly `files` in one commit. Returns the commit URL."""
    from huggingface_hub import CommitOperationAdd

    repo_type = "dataset" if target == "dataset" else "space"
    # A free account can host Gradio Spaces only on ZeroGPU (HF docs, checked 2026-10-23); the app itself runs on CPU.
    space = {"space_sdk": "gradio", "space_hardware": "zero-a10g"} if repo_type == "space" else {}
    api.create_repo(repo, repo_type=repo_type, exist_ok=True, private=True if private else None, **space)
    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)) for src, dst in files]
    info = api.create_commit(repo, operations=ops, repo_type=repo_type, commit_message="Upload from beam-ja-yomi-bench")
    return info.commit_url


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("dataset", "space"), required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--yes", action="store_true", help="really upload (otherwise list only)")
    ap.add_argument("--private", action="store_true", help="create the repo as private (a new repo only)")
    ap.add_argument("--results-dir", type=Path, help="default: results/ (cache/smoke only for a private trial)")
    args = ap.parse_args()

    if args.results_dir is None:
        problems = check_release(ROOT)
        if problems:
            sys.exit("results/ and data/yomi.jsonl do not belong together:\n" + "\n".join(problems[:20]))
    elif not args.private:
        sys.exit("--results-dir is only for a --private trial; a public upload uses results/")
    if args.target == "dataset" and not args.private:
        left = unfilled_placeholders((ROOT / "hf" / "dataset_card.md").read_text(encoding="utf-8"))
        if left:
            sys.exit(f"hf/dataset_card.md still has placeholders: {left}")
    files = files_for(args.target, ROOT, args.results_dir.resolve() if args.results_dir else None)
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

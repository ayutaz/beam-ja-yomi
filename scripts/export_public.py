"""Copy the files meant for a public code repository into a separate folder (a fresh history starts there).

  uv run scripts/export_public.py OUT_DIR             # copy, then run the tests inside OUT_DIR

Only an allow-list is copied: code, tests, the verified dataset, results, both Spaces, the English README and the
license. Part 2's model, teaching data and results are on Hugging Face, not here.
Working notes (docs/, CLAUDE.md, review/, probe/, drafts, cache/) never leave this repository.
Nothing is pushed anywhere.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP_FILES = ("pyproject.toml", "uv.lock", ".gitignore", ".gitattributes", ".python-version", "LICENSE")
SPACE_FILES = ("app.py", "README.md", "requirements.txt")
RESULT_FILE = re.compile(r"^(ojt|ojt_vanilla|beam_(?:direct|fix)-[a-z]+-[a-z]+-r\d+)\.jsonl$|^summary\.json$|^report\.md$")
PRIVATE_SCRIPTS = {"make_smoke_set.py"}  # builds the smoke set from probe/, which stays private


def public_files(root: Path = ROOT) -> list[tuple[Path, str]]:
    """(local file, path in the public repository)."""
    root = Path(root)
    out = [(root / "hf" / "README_github.md", "README.md")]
    out += [(root / name, name) for name in TOP_FILES if (root / name).is_file()]
    for top in ("yomibench", "tests"):
        out += [(p, p.relative_to(root).as_posix()) for p in sorted((root / top).rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts]
    out += [(p, f"scripts/{p.name}") for p in sorted((root / "scripts").glob("*"))
            if p.suffix in (".py", ".sh") and p.name not in PRIVATE_SCRIPTS]
    out += [(root / "space" / name, f"space/{name}") for name in SPACE_FILES]
    out += [(root / "space_distill" / name, f"space_distill/{name}") for name in SPACE_FILES]  # Part 2
    out += [(root / "data" / "yomi.jsonl", "data/yomi.jsonl")]
    out += [(root / "hf" / "figures" / "results.png", "hf/figures/results.png")]  # README and space/app.py use it
    out += [(p, f"results/{p.name}") for p in sorted((root / "results").glob("*")) if RESULT_FILE.match(p.name)]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty")
    if out == ROOT or ROOT in out.parents:
        sys.exit("OUT_DIR must be outside this repository")
    files = public_files()
    missing = [str(src) for src, _ in files if not src.is_file()]
    if missing:
        sys.exit(f"missing: {missing}")
    for src, dst in files:
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / dst)
    print(f"{len(files)} files -> {out}")
    sys.exit(subprocess.run(["uv", "run", "--group", "space", "pytest", "-q"], cwd=out).returncode)


if __name__ == "__main__":
    main()

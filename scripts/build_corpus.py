"""Build the sentence pool for the teacher from Japanese Wikipedia (plan 2026-10-24, Task 2).

Source: Hugging Face `wikimedia/wikipedia`, config `20231101.ja` (CC BY-SA 3.0 / GFDL, checked 2026-10-24).
Two of its 15 parquet shards are chosen with the seed, downloaded to cache/wiki/ (not in git), and turned
into data/distill/pool.jsonl. Sentences close to the 300 evaluation sentences (data/yomi.jsonl) are removed.

    uv run scripts/build_corpus.py --n 30000
"""

import argparse
import datetime
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yomibench import ROOT  # noqa: E402
from yomibench.corpus import LeakFilter, build_pool  # noqa: E402
from yomibench.data import DATA_PATH, load_items  # noqa: E402

REPO, CONFIG, N_SHARDS = "wikimedia/wikipedia", "20231101.ja", 15
OUT_DIR = ROOT / "data" / "distill"


def shard_names(k: int, seed: int) -> list[str]:
    picks = sorted(random.Random(seed).sample(range(N_SHARDS), k))
    return [f"{CONFIG}/train-{i:05d}-of-{N_SHARDS:05d}.parquet" for i in picks]


def iter_articles(paths: list[Path]):
    for path in paths:
        table = pq.read_table(path, columns=["id", "url", "title", "text"])
        yield from table.to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    names = shard_names(args.shards, args.seed)
    paths = [Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=ROOT / "cache" / "wiki"))
             for name in names]
    leak = LeakFilter([i["text"] for i in load_items(DATA_PATH)])
    stats = {}
    pool = build_pool(iter_articles(paths), leak, n=args.n, seed=args.seed, stats=stats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "pool.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in pool:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats.update(source=f"{REPO} {CONFIG}", license="CC BY-SA 3.0 / GFDL", shards=names, seed=args.seed,
                 pool=len(pool), built=datetime.date.today().isoformat())
    (OUT_DIR / "pool_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n",
                                             encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

import subprocess
import sys


def _run(*args):
    return subprocess.run([sys.executable, "scripts/run_eval.py", *args], capture_output=True, text=True,
                          encoding="utf-8")


def test_drafts_or_limit_cannot_write_into_the_real_results_dir():
    for extra in (["--allow-unverified"], ["--limit", "5"]):
        r = _run("--system", "ojt", *extra)
        assert r.returncode == 2 and "--results-dir" in r.stderr, (extra, r.stderr)

#!/bin/sh
# Phase 3: every condition into one results directory, then the summary.
#   sh scripts/run_all.sh results --offline          # the real run, from the cache only (no API calls)
#   sh scripts/run_all.sh DIR --offline --data cache/prelim/drafts.jsonl --allow-unverified   # cache check on drafts
# The first argument is the results directory; the rest go to every run_eval.py call.
set -e
D=$1
shift
E="uv run scripts/run_eval.py --results-dir $D $*"
$E --system ojt
$E --system ojt_vanilla
$E --system beam_direct --effort low --runs 3
$E --system beam_fix --effort low --runs 3
$E --system beam_direct --effort low --lang ja
for e in medium high; do
  $E --system beam_direct --effort $e
  $E --system beam_fix --effort $e
done
uv run scripts/summarize.py --results-dir "$D"
echo ALL DONE

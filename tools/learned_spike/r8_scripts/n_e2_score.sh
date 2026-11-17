#!/bin/bash
# Round 8 arm N: E2's own networks on the harness (reproduction, four seeds; and seeds 0-1, the comparison baseline).
set -u
cd "$(dirname "$0")/../../.."
source tools/learned_spike/r8_scripts/n_env.sh
nice -n 10 $TPY -m tools.learned_spike.r8_scripts.n_score --root E2=$NOUT/sets/E2 --seeds 0 1 2 3 --out $NOUT/score/E2_s0123 > $NOUT/logs/score_E2_s0123.log 2>&1
nice -n 10 $TPY -m tools.learned_spike.r8_scripts.n_score --root E2=$NOUT/sets/E2 --seeds 0 1 --out $NOUT/score/s01 > $NOUT/logs/score_E2_s01.log 2>&1

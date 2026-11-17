#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# Round 6 arm A2 inputs at stage50-v2s: J keeping every still-ball run (dfl_input c_train_j100).
#   cache_c_train2sj100  DFL raw+degraded from c_train_j100, the rest linked from cache_c_train2s
#   aug2sj100            the DFL round 4 copies (augment r3) rebuilt from c_train_j100, the 23 others linked from aug2s
# Markers via r5_inputs mark. Two processes at nice 15. Ends with $R5W/A2_INPUTS_BUILT. Resumable.
set -uo pipefail
SP=${SCRATCH:?}
source $SP/r3_out/scripts/env.sh
R5W=$SP/r5_work; L=$R5W/logs; EC=$SP/r4_work/engine_data_raw; PY=$HOME/.venvs/barca/bin/python
: "${CODE:?CODE=checkout with c_train_j100}"
export FEATURE_VERSION=stage50-v2s
CJ=$R5W/cache_c_train2sj100; AJ=$R5W/aug2sj100
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $CODE || exit 1
START=$(( $(cat $L/r6_build_inputs_a2.log 2>/dev/null | wc -l) + 1 ))
say() { echo "$(date '+%F %T') $*" >> $L/r6_build_inputs_a2.log; }
say "start, CODE $CODE commit $(git rev-parse HEAD), dirty $(git status --porcelain | wc -l | tr -d ' ')"
R="nice -n 15 $PY -m tools.learned_spike.r5_inputs"
t0=$(date +%s)
( $R cache --kind c_train_j100 --base $R5W/cache_c_train2s --engine-cache $EC --out $CJ > $L/cache_c_train2sj100.log 2>&1; say "cache exit $?"
  $R mark --kind c_train_j100 --dir $CJ >> $L/cache_c_train2sj100.log 2>&1; say "cache mark exit $?"
  $R aug --aug-version r3 --dfl-input c_train_j100 --link-from $R5W/aug2s --labels $CA --engine-cache $EC --out $AJ --games DFL-J03WMX DFL-J03WN1 > $L/aug2sj100_1.log 2>&1; say "aug part 1 exit $?" ) &
( $R aug --aug-version r3 --dfl-input c_train_j100 --link-from $R5W/aug2s --labels $CA --engine-cache $EC --out $AJ --games DFL-J03WOH DFL-J03WOY DFL-J03WPY > $L/aug2sj100_2.log 2>&1; say "aug part 2 exit $?" ) &
wait; say "builders done in $(( $(date +%s) - t0 )) s"
$R mark --kind aug --dir $AJ >> $L/aug2sj100_mark.log 2>&1; say "aug mark exit $?"
sed -n "$START,\$p" $L/r6_build_inputs_a2.log | grep -q "exit [1-9]" && { say "a step FAILED"; exit 1; }
for f in $CJ $AJ; do [ -f $f/dfl_still.json ] && [ -f $f/feature_version.json ] || { say "$f not marked"; exit 1; }; done
touch $R5W/A2_INPUTS_BUILT; say "done"

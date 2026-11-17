#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# Round 4 aug copies with the ball component (augment.py 8), worktree r4smooth. New folder only.
set -uo pipefail
SP=${SCRATCH:?}
source $SP/r3_out/scripts/env.sh
W=$SP/r4smooth_work; L=$W/logs; OUT=$W/aug_c_train_ball; EC=$SP/r4_work/engine_data_raw
PY=$HOME/.venvs/barca/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $SP/r4smooth || exit 1
[ -e $OUT ] && { echo "$OUT exists" >> $L/build.log; exit 1; }
say() { echo "$(date '+%F %T') $*" >> $L/build.log; }
say "start, commit $(git rev-parse HEAD), dirty $(git status --porcelain | wc -l | tr -d ' ')"
R="nice -n 5 $PY -m tools.learned_spike.r4_inputs aug --ball --base $SP/r3aug_cache --labels $CA --engine-cache $EC --out $OUT"
i=0
for G in "METRICA-S1 DFL-J03WMX" "METRICA-S2 DFL-J03WN1" "METRICA-S3 DFL-J03WOH" "DFL-J03WOY DFL-J03WPY"; do
  i=$((i+1)); ( $R --games $G > $L/aug_$i.log 2>&1; say "part $i ($G) exit $?" ) &
  sleep 2
done
wait
nice -n 5 $PY -m tools.learned_spike.r4_inputs mark --kind aug --ball --dir $OUT > $L/mark.log 2>&1; say "mark exit $?"
say "done"; touch $W/AUG_BALL_DONE

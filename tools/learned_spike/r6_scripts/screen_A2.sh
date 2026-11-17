#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# Round 6 arm A2 (note 6.2): recipe AQCSL = AQCSJ with every DFL still-ball run kept in view (input c_train_j100).
# Waits for arm A's chain ($SP/r6_train/FULL_R6_DONE, exit 0) and $SP/r5_work/A2_INPUTS_BUILT, then from CODE:
#   1. Metrica-left-out and SkillCorner-left-out (under ABCSL, linked) seeds 0-3, 4 at a time, into $SP/r6_work/screen;
#      DFL-left-out seeds 0-3 linked from AQCS (never trained).
#   2. r5_measures run and r4_screen score --r5 --tests T-M T-D T-S T-Sp for AQCSL, as measure_A.sh did for AQCSJ.
# Ends with $SP/r6_work/A2_SCREEN_DONE. Env: CODE (required). Log $SP/r6_work/logs/screen_A2.log. Resumable.
set -uo pipefail
SP=${SCRATCH:?}
: "${CODE:?CODE=checkout with letter L}"
LOG=$SP/r6_work/logs/screen_A2.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
say "start, CODE $CODE"
until [ -f $SP/r6_train/FULL_R6_DONE ] && [ -f $SP/r5_work/A2_INPUTS_BUILT ]; do sleep 60; done
[ "$(cat $SP/r6_train/FULL_R6_DONE)" = 0 ] || { say "arm A's chain ended with $(cat $SP/r6_train/FULL_R6_DONE); stopping"; exit 1; }
source $SP/r5_work/scripts/r6_env.sh
export OUTROOT=$SP/r6_work/screen JOBS=4 THREADS=2
cd $CODE || exit 1
say "commit $(git rev-parse HEAD), dirty $(git status --porcelain | wc -l | tr -d ' ')"
ONLY_PROVIDERS="metrica skillcorner" bash tools/learned_spike/r5_screen_launch.sh AQCSL 0 1 2 3 >> $LOG 2>&1 \
  || { say "training FAILED (see $OUTROOT/AQCSL/logs/launch.log)"; exit 1; }
say "metrica and skillcorner networks done"
ONLY_PROVIDERS=dfl bash tools/learned_spike/r5_screen_launch.sh AQCSL 0 1 2 3 >> $LOG 2>&1 \
  || { say "DFL-left-out link FAILED (AQCS dfl seeds 0-3 missing?)"; exit 1; }
say "dfl linked from AQCS"
nice -n 5 $TPY -m tools.learned_spike.r5_measures run --recipe-dir $OUTROOT/AQCSL --seeds 0 1 2 3 --sc-seeds 0 1 2 3 $SCORE_INPUTS2S --defect-dir $DEFECT2S --threads 2 --out $SP/r6_work/measures_AQCSL > $SP/r6_work/logs/measures_AQCSL.log 2>&1
rc1=$?; say "measures exit $rc1"
nice -n 5 $TPY -m tools.learned_spike.r4_screen score --r5 --recipe-dir $OUTROOT/AQCSL --seeds 0 1 2 3 $SCORE_INPUTS2S --tests T-M T-D T-S T-Sp --out $SP/r6_work/score_AQCSL --title "Round 6, recipe AQCSL (arm A2), seeds 0-3" > $SP/r6_work/logs/score_AQCSL.log 2>&1
rc2=$?; say "score exit $rc2"
[ $rc1 = 0 ] && [ $rc2 = 0 ] || exit 1
touch $SP/r6_work/A2_SCREEN_DONE; say "done"

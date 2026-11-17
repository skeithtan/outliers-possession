#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# STATUS: round 6 copy of full_R5.sh (rehearsed for AQCVX); dry-checked for AQCSJ with STEPS=4 in scratch folders (see R6_TOOLS_READY).
# Round 6 full training of RECIPE=AQCSJ (or AQCS): full_R5.sh with the stage50-v2s inputs (r6_env.sh) and CA2S for the OOF step.
# Stage 1: the left-out-provider networks still missing for seeds 0-3 (r5_screen_launch.sh, resumable: DONE networks
#   under $OUTROOT/RECIPE are reused; for AQCSJ the DFL-left-out networks are trained under AQCS and linked by the launcher).
# Stage 2: 8 folds x 4 seeds (kfold --train-only), then the 4 finals with --folds, then the out-of-fold probabilities.
# The switches of stage 2 are the launcher's own (recipe_switches), so screening and full training cannot differ.
# Env: RECIPE RUNDIR CODE (required); see r5_chain_common.sh. Start with nohup. Resumable (kfold/final skip finished models).
CHAIN=full_R6
source ${SCRATCH:?}/r5_work/scripts/r6_chain_common.sh
trap finish EXIT
set -u
say "recipe $RECIPE, RUNDIR $RUNDIR, CODE $CODE ($(git rev-parse HEAD)), screen root $OUTROOT, seeds $SEEDS${STEPS:+, STEPS $STEPS (REHEARSAL)}"
say "stage 1: left-out-provider networks, seeds $SEEDS"
STEPS=${STEPS:-} EVAL_EVERY=${EVAL_EVERY:-} bash tools/learned_spike/r5_screen_launch.sh "$RECIPE" $SEEDS >> "$RUNDIR/logs/$CHAIN.log" 2>&1 \
  || { say "stage 1 failed (see $OUTROOT/$RECIPE/logs/launch.log)"; exit 1; }
SW=$(recipe_switches) || exit 1
say "switches (from the launcher): $SW"
export SW STEP_ARGS EXTD RUNDIR TPY THREADS NICE GLOG
job() {
  kind=$1; a=$2; b=$3
  while ! grep -q "paused=0" "$GLOG.last" 2>/dev/null; do sleep 60; done
  echo "$(date '+%F %T') start $kind $a $b" >> "$RUNDIR/logs/jobs.log"
  case $kind in
    fold)  nice -n $NICE $TPY -m tools.learned_spike.kfold --fold $a --seeds $b --ext $EXTD --out $RUNDIR/folds --threads $THREADS --train-only --no-pff $STEP_ARGS $SW > $RUNDIR/logs/fold${a}_s$b.log 2>&1 ;;
    final) nice -n $NICE $TPY -m tools.learned_spike.final --ext $EXTD --out $RUNDIR/final --seeds $a --threads $THREADS --no-pff --folds $RUNDIR/folds $SW > $RUNDIR/logs/final_s$a.log 2>&1 ;;
  esac
  rc=$?; echo "$(date '+%F %T') end $kind $a $b rc=$rc" >> "$RUNDIR/logs/jobs.log"
  [ $rc = 0 ] || echo "FAILED $kind $a $b" >> "$RUNDIR/logs/FAILED"
}
export -f job
rm -f "$RUNDIR/logs/FAILED"
say "stage 2: 8 folds x seeds $SEEDS, $JOBS at a time"
for s in $SEEDS; do for k in 0 1 2 3 4 5 6 7; do echo "fold $k $s"; done; done > "$RUNDIR/logs/stage2.jobs"
xargs -P "$JOBS" -L 1 bash -c 'job $0 $1 $2' < "$RUNDIR/logs/stage2.jobs"
[ -f "$RUNDIR/logs/FAILED" ] && { say "fold training failed: $(cat $RUNDIR/logs/FAILED | tr '\n' ' ')"; exit 1; }
say "finals (step count = median of the fold models' best steps, --folds)"
for s in $SEEDS; do echo "final $s x"; done > "$RUNDIR/logs/finals.jobs"
xargs -P "$JOBS" -L 1 bash -c 'job $0 $1 $2' < "$RUNDIR/logs/finals.jobs"
[ -f "$RUNDIR/logs/FAILED" ] && { say "finals failed"; exit 1; }
say "out-of-fold probabilities (r3_nested predict, --cache CA2S)"
nice -n $NICE $TPY -m tools.learned_spike.r3_nested predict --folds $RUNDIR/folds --cache $CA2S --seeds $SEEDS --threads $THREADS --out $RUNDIR/oof_p5 > $RUNDIR/logs/oof.log 2>&1 || { say "oof failed"; exit 1; }
say "done"

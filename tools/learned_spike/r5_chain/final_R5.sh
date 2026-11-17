#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# STATUS: NOT REHEARSED as a script (round 5 ended first). r5_final.sh itself was rehearsed directly (REHEARSAL=1, AS_MATCH=DFL-J03WMX, 60-step stage-v3 run): all steps ran, exit 0.
# Round 5 final scoring: tools/learned_spike/r5_final.sh RUNDIR RUNDIR/final_out at the chosen pair and bonus.
#   final_R5.sh external : STOP_BEFORE_LABEL_FREE=1 (preflight, export, costs-fixed, G6, SkillCorner agreement baselines)
#   final_R5.sh full     : the second call, the single competition-match step (G7, G8, decide). ONLY THE COORDINATOR STARTS IT.
# Rehearsal: REHEARSAL=1 AS_MATCH=<development match> (passed through to r5_final.sh; the label-free step then runs on it).
# Env: RECIPE RUNDIR CODE; optional EXT (round 4's ext_copy), FEATURE_CACHE ($C_TRAIN5, or $C_TRAIN5J for AQCVXJ),
#   R3_EVAL_JOBS (3), GUARD_LAST, and r5_final.sh's own options (R3_EVAL_REHEARSAL_GAMES, REPRO_GAMES, DRY_RUN).
MODE=${1:-}; case "$MODE" in external|full) ;; *) echo "usage: $0 external|full"; exit 2;; esac
CHAIN=final_R5_$MODE
source ${SCRATCH:?}/r5_work/scripts/r5_chain_common.sh
trap finish EXIT
set -u
chosen
[ -n "${C23_SHIPS:-}" ] && { say "costs_R5 says c23 ships"; exit 3; }
[ -f "$RUNDIR/gates/oof_ratio.json" ] || { say "run gates_R5.sh first (oof_ratio.json)"; exit 2; }
DEFAULT_FC=$C_TRAIN5; [ "$RECIPE" = AQCVXJ ] && DEFAULT_FC=$C_TRAIN5J
export CA=$CA5 C23_DEV=$SP/r3_out/c23_dev PAIR=$PAIR_HA,$PAIR_AD PRIOR=$PRIOR OOF_RATIO=$RUNDIR/gates/oof_ratio.json
export EXT=${EXT:-$SP/r4_work/ext_copy} FEATURE_CACHE=${FEATURE_CACHE:-$DEFAULT_FC} GUARD_LAST=${GUARD_LAST:-$SP/r3_out/guard/battery.log.last}
export THREADS R3_EVAL_JOBS=${R3_EVAL_JOBS:-3} NICE
[ "$MODE" = external ] && export STOP_BEFORE_LABEL_FREE=1
say "r5_final.sh $MODE: pair $PAIR, prior $PRIOR, CA $CA, FEATURE_CACHE $FEATURE_CACHE, EXT $EXT${REHEARSAL:+, REHEARSAL on ${AS_MATCH:-?}}"
bash tools/learned_spike/r5_final.sh "$RUNDIR" "$RUNDIR/final_out" > "$RUNDIR/logs/r5_final_$MODE.out" 2>&1
rc=$?; say "r5_final.sh exit $rc (0 ok, 3 a gate failed: c23 ships, 4 tooling failure)"; exit $rc

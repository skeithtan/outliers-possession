#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# Round 7 copy of final_R6.sh for candidate E2 (stage50-v2t: CA2T, C_VAL2T, C_TRAIN2TJ; variants R7 / R7E). The R6 STATUS lines below describe the R6 script.
# STATUS: REHEARSED 13 Nov with REHEARSAL=1 AS_MATCH=DFL-J03WN1: external call (STOP_BEFORE_LABEL_FREE) and full call, exit 0; G7 recorded (fail) and the script went on to G8 (a, c, d pass; b fresh clone not run). The competition-match step is NOT rehearsed on the real data (only the coordinator starts it). Also REHEARSED 14 Nov with PF_WEIGHTS (VARIANT=R6E): external and full calls exit 0, repro_check numpy == torch with the member blended.
# Candidate network + per-frame member: set PF_WEIGHTS (the member's weights file, pf_export.py) and optionally
# PF_WEIGHT (0.5); r5_final.sh then runs VARIANT=R7E and scores the blend as engine.learned decodes it (LEARNED_PF on). Unset: the network alone.
# Round 7 final scoring: VARIANT=R7 tools/learned_spike/r5_final.sh RUNDIR RUNDIR/final_out at the chosen pair and bonus
# (stage50-v2t, learned_v7.npz, LEARNED_STAGE_RESULT_R7 / R7E; a failed G7 is recorded and the script goes on to G8).
#   final_R5.sh external : STOP_BEFORE_LABEL_FREE=1 (preflight, export, costs-fixed, G6, SkillCorner agreement baselines)
#   final_R5.sh full     : the second call, the single competition-match step (G7, G8, decide). ONLY THE COORDINATOR STARTS IT.
# Rehearsal: REHEARSAL=1 AS_MATCH=<development match> (passed through to r5_final.sh; the label-free step then runs on it).
# Env: RECIPE RUNDIR CODE; optional EXT (round 4's ext_copy), FEATURE_CACHE ($C_TRAIN2TJ for AQCTJ, $C_TRAIN2T for AQCS),
#   R3_EVAL_JOBS (3), GUARD_LAST, and r5_final.sh's own options (R3_EVAL_REHEARSAL_GAMES, REPRO_GAMES, DRY_RUN).
MODE=${1:-}; case "$MODE" in external|full) ;; *) echo "usage: $0 external|full"; exit 2;; esac
CHAIN=final_R7_$MODE
source ${SCRATCH:?}/r5_work/scripts/r7_chain_common.sh
trap finish EXIT
set -u
chosen
[ -n "${C23_SHIPS:-}" ] && { say "costs_R5 says c23 ships"; exit 3; }
[ -f "$RUNDIR/gates/oof_ratio.json" ] || { say "run gates_R5.sh first (oof_ratio.json)"; exit 2; }
DEFAULT_FC=$C_TRAIN2T; [ "$RECIPE" = AQCTJ ] && DEFAULT_FC=$C_TRAIN2TJ
VAR=R7; [ -n "${PF_WEIGHTS:-}" ] && VAR=R7E
export VARIANT=$VAR PF_WEIGHTS=${PF_WEIGHTS:-} PF_WEIGHT=${PF_WEIGHT:-0.5} CA=$CA2T C23_DEV=$SP/r3_out/c23_dev PAIR=$PAIR_HA,$PAIR_AD PRIOR=$PRIOR OOF_RATIO=$RUNDIR/gates/oof_ratio.json
export EXT=${EXT:-$SP/r4_work/ext_copy} FEATURE_CACHE=${FEATURE_CACHE:-$DEFAULT_FC} GUARD_LAST=${GUARD_LAST:-$SP/r3_out/guard/battery.log.last}
export THREADS R3_EVAL_JOBS=${R3_EVAL_JOBS:-3} NICE
[ "$MODE" = external ] && export STOP_BEFORE_LABEL_FREE=1
say "r5_final.sh $MODE: pair $PAIR, prior $PRIOR, CA $CA, FEATURE_CACHE $FEATURE_CACHE, EXT $EXT${REHEARSAL:+, REHEARSAL on ${AS_MATCH:-?}}"
bash tools/learned_spike/r5_final.sh "$RUNDIR" "$RUNDIR/final_out" > "$RUNDIR/logs/r5_final_$MODE.out" 2>&1
rc=$?; say "r5_final.sh exit $rc (0 ok, 3 G6 or G8 failed: c23 ships, 4 tooling failure; G7 is recorded)"; exit $rc

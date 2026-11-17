#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# Round 7 copy of gates_R6.sh for candidate E2 (stage50-v2t: CA2T, C_VAL2T, C_TRAIN2TJ; variants R7 / R7E). The R6 STATUS lines below describe the R6 script.
# STATUS: REHEARSED 13 Nov on the same scratch run with a stand-in pair (12, 12) bonus 0: G5, G4b, oof ratio, exit 0. Also REHEARSED 13-14 Nov with PF_WEIGHTS (stand-in member): G5, G4b, oof ratio through engine.learned, exit 0.
# Candidate network + per-frame member: set PF_WEIGHTS (the member's weights file, pf_export.py) and optionally
# PF_WEIGHT (0.5); every step then scores the blend as engine.learned decodes it (LEARNED_PF on). Unset: the network alone.
# Round 7 gates (stage50-v2t caches CA2T, C_VAL2T) on external data at the chosen pair and bonus (RUNDIR/costs/chosen.env): G5 (the eight development matches
# out of fold), G4b (SkillCorner out of fold, r4_screen score-sc-oof), the oof-ratio JSON of G7 (r4_oof_ratio).
# Round 4's gates_oof.sh generalised. Env: RECIPE RUNDIR CODE. Reads no PFF and nothing of the competition match.
CHAIN=gates_R7
source ${SCRATCH:?}/r5_work/scripts/r7_chain_common.sh
trap finish EXIT
set -u
chosen
[ -n "${C23_SHIPS:-}" ] && { say "costs_R5 says c23 ships: no gates"; exit 3; }
G=$RUNDIR/gates; mkdir -p $G
C23_DEV=$SP/r3_out/c23_dev; C23_DFL=$SP/r4_work/c23_dfl_c; C23_SC=$SP/sc_fine/c23_sc
REH=${STEPS:+--rehearsal}
PFA=""; [ -n "${PF_WEIGHTS:-}" ] && PFA="--pf-weights $PF_WEIGHTS --pf-weight ${PF_WEIGHT:-0.5}"
say "pair ($PAIR_HA, $PAIR_AD), bonus $PRIOR, folds $RUNDIR/folds seeds $SEEDS"
CA=$CA2T C_VAL=$C_VAL2T C23_DEV=$C23_DEV C23_DFL=$C23_DFL ENGINE_CACHE=$SP/r4_work/engine_data_raw PF_WEIGHTS=${PF_WEIGHTS:-} PF_WEIGHT=${PF_WEIGHT:-0.5} nice -n $NICE $TPY $SP/r5_work/scripts/g5_dev_oof.py $RUNDIR/folds $G/g5_dev_oof.json $PAIR_HA $PAIR_AD $PRIOR > $G/g5.log 2>&1
r1=$?; say "G5 exit $r1: $(tail -1 $G/g5.log)"
nice -n $NICE $TPY -m tools.learned_spike.r4_screen score-sc-oof --folds $RUNDIR/folds --seeds $SEEDS --cache $CA2T --sc-labels $PG --c23-sc $C23_SC --engine-cache $SP/r4_work/engine_data_raw --pairs $PAIR_HA $PAIR_AD --prior-bonus $PRIOR --threads $THREADS $REH $PFA --out $G/g4b_sc_oof --title "G4b ($RECIPE): fold models, SkillCorner seen, out of fold, pair ($PAIR_HA, $PAIR_AD), prior $PRIOR" > $G/g4b.log 2>&1
r2=$?; say "G4b exit $r2"
nice -n $NICE $TPY -m tools.learned_spike.r4_oof_ratio --folds $RUNDIR/folds --seeds $SEEDS --cache $CA2T --dfl-cache $C_VAL2T --c23-dev $C23_DEV --c23-dfl $C23_DFL --c23-sc $C23_SC --pair $PAIR_HA $PAIR_AD --prior-bonus $PRIOR --threads $THREADS --engine-cache $SP/r4_work/engine_data_raw $PFA --out $G/oof_ratio.json > $G/oof_ratio.log 2>&1
r3=$?; say "oof ratio exit $r3 ($G/oof_ratio.json)"
[ $r1 = 0 ] && [ $r2 = 0 ] && [ $r3 = 0 ]

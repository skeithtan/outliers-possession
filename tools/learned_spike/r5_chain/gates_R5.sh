#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# STATUS: NOT REHEARSED as a script (round 5 ended first). Its three steps were rehearsed on a stage-v3 rehearsal run: g5_dev_oof.py, r4_screen score-sc-oof, r4_oof_ratio (all exit 0).
# Round 5 gates on external data at the chosen pair and bonus (RUNDIR/costs/chosen.env): G5 (the eight development matches
# out of fold), G4b (SkillCorner out of fold, r4_screen score-sc-oof), the oof-ratio JSON of G7 (r4_oof_ratio).
# Round 4's gates_oof.sh generalised. Env: RECIPE RUNDIR CODE. Reads no PFF and nothing of the competition match.
CHAIN=gates_R5
source ${SCRATCH:?}/r5_work/scripts/r5_chain_common.sh
trap finish EXIT
set -u
chosen
[ -n "${C23_SHIPS:-}" ] && { say "costs_R5 says c23 ships: no gates"; exit 3; }
G=$RUNDIR/gates; mkdir -p $G
C23_DEV=$SP/r3_out/c23_dev; C23_DFL=$SP/r4_work/c23_dfl_c; C23_SC=$SP/sc_fine/c23_sc
REH=${STEPS:+--rehearsal}
say "pair ($PAIR_HA, $PAIR_AD), bonus $PRIOR, folds $RUNDIR/folds seeds $SEEDS"
CA=$CA5 C_VAL=$C_VAL5 C23_DEV=$C23_DEV C23_DFL=$C23_DFL nice -n $NICE $TPY $SP/r5_work/scripts/g5_dev_oof.py $RUNDIR/folds $G/g5_dev_oof.json $PAIR_HA $PAIR_AD $PRIOR > $G/g5.log 2>&1
r1=$?; say "G5 exit $r1: $(tail -1 $G/g5.log)"
nice -n $NICE $TPY -m tools.learned_spike.r4_screen score-sc-oof --folds $RUNDIR/folds --seeds $SEEDS --cache $CA5 --sc-labels $PG --c23-sc $C23_SC --engine-cache $SP/r4_work/engine_data_raw --pairs $PAIR_HA $PAIR_AD --prior-bonus $PRIOR --threads $THREADS $REH --out $G/g4b_sc_oof --title "G4b ($RECIPE): fold models, SkillCorner seen, out of fold, pair ($PAIR_HA, $PAIR_AD), prior $PRIOR" > $G/g4b.log 2>&1
r2=$?; say "G4b exit $r2"
nice -n $NICE $TPY -m tools.learned_spike.r4_oof_ratio --folds $RUNDIR/folds --seeds $SEEDS --cache $CA5 --dfl-cache $C_VAL5 --c23-dev $C23_DEV --c23-dfl $C23_DFL --c23-sc $C23_SC --pair $PAIR_HA $PAIR_AD --prior-bonus $PRIOR --threads $THREADS --out $G/oof_ratio.json > $G/oof_ratio.log 2>&1
r3=$?; say "oof ratio exit $r3 ($G/oof_ratio.json)"
[ $r1 = 0 ] && [ $r2 = 0 ] && [ $r3 = 0 ]

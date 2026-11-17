#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# STATUS: NOT REHEARSED: neither the script nor r4_pff_test on the SkillCorner stand-ins ran in round 5 (round 5 ended first). PFF was never read.
# Round 5 G3b: the final models on the 43 open PFF matches (r4_pff_test run and score), at the chosen pair and bonus.
# THE ONLY SCRIPT THAT READS PFF; ONLY THE COORDINATOR STARTS IT, once. Round 4's g3b_pff.sh generalised.
# Rehearsal: GAMES_SET=sc runs the same tool on the 20 SkillCorner matches (stand-ins; no PFF file is read).
# Env: RECIPE RUNDIR CODE; optional GAMES_SET (pff), TP_JOBS (4), C23_PFF (c23 on PFF: round 4's r4_work/tp/c23_pff).
CHAIN=g3b_R5
source ${SCRATCH:?}/r5_work/scripts/r5_chain_common.sh
trap finish EXIT
set -u
chosen
[ -n "${C23_SHIPS:-}" ] && { say "costs_R5 says c23 ships"; exit 3; }
GS=${GAMES_SET:-pff}
case "$GS" in pff) C23=${C23_PFF:-$SP/r4_work/tp/c23_pff}; W=$RUNDIR/tp;; sc) C23=$SP/sc_fine/c23_sc; W=$RUNDIR/tp_rehearsal_sc;; *) echo "GAMES_SET pff or sc"; exit 2;; esac
mkdir -p $W
WTS=$RUNDIR/g3b_weights.npz
NETS=""; for s in $SEEDS; do NETS="$NETS $RUNDIR/final/net_s$s.pt"; done
nice -n $NICE $TPY -m tools.learned_spike.export --out $WTS $NETS > $W/export.log 2>&1 || { say "export failed"; exit 1; }
say "games set $GS, weights $WTS ($($EPY -c "import sys; from engine import learned; print(learned.model_version(learned.load_weights(sys.argv[1])))" $WTS)), pair ($PAIR_HA, $PAIR_AD), bonus $PRIOR, c23 $C23"
OMP_NUM_THREADS=1 nice -n $NICE $EPY -m tools.learned_spike.r4_pff_test run --games-set $GS --weights $WTS --pair $PAIR_HA $PAIR_AD --prior-bonus $PRIOR --work $W --jobs ${TP_JOBS:-4} > $W/run.log 2>&1 || { say "run failed"; exit 1; }
nice -n $NICE $EPY -m tools.learned_spike.r4_pff_test score --games-set $GS --weights $WTS --pair $PAIR_HA $PAIR_AD --prior-bonus $PRIOR --work $W --c23 $C23 --out $W/result > $W/score.log 2>&1 || { say "score failed"; exit 1; }
say "result: $W/result ($(tail -1 $W/score.log))"

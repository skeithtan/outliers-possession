#!/bin/bash
# Round 8 arm N, test (4): final networks of one set on the 43 OPEN PFF matches (test only), member F5T (final) x 0.25,
# (12, 12), prior 0, kickoff rule, through the engine (r4_pff_test run + score, as g3b_R7.sh for E2).
# usage: n_pff.sh SET SEED [SEED ...]   SET: E2 | K200 | K400 (networks $NOUT/sets/SET/final/net_s<s>.pt)
set -u
cd "$(dirname "$0")/../../.."
source tools/learned_spike/r8_scripts/n_env.sh
SET=$1; shift; SEEDS="$*"; TAG=$SET"_s"$(echo $SEEDS | tr ' ' '_')
W=$NOUT/pff/$TAG; mkdir -p $W
NETS=""; for s in $SEEDS; do NETS="$NETS $NOUT/sets/$SET/final/net_s$s.pt"; done
WTS=$W/weights.npz
EPY=$HOME/.venvs/barca/bin/python
batt() { pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d %; }
while [ "$(batt)" -lt 35 ] || ! grep -q "paused=0" "$GLOG.last" 2>/dev/null; do sleep 60; done
$TPY -m tools.learned_spike.export --out $WTS $NETS > $W/export.log 2>&1 || { echo "export failed"; exit 1; }
PFA="--pf-weights $R7M/pf_F5T_ffinal.npz --pf-weight 0.25"
OMP_NUM_THREADS=1 nice -n 10 $EPY -m tools.learned_spike.r4_pff_test run --games-set pff --weights $WTS --pair 12 12 $PFA --work $W --jobs ${TP_JOBS:-1} > $W/run.log 2>&1 || { echo "run failed"; exit 1; }
nice -n 10 $EPY -m tools.learned_spike.r4_pff_test score --games-set pff --weights $WTS --pair 12 12 $PFA --work $W --c23 $SP/r4_work/tp/c23_pff --out $W/result > $W/score.log 2>&1 || { echo "score failed"; exit 1; }
md5 -q $WTS > $W/weights.md5
echo "$(date '+%F %T') $TAG done: $(tail -1 $W/score.log)"

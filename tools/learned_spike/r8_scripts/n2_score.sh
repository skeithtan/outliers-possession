#!/bin/bash
# Round 8 N2 (protocol R8 note 6.3): the four tests of the N2 fine-tunes (four seeds) beside E2's four-seed scores, then the table.
set -u
cd "$(dirname "$0")/../../.."
source tools/learned_spike/r8_scripts/n_env.sh
bash tools/learned_spike/r8_scripts/n_sets.sh
L=$NOUT/logs
nice -n 10 $TPY -m tools.learned_spike.r8_scripts.n_score --root N2=$NOUT/sets/N2 --tests T-Moof T-D T-S T-Sp --seeds 0 1 2 3 --out $NOUT/score/E2_s0123 > $L/n2_score.log 2>&1 || exit 1
nice -n 10 $TPY -m tools.learned_spike.r8_scripts.n_pff score --inputs $NOUT/pff_inputs --c23 $SP/r4_work/tp/c23_pff --out $NOUT/pff_score --root N2=$NOUT/sets/N2 --seeds 0 1 2 3 > $L/n2_pff.log 2>&1 || exit 1
$TPY -m tools.learned_spike.r8_scripts.n_report --score $NOUT/score/E2_s0123 --pff $NOUT/pff_score --seeds 0123 --base E2 --cands N2 > $NOUT/TABLES_N2.md 2> $L/n2_report.log
echo "n2 scoring done"

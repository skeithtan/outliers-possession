#!/bin/bash
# Round 8 arm N: the remaining scores once every fine-tune and the PFF inputs exist (one process at a time),
# then the tables. Seeds 0 and 1 (cut for time, see REPORT.md); E2's four seeds on PFF as the reproduction.
set -u
cd "$(dirname "$0")/../../.."
source tools/learned_spike/r8_scripts/n_env.sh
bash tools/learned_spike/r8_scripts/n_sets.sh
L=$NOUT/logs
nice -n 10 $TPY -m tools.learned_spike.r8_scripts.n_score --root K200=$NOUT/sets/K200 --root K400=$NOUT/sets/K400 --tests T-Moof T-D T-S T-Sp --seeds 0 1 --out $NOUT/score/s01 > $L/score_K_all.log 2>&1 || exit 1
RUNS=$SP/r7_gates/E2/tp/m_31e88976ea8d_12_12_pf0b5e81aaw0.25/runs
P="nice -n 10 $TPY -m tools.learned_spike.r8_scripts.n_pff score --inputs $NOUT/pff_inputs --c23 $SP/r4_work/tp/c23_pff --out $NOUT/pff_score"
$P --root E2=$NOUT/sets/E2 --seeds 0 1 2 3 --check-csv-dir $RUNS > $L/pff_E2_s0123.log 2>&1 || exit 1
$P --root E2=$NOUT/sets/E2 --root K200=$NOUT/sets/K200 --root K400=$NOUT/sets/K400 --seeds 0 1 > $L/pff_s01.log 2>&1 || exit 1
$TPY -m tools.learned_spike.r8_scripts.n_report --score $NOUT/score/s01 --pff $NOUT/pff_score --seeds 01 > $NOUT/TABLES_s01.md 2> $L/report.log
echo "final scoring done"

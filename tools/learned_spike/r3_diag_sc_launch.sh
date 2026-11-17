#!/bin/bash
# The unseen-provider probe on real automatic tracking (r3_diag.py, pool no-skillcorner;
# informational, decides nothing).
#
# usage: tools/learned_spike/r3_diag_sc_launch.sh OUT CACHE TORCH_PYTHON GUARD_LOG C23_DIR \
#          R3B_DIR SC_FINE C23_SC
#   OUT        new output folder outside the repository
#   CACHE      the round 2 build.py cache (labels hold `raw`), read only
#   GUARD_LOG  log of a RUNNING battery guard (as r3_launch.sh)
#   C23_DIR    c23 CSVs of the 8 development matches (r3_c23_dev.py)
#   R3B_DIR    the R3b run folder (reads interim/p5, interim/check.json, folds/fold*/net_s0.pt)
#   SC_FINE    cache_possgroup of tools/skillcorner_fine.py (fine SkillCorner labels)
#   C23_SC     c23 CSVs of the 20 SkillCorner matches (r3_c23_dev.py --games SC-...)
# MTAH_DATA_DIR must be set (engine.data.graded_blocks).
#
# Order: train no-skillcorner (seed 0, the R3b recipe, 3 Metrica + 5 DFL, one worker, 2 threads),
# its probabilities on the 8 development and 20 SkillCorner matches, the R3b seed-0 fold models on
# their held-out SkillCorner matches (no training), then the scores (OUT/RESULT_sc.md,
# OUT/result_sc.json). Before every job it waits until the guard reports paused=0. It starts
# nothing if the guard is not running. It stops nothing but its own children. Resumable: a
# finished model (files present with the same step count) is skipped.
# Smoke: STEPS=20 EVAL_EVERY=10 THREADS=1.
set -euo pipefail
OUT=$1; CACHE=$2; TPY=$3; GLOG=$4; C23=$5; R3B=$6; FINE=$7; C23SC=$8
THREADS=${THREADS:-2}
STEP_ARGS=""; [ -n "${STEPS:-}" ] && STEP_ARGS="--steps $STEPS --eval-every ${EVAL_EVERY:-200}"
cd "$(dirname "$0")/../.."
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
gpid=$(cat "$GLOG.pid" 2>/dev/null || true)
if [ -z "$gpid" ] || ! kill -0 "$gpid" 2>/dev/null; then echo "battery guard not running ($GLOG.pid)"; exit 2; fi
wait_guard() { while ! grep -q "paused=0" "$GLOG.last" 2>/dev/null; do sleep 60; done; }
for f in "$FINE/labels" "$C23SC" "$R3B/interim/check.json"; do
  [ -e "$f" ] || { echo "missing $f"; exit 2; }
done
mkdir -p "$OUT/logs"
name=no-skillcorner
echo "$(stamp) start $name, seed 0, threads $THREADS ${STEP_ARGS:-full steps}, commit $(git rev-parse HEAD)" | tee -a "$OUT/logs/launch.log"
wait_guard
echo "$(stamp) train $name" | tee -a "$OUT/logs/launch.log"
$TPY -m tools.learned_spike.r3_diag train --name "$name" --cache "$CACHE" --out "$OUT" \
  --threads "$THREADS" $STEP_ARGS > "$OUT/logs/train_$name.log" 2>&1
echo "$(stamp) $(tail -1 "$OUT/logs/train_$name.log")" | tee -a "$OUT/logs/launch.log"
wait_guard
echo "$(stamp) predict $name" | tee -a "$OUT/logs/launch.log"
$TPY -m tools.learned_spike.r3_diag predict --name "$name" --cache "$CACHE" --out "$OUT" \
  --threads "$THREADS" > "$OUT/logs/predict_$name.log" 2>&1
wait_guard
echo "$(stamp) predict-folds (R3b seed 0)" | tee -a "$OUT/logs/launch.log"
$TPY -m tools.learned_spike.r3_diag predict-folds --cache "$CACHE" --out "$OUT" --r3b "$R3B" \
  --threads "$THREADS" > "$OUT/logs/predict_folds.log" 2>&1
wait_guard
echo "$(stamp) score" | tee -a "$OUT/logs/launch.log"
$TPY -m tools.learned_spike.r3_diag score --names "$name" --cache "$CACHE" --out "$OUT" \
  --c23 "$C23" --r3b "$R3B" --sc-fine "$FINE" --c23-sc "$C23SC" 2>&1 \
  | tee -a "$OUT/logs/score.log" | sed "s/^/$(stamp) /" >> "$OUT/logs/launch.log"
echo "$(stamp) done" | tee -a "$OUT/logs/launch.log"

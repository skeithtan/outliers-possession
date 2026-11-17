#!/bin/bash
# R3b unseen-provider diagnostic launcher (LEARNED_STAGE_PROTOCOL_R3.md §10, informational).
#
# usage: tools/learned_spike/r3_diag_launch.sh OUT CACHE TORCH_PYTHON GUARD_LOG C23_DIR R3B_DIR
#   OUT        new output folder outside the repository
#   CACHE      the round 2 build.py cache (labels hold `raw`)
#   GUARD_LOG  log of a RUNNING battery guard (as r3_launch.sh)
#   C23_DIR    c23 CSVs of the 8 development matches (r3_c23_dev.py)
#   R3B_DIR    the R3b run folder (reads interim/p5 and interim/check.json only)
# MTAH_DATA_DIR must be set (engine.data.graded_blocks), as for r3_nested check.
#
# Order: model no-metrica, then model no-dfl (seed 0, the R3b recipe, one training worker,
# 2 threads), then the probabilities of both, then the scores (OUT/RESULT.md, OUT/result.json).
# Before every job it waits until the guard reports paused=0. It starts nothing if the guard is
# not running. It stops nothing but its own children. Resumable: a finished model (files present
# with the same step count) is skipped; predict and score are cheap and always rerun.
# Smoke: STEPS=20 EVAL_EVERY=10 THREADS=1.
set -euo pipefail
OUT=$1; CACHE=$2; TPY=$3; GLOG=$4; C23=$5; R3B=$6
THREADS=${THREADS:-2}
STEP_ARGS=""; [ -n "${STEPS:-}" ] && STEP_ARGS="--steps $STEPS --eval-every ${EVAL_EVERY:-200}"
cd "$(dirname "$0")/../.."
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
gpid=$(cat "$GLOG.pid" 2>/dev/null || true)
if [ -z "$gpid" ] || ! kill -0 "$gpid" 2>/dev/null; then echo "battery guard not running ($GLOG.pid)"; exit 2; fi
wait_guard() { while ! grep -q "paused=0" "$GLOG.last" 2>/dev/null; do sleep 60; done; }
mkdir -p "$OUT/logs"
echo "$(stamp) start, diagnostic seed 0, threads $THREADS ${STEP_ARGS:-full steps}, commit $(git rev-parse HEAD)" | tee -a "$OUT/logs/launch.log"
for name in no-metrica no-dfl; do
  wait_guard
  echo "$(stamp) train $name" | tee -a "$OUT/logs/launch.log"
  $TPY -m tools.learned_spike.r3_diag train --name "$name" --cache "$CACHE" --out "$OUT" \
    --threads "$THREADS" $STEP_ARGS > "$OUT/logs/train_$name.log" 2>&1
  echo "$(stamp) $(tail -1 "$OUT/logs/train_$name.log")" | tee -a "$OUT/logs/launch.log"
done
for name in no-metrica no-dfl; do
  wait_guard
  echo "$(stamp) predict $name" | tee -a "$OUT/logs/launch.log"
  $TPY -m tools.learned_spike.r3_diag predict --name "$name" --cache "$CACHE" --out "$OUT" \
    --threads "$THREADS" > "$OUT/logs/predict_$name.log" 2>&1
done
wait_guard
echo "$(stamp) score" | tee -a "$OUT/logs/launch.log"
$TPY -m tools.learned_spike.r3_diag score --cache "$CACHE" --out "$OUT" --c23 "$C23" \
  --r3b "$R3B" 2>&1 | tee -a "$OUT/logs/score.log" | sed "s/^/$(stamp) /" >> "$OUT/logs/launch.log"
echo "$(stamp) done" | tee -a "$OUT/logs/launch.log"

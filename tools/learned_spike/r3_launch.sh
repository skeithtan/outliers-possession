#!/bin/bash
# Round 3 training launcher (LEARNED_STAGE_PROTOCOL_R3.md Section 2; DRAFT until frozen).
#
# usage: tools/learned_spike/r3_launch.sh CACHE EXT OUT TORCH_PYTHON ENGINE_PYTHON GUARD_LOG C23_DIR
#   CACHE, EXT     the round 2 build.py cache and c22.py cache (labels already hold `raw`)
#   OUT            new output folder outside the repository
#   GUARD_LOG      log of a RUNNING battery guard (battery_guard2.sh GUARD_LOG, HANDOVER=0)
#   C23_DIR        c23 CSVs of the 8 development matches (r3_c23_dev.py)
#
# Order: seed 0 on folds 0-7 (one training worker, 2 threads), then the interim kill check
# (r3_nested predict + check, Section 3; exit 3 = kill: stop, c23 ships), then seeds 1-3 fold by
# fold, then fold selection and test predictions (kfold --select-only, per-fold costs recorded on
# the raw references), then the 4 final models, then the nested final costs (r3_nested check on
# the 4-seed fold ensembles). Before every job it waits until the guard reports paused=0. It
# starts nothing if the guard is not running. It stops nothing but its own children.
# R3b (no PFF, SkillCorner at 1/4): NO_PFF=1 MIX=sc_quarter.
# Smoke: STEPS=20 EVAL_EVERY=10 FINAL_STEPS=10 FOLDS="0" SEEDS="0" NO_CHECK=1.
set -euo pipefail
CACHE=$1; EXT=$2; OUT=$3; TPY=$4; EPY=$5; GLOG=$6; C23=$7
FOLDS=${FOLDS:-0 1 2 3 4 5 6 7}; SEEDS=${SEEDS:-0 1 2 3}; THREADS=2
SW="--target raw --short-weight ${SHORT_WEIGHT:-off} --mix ${MIX:-match}${NO_PFF:+ --no-pff}"
STEP_ARGS=""; [ -n "${STEPS:-}" ] && STEP_ARGS="--steps $STEPS --eval-every ${EVAL_EVERY:-200}"
FINAL_ARGS="--folds $OUT/folds"; [ -n "${FINAL_STEPS:-}" ] && FINAL_ARGS="--steps $FINAL_STEPS"
cd "$(dirname "$0")/../.."
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
gpid=$(cat "$GLOG.pid" 2>/dev/null || true)
if [ -z "$gpid" ] || ! kill -0 "$gpid" 2>/dev/null; then echo "battery guard not running ($GLOG.pid)"; exit 2; fi
wait_guard() { while ! grep -q "paused=0" "$GLOG.last" 2>/dev/null; do sleep 60; done; }
mkdir -p "$OUT/logs" "$OUT/folds" "$OUT/final" "$OUT/interim"
echo "$(stamp) start, switches: $SW, commit $(git rev-parse HEAD)" | tee -a "$OUT/logs/launch.log"
train_seed() {
  for k in $FOLDS; do
    wait_guard
    echo "$(stamp) fold $k seed $1" | tee -a "$OUT/logs/launch.log"
    $TPY -m tools.learned_spike.kfold --fold "$k" --seeds "$1" --cache "$CACHE" --ext "$EXT" \
      --out "$OUT/folds" --threads $THREADS --train-only $SW $STEP_ARGS \
      > "$OUT/logs/fold${k}_s$1.log" 2>&1
  done
}
first=$(echo $SEEDS | awk '{print $1}')
train_seed "$first"
if [ -z "${NO_CHECK:-}" ]; then
  wait_guard
  $TPY -m tools.learned_spike.r3_nested predict --folds "$OUT/folds" --cache "$CACHE" \
    --seeds "$first" --out "$OUT/interim/p5" > "$OUT/logs/interim_predict.log" 2>&1
  set +e
  $TPY -m tools.learned_spike.r3_nested check --pred "$OUT/interim/p5" --cache "$CACHE" \
    --c23 "$C23" --out "$OUT/interim/check.json" | tee -a "$OUT/logs/launch.log"
  code=${PIPESTATUS[0]}; set -e
  if [ "$code" != 0 ]; then echo "$(stamp) INTERIM KILL (exit $code): training stops, c23 ships" | tee -a "$OUT/logs/launch.log"; exit 3; fi
fi
for s in $SEEDS; do [ "$s" = "$first" ] || train_seed "$s"; done
for k in $FOLDS; do
  wait_guard
  $TPY -m tools.learned_spike.kfold --fold "$k" --seeds $SEEDS --cache "$CACHE" --ext "$EXT" \
    --out "$OUT/folds" --threads $THREADS --select-only --select-ref raw $SW $STEP_ARGS \
    --engine-python "$EPY" > "$OUT/logs/fold${k}_select.log" 2>&1
done
wait_guard
echo "$(stamp) final models" | tee -a "$OUT/logs/launch.log"
$TPY -m tools.learned_spike.final --cache "$CACHE" --ext "$EXT" --out "$OUT/final" --seeds $SEEDS \
  --threads $THREADS $SW $FINAL_ARGS > "$OUT/logs/final.log" 2>&1
if [ -z "${NO_CHECK:-}" ]; then
  $TPY -m tools.learned_spike.r3_nested predict --folds "$OUT/folds" --cache "$CACHE" \
    --seeds $SEEDS --out "$OUT/oof_p5" > "$OUT/logs/oof_predict.log" 2>&1
  $TPY -m tools.learned_spike.r3_nested check --pred "$OUT/oof_p5" --cache "$CACHE" \
    --c23 "$C23" --out "$OUT/final/nested_costs.json" | tee -a "$OUT/logs/launch.log" || true
fi
echo "$(stamp) done" | tee -a "$OUT/logs/launch.log"

#!/bin/bash
# Run the training of the learned stage (LEARNED_STAGE_PROTOCOL.md, Sections 2, 3 and 6, with
# LEARNED_STAGE_PROTOCOL_R2.md Section 1: the 43 PFF training matches join every fold's training set
# and the final models, raw view only; M's cost grid keeps H-A <= 2 x alive-dead).
#
# usage: tools/learned_spike/run_stage.sh CACHE EXT OUT TORCH_PYTHON ENGINE_PYTHON
#   CACHE          build.py cache: feats/<view>/<game>.npz and labels/<game>.npz
#   EXT            c22.py cache: views/raw/engine_data (engine npz), views/raw/runs, refs
#   OUT            output folder, outside the repository (weights are never committed)
#   TORCH_PYTHON   python of the torch venv (torch 2.8.0, numpy 2.0.2, pandas 2.3.3)
#   ENGINE_PYTHON  python of the engine venv (~/.venvs/barca)
#
# Steps:
#   1. The feature caches of the 28 matches and the 43 PFF training matches (existing files are
#      kept; every PFF reference md5 is checked against R2 Section 8 first).
#   2. c22 on the 28 matches (engine.run, default config, cached by run key). PFF needs no c22 run.
#   3. 32 fold models (8 folds x seeds 0-3), 4 at a time with 2 threads each.
#   4. Per fold: costs on the validation pair (costs.json first), then the test predictions.
#   5. The final ensemble on the 28 + 43 matches: seeds 0-3, 4 at a time, S steps from the folds.
#   6. The final costs: the median of the 8 fold costs per candidate.
# Each job writes its log to OUT/logs/. The script stops at the first failure.
#
# Smoke test only: the variables STEPS, EVAL_EVERY and FINAL_STEPS override the step counts
# of the protocol (2000, 200, and the median of the fold best steps). FOLDS, SEEDS and PAR
# override the folds (0-7), the seeds (0-3) and the jobs at a time (4). With fewer than eight
# folds, step 6 (the median of the 8 fold costs) is skipped.
set -euo pipefail
CACHE=$1; EXT=$2; OUT=$3; TPY=$4; EPY=$5
PAR=${PAR:-4}; THREADS=2
FOLDS=${FOLDS:-0 1 2 3 4 5 6 7}; SEEDS=${SEEDS:-0 1 2 3}
STEP_ARGS=""; FINAL_ARGS="--folds $OUT/folds"
if [ -n "${STEPS:-}" ]; then STEP_ARGS="--steps $STEPS --eval-every ${EVAL_EVERY:-200}"; fi
if [ -n "${FINAL_STEPS:-}" ]; then FINAL_ARGS="--steps $FINAL_STEPS"; fi
cd "$(dirname "$0")/../.."
mkdir -p "$OUT/logs" "$OUT/folds" "$OUT/final"
ALL=$($TPY -c "from tools.learned_spike import folds; print(' '.join(folds.ALL))")
TRAIN_ALL=$($TPY -c "from tools.learned_spike import folds; print(' '.join(folds.TRAIN_ALL))")
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "$(stamp) 1. feature caches"
$EPY -m tools.learned_spike.build --cache "$CACHE" --ref-cache "$EXT" \
  --engine-cache "$EXT/views/raw/engine_data" --no-pff-engine-cache --games $TRAIN_ALL \
  > "$OUT/logs/build.log" 2>&1

echo "$(stamp) 2. c22 runs"
$EPY -m tools.learned_spike.c22 --ext "$EXT" --jobs $PAR --games $ALL > "$OUT/logs/c22.log" 2>&1

echo "$(stamp) 3. fold models"
export CACHE EXT OUT TPY EPY THREADS STEP_ARGS FINAL_ARGS SEEDS
for k in $FOLDS; do for s in $SEEDS; do echo "$k $s"; done; done | \
  xargs -P $PAR -L 1 bash -c '$TPY -m tools.learned_spike.kfold --fold $0 --seeds $1 \
    --cache "$CACHE" --ext "$EXT" --out "$OUT/folds" --threads $THREADS --train-only $STEP_ARGS \
    --engine-python "$EPY" > "$OUT/logs/fold$0_s$1.log" 2>&1 || { echo "fold $0 seed $1 failed"; exit 255; }'

echo "$(stamp) 4. fold costs and test predictions"
for k in $FOLDS; do echo "$k"; done | \
  xargs -P $PAR -L 1 bash -c '$TPY -m tools.learned_spike.kfold --fold $0 --cache "$CACHE" \
    --ext "$EXT" --out "$OUT/folds" --threads $THREADS --select-only --seeds $SEEDS $STEP_ARGS \
    --engine-python "$EPY" > "$OUT/logs/fold$0_select.log" 2>&1 || { echo "fold $0 select failed"; exit 255; }'

echo "$(stamp) 5. final ensemble"
for s in $SEEDS; do echo "$s"; done | \
  xargs -P $PAR -L 1 bash -c '$TPY -m tools.learned_spike.final --cache "$CACHE" --ext "$EXT" \
    --out "$OUT/final" $FINAL_ARGS --seeds $0 --threads $THREADS \
    > "$OUT/logs/final_s$0.log" 2>&1 || { echo "final seed $0 failed"; exit 255; }'

echo "$(stamp) 6. final costs"
if [ "$FOLDS" = "0 1 2 3 4 5 6 7" ]; then
  $TPY -m tools.learned_spike.final --costs --folds "$OUT/folds" --out "$OUT/final" \
    > "$OUT/logs/final_costs.log" 2>&1
  cat "$OUT/logs/final_costs.log"
else
  echo "smoke: FOLDS=$FOLDS, the median of the 8 fold costs is skipped"
fi
echo "$(stamp) done"

#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# R3 + AUG (defect augmentation, branch learned-r3-aug, docs/plan/calibration/R3_AUG.md):
# folds 0-7, seed 0, R3b recipe + --aug-cache; 4 fold processes x 2 threads. Then the out-of-fold
# check on the 8 development matches (r3_nested predict + check, as the launcher's interim check)
# and the stressed views (r3aug_stress.py) for this run AND for R3b seed 0, same views and seeds.
set -uo pipefail
SP=${SCRATCH:?}
TPY=${TORCH_SCRATCH:?}/venv-torch/bin/python
source $SP/r3_out/scripts/env.sh
export CA TPY; export EXT="$(dirname "$CA")/ext"; EC="$EXT/views/raw/engine_data"
export AUG=$SP/r3aug_cache OUT=$SP/r3aug_train
export SW="--target raw --short-weight off --mix sc_quarter --no-pff --aug-cache $AUG"
cd $SP/r3aug || exit 1
mkdir -p $OUT/logs $OUT/folds $OUT/interim $OUT/stress
LOG=$OUT/logs/launch.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
[ -f $AUG/BUILD_DONE ] && [ -f $AUG/manifest.json ] || { say "aug cache not built"; touch $OUT/AUG_FAILED; exit 1; }
say "start, switches: $SW, commit $(git rev-parse HEAD), dirty: $(git status --porcelain -- tools engine | wc -l | tr -d ' ')"
job() {
  k=$1
  echo "$(date '+%Y-%m-%d %H:%M:%S') fold $k seed 0 starts" >> "$OUT/logs/launch.log"
  $TPY -m tools.learned_spike.kfold --fold "$k" --seeds 0 --cache "$CA" --ext "$EXT" \
    --out "$OUT/folds" --threads 2 --train-only $SW > "$OUT/logs/fold${k}_s0.log" 2>&1 \
    && echo "$(date '+%Y-%m-%d %H:%M:%S') fold $k seed 0 done" >> "$OUT/logs/launch.log" \
    || echo "$(date '+%Y-%m-%d %H:%M:%S') FAILED fold $k seed 0" >> "$OUT/logs/launch.log"
}
export -f job
printf "%s\n" 0 1 2 3 4 5 6 7 | xargs -P 4 -I{} bash -c 'job {}'
for k in 0 1 2 3 4 5 6 7; do
  [ -f $OUT/folds/fold$k/net_s0.pt ] || { say "fold $k missing: stop"; touch $OUT/AUG_FAILED; exit 1; }
done
say "8 models trained; out-of-fold check"
$TPY -m tools.learned_spike.r3_nested predict --folds $OUT/folds --cache "$CA" --seeds 0 \
  --out $OUT/interim/p5 > $OUT/logs/interim_predict.log 2>&1
$TPY -m tools.learned_spike.r3_nested check --pred $OUT/interim/p5 --cache "$CA" \
  --c23 $SP/r3_out/c23_dev --out $OUT/interim/check.json | tee -a "$LOG"
say "interim check exit ${PIPESTATUS[0]} (informational here: no kill rule for this candidate)"
$TPY -m tools.learned_spike.r3aug_stress views --out $AUG/stress --engine-cache "$EC" \
  > $OUT/logs/stress_views.log 2>&1 || { say "stress views failed"; touch $OUT/AUG_FAILED; exit 1; }
for run in aug:$OUT r3b:$SP/r3b_train; do
  name=${run%%:*}; dir=${run#*:}
  $TPY -m tools.learned_spike.r3aug_stress predict --views $AUG/stress --folds $dir/folds \
    --seed 0 --out $OUT/stress/${name}_s0 > $OUT/logs/stress_predict_$name.log 2>&1
  $TPY -m tools.learned_spike.r3aug_stress score --pred $OUT/stress/${name}_s0 --cache "$CA" \
    --check $dir/interim/check.json --out $OUT/stress/${name}_s0.json | sed "s/^/$name /" | tee -a "$LOG"
done
$TPY - <<EOF | tee $OUT/compare.txt | tee -a "$LOG"
import json
a = json.load(open("$OUT/stress/aug_s0.json")); b = json.load(open("$OUT/stress/r3b_s0.json"))
ca = json.load(open("$OUT/interim/check.json")); cb = json.load(open("$SP/r3b_train/interim/check.json"))
print("balanced mean of provider means, spellF1 vs references as provided (seed 0, out of fold)")
print("%-28s %8s %8s %8s" % ("view / costs", "R3b", "AUG", "AUG-R3b"))
print("%-28s %8.4f %8.4f %+8.4f" % ("raw", b["raw"]["balanced"], a["raw"]["balanced"], a["raw"]["balanced"] - b["raw"]["balanced"]))
for v in ("stress", "stress_noswap", "swap_only"):
    for c in ("raw_costs", "nested_costs"):
        x, y = b["views"][v][c + "_balanced"], a["views"][v][c + "_balanced"]
        print("%-28s %8.4f %8.4f %+8.4f" % (v + " / " + c, x, y, y - x))
print("per match, stress / raw_costs:")
for g in sorted(a["views"]["stress"]["raw_costs"]):
    x, y = b["views"]["stress"]["raw_costs"][g], a["views"]["stress"]["raw_costs"][g]
    print("  %-12s %8.4f %8.4f %+8.4f" % (g, x, y, y - x))
print("interim delta vs c23 (raw): R3b", {g: round(m["delta"], 3) for g, m in cb["matches"].items()})
print("interim delta vs c23 (raw): AUG", {g: round(m["delta"], 3) for g, m in ca["matches"].items()})
EOF
touch $OUT/AUG_DONE
say "done"

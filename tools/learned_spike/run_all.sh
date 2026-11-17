#!/bin/bash
# Run the whole learned spike: features, eight LOMO folds, summaries, stress view, hybrids,
# BAR-ATH label-free sanity numbers. See tools/learned_spike/README.md.
#
# usage: tools/learned_spike/run_all.sh WORK TORCH_PYTHON ENGINE_PYTHON EXTCACHE
#   WORK           scratch folder for caches and outputs (outside the repository)
#   TORCH_PYTHON   python of a venv with torch, numpy 2.0.2 and pandas 2.3.3
#   ENGINE_PYTHON  python of the engine venv (~/.venvs/barca)
#   EXTCACHE       the external_sweep cache: Metrica references, raw engine data, c22 runs
set -e
WORK=$1; TPY=$2; EPY=$3; EXT=$4
cd "$(dirname "$0")/../.."
GAMES="METRICA-S1 METRICA-S2 METRICA-S3 DFL-J03WMX DFL-J03WN1 DFL-J03WOH DFL-J03WOY DFL-J03WPY"
mkdir -p "$WORK"

# 1. c22 on the raw view (cached runs are reused) and its run keys.
$EPY tools/analysis/external_sweep.py --field CONTACT_POINT --values '"kink_min"' \
  --out "$WORK/c22_raw" --cache "$EXT" --view raw --jobs 4

# 2. Features and labels.
$EPY -m tools.learned_spike.build --cache "$WORK/cache" --ref-cache "$EXT" \
  --engine-cache "$EXT/views/raw/engine_data"

# 3. Eight folds, four at a time with two threads each (about 25 min per batch on an M-series Mac).
mkdir -p "$WORK/lomo/logs"
i=0
for g in $GAMES; do
  $TPY -m tools.learned_spike.lomo --cache "$WORK/cache" --out "$WORK/lomo" --test $g \
    --threads 2 > "$WORK/lomo/logs/$g.log" 2>&1 &
  i=$((i+1)); if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait

# 4. Model against c22, raw view.
$TPY -m tools.learned_spike.summarise --lomo "$WORK/lomo" --cache "$WORK/cache" \
  --c22-runs "$EXT/views/raw/runs" --c22-keys "$WORK/c22_raw/CONTACT_POINT.json" \
  --out "$WORK/summary_raw.json"

# 5. Stress view: 30 % of the visible graded ball hidden in 2 to 30 s gaps, both systems.
for g in $GAMES; do
  mkdir -p "$WORK/stress/$g"
  if [ $g = METRICA-S3 ]; then R=~/mtah2027/sample-data/data; else R=~/mtah2027/refs; fi
  MTAH_DATA_DIR=$R $EPY -m tools.learned_spike.stress engine --game $g \
    --out "$WORK/stress/$g/c22.csv" --engine-cache "$EXT/views/raw/engine_data" > /dev/null
  $TPY -m tools.learned_spike.stress model --game $g --lomo "$WORK/lomo" \
    --out "$WORK/stress/$g/model.csv"
done
$TPY -m tools.learned_spike.summarise --lomo "$WORK/stress" --cache "$WORK/cache" \
  --pred-name model.csv --c22-name c22.csv --out "$WORK/summary_stress.json"

# 6. Hybrids.
$TPY -m tools.learned_spike.hybrid --lomo "$WORK/lomo" --cache "$WORK/cache" \
  --c22-runs "$EXT/views/raw/runs" --c22-keys "$WORK/c22_raw/CONTACT_POINT.json"
for h in hyb_c22dead hyb_modeldead; do
  $TPY -m tools.learned_spike.summarise --lomo "$WORK/lomo" --cache "$WORK/cache" \
    --c22-runs "$EXT/views/raw/runs" --c22-keys "$WORK/c22_raw/CONTACT_POINT.json" \
    --pred-name $h.csv --out "$WORK/summary_$h.json"
done

# 7. BAR-ATH: label-free numbers only.
$TPY -m tools.learned_spike.bar_sanity --lomo "$WORK/lomo" --out "$WORK/bar_ath_model.csv"

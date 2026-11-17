#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# Round 7 inputs (recipe AQCTJ) at FEATURE_VERSION (default stage50-v2t), new folders under OUTW:
#   aug2t                       round 4's 16 copies (augment version r3) of all 28 matches, DFL from c_train
#   cache_c_train2tj            DFL raw+degraded from c_train_j, the rest linked from BASE (cache_c_train2t)
#   aug2tj                      the DFL copies rebuilt from c_train_j, the other 23 linked from aug2t
#   defect_eval_<version>       the defect evaluation view (k 100, 101, round 3 ranges)
# Markers: dfl_still.json and feature_version.json (r5_inputs mark). Waits for $R5W/V2T_READY (NOWAIT=1 skips).
# Env: CODE (checkout with the v2t features, required). Smoke only: OUTW (output base), BASE, GAMES (subset of
#   matches), NOWAIT=1, SKIP_MARK=1 (incomplete smoke folders cannot be marked). 4 processes at nice 5.
# Ends with $OUTW/R7_INPUTS_BUILT (not in smoke mode). Resumable.
set -uo pipefail
SP=${SCRATCH:?}
source $SP/r3_out/scripts/env.sh
R5W=$SP/r5_work; OUTW=${OUTW:-$R5W}; L=$OUTW/logs; EC=$SP/r4_work/engine_data_raw; PY=$HOME/.venvs/barca/bin/python
: "${CODE:?CODE=checkout that contains stage50-v2t}"
export FEATURE_VERSION=${FEATURE_VERSION:-stage50-v2t}; V=$FEATURE_VERSION
BASE=${BASE:-$R5W/cache_c_train2t}
A=$OUTW/aug2t; CJ=$OUTW/cache_c_train2tj; AJ=$OUTW/aug2tj; DE=$OUTW/defect_eval_$V
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $L; cd $CODE || exit 1
say() { echo "$(date '+%F %T') $*" >> $L/r7_build_inputs_$V.log; }
START=$(( $(cat $L/r7_build_inputs_$V.log 2>/dev/null | wc -l) + 1 ))
say "start, feature version $V, CODE $CODE commit $(git rev-parse HEAD), dirty $(git status --porcelain | wc -l | tr -d ' '), OUTW $OUTW"
[ -n "${NOWAIT:-}" ] || until [ -f $R5W/V2T_READY ]; do sleep 30; done
$PY -c "import sys; sys.path.insert(0,'.'); from tools.learned_spike import features; features.check_version('$V')" \
  || { say "feature version $V unknown to $CODE"; exit 1; }
R="nice -n 5 $PY -m tools.learned_spike.r5_inputs"
ALL=( ${GAMES:-$($PY -c "import sys; sys.path.insert(0,'.'); from tools.learned_spike import folds; print(' '.join(folds.ALL))")} )
DFL=( $(printf '%s\n' "${ALL[@]}" | grep '^DFL-') )
n=${#ALL[@]}; t=$(( (n + 3) / 4 ))
t0=$(date +%s)
for i in 0 1 2 3; do
  part=( "${ALL[@]:$((i * t)):$t}" )
  [ ${#part[@]} -gt 0 ] || continue
  ( $R aug --aug-version r3 --dfl-input c_train --labels $CA --engine-cache $EC --out $A --games ${part[@]} > $L/aug2t_$i.log 2>&1; say "aug2t part $i exit $?" ) &
done
wait; say "aug2t builders done in $(( $(date +%s) - t0 )) s"
[ -n "${SKIP_MARK:-}" ] || { $R mark --kind aug --dir $A >> $L/aug2t_mark.log 2>&1 || { say "aug2t mark FAILED"; exit 1; }; say "aug2t marked"; }
t0=$(date +%s)
( $R cache --kind c_train_j --base $BASE --engine-cache $EC --out $CJ ${GAMES:+--games ${DFL[@]}} > $L/cache_c_train2tj.log 2>&1; say "cache_c_train2tj exit $?"
  [ -n "${SKIP_MARK:-}" ] || { $R mark --kind c_train_j --dir $CJ >> $L/cache_c_train2tj.log 2>&1; say "cache_c_train2tj mark exit $?"; }
  $R defect-eval --labels $CA --engine-cache $EC --out $DE ${GAMES:+--games $(printf '%s\n' "${ALL[@]}" | grep -E '^(DFL|METRICA)-' | tr '\n' ' ')} > $L/defect_eval_$V.log 2>&1; say "defect eval exit $?" ) &
if [ -n "${SKIP_MARK:-}" ]; then   # smoke: an incomplete aug2t cannot be marked; a stand-in marker lets aug2tj link to it
  printf '{"aug_version": "r3", "dfl_input": "c_train", "feature_version": "%s", "smoke": "stand-in, incomplete folder"}\n' "$V" > $A/dfl_still.json
fi
LINK="--link-from $A"
h=$(( (${#DFL[@]} + 1) / 2 ))
G1="${DFL[*]:0:$h}"; G2=""; [ ${#DFL[@]} -gt $h ] && G2="${DFL[*]:$h}"
grp=0
for G in "$G1" "$G2"; do
  grp=$((grp + 1)); [ -n "$G" ] || continue
  ( $R aug --aug-version r3 --dfl-input c_train_j $LINK --labels $CA --engine-cache $EC --out $AJ --games $G > $L/aug2tj_$grp.log 2>&1; say "aug2tj part $grp exit $?" ) &
done
wait; say "J builders done in $(( $(date +%s) - t0 )) s"
if [ -z "${SKIP_MARK:-}" ]; then
  $R mark --kind aug --dir $AJ >> $L/aug2tj_mark.log 2>&1 || { say "aug2tj mark FAILED"; exit 1; }
  for f in $A $CJ $AJ; do [ -f $f/dfl_still.json ] && [ -f $f/feature_version.json ] || { say "$f not marked"; exit 1; }; done
  [ -f $DE/defect_eval.json ] || { say "defect eval incomplete"; exit 1; }
  sed -n "$START,\$p" $L/r7_build_inputs_$V.log | grep -q "exit [1-9]\|FAILED" && { say "a step FAILED"; exit 1; }
  touch $OUTW/R7_INPUTS_BUILT
fi
sed -n "$START,\$p" $L/r7_build_inputs_$V.log | grep -q "exit [1-9]\|FAILED" && { say "a step FAILED"; exit 1; }
say "done"

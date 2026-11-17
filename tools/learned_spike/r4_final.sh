#!/bin/bash
# Round 4 final scoring of the fully trained recipe: G6, G7 and G8 (LEARNED_STAGE_PROTOCOL_R4.md
# Section 5), with the round 3 machinery of r3_eval.py, variant R4.
#
# usage: tools/learned_spike/r4_final.sh RUN_DIR OUT_DIR
#   RUN_DIR  the full training (folds/fold<k>/net_s<s>.pt and model_s<s>.json, final/net_s<s>.pt
#            and final_s<s>.json, oof_p5/), read only
#   OUT_DIR  a new folder outside the repository: OUT_DIR/work, OUT_DIR/learned_v4.npz,
#            OUT_DIR/LEARNED_STAGE_RESULT_R4.md and .json
# Environment (required): CA (build.py cache), MTAH_DATA_DIR (competition data folder), C23_DEV
#   (c23 CSVs of the 8 development matches, raw view), PAIR=H,A and PRIOR=B (the pair and prior
#   bonus of Section 4, from r4_choose costs), OOF_RATIO (JSON of r4_oof_ratio.py at that pair
#   and bonus; G7's count-ratio bound).
#   Optional: EXT, TORCH_PY, ENGINE_PY, GUARD_LAST, EXPLANATIONS (Section 8c JSON),
#   FEATURE_CACHE (the 8c band; default $CA), FRESH_CLONE_LOG (criterion 8 part b), THREADS,
#   NICE, R3_EVAL_JOBS (parallel external engine runs, default 1). Rehearsal only, to keep the
#   load low: R3_EVAL_REHEARSAL_GAMES (the stress step on these matches only) and REPRO_GAMES
#   (repro_check on these open matches only).
# DRY_RUN=1: print every command that would run, run nothing (the STOP-style check).
# STOP_BEFORE_LABEL_FREE=1: stop after every external-match step (criterion 5), before the
#   competition match; re-run without it after the go (every step resumes from its cache).
# REHEARSAL=1 AS_MATCH=<development match>: problems are recorded, not fatal; the switch-off
#   identity (which runs c23 on the competition match) is NOT run; the label-free step runs on
#   AS_MATCH in place of the competition match.
#
# Order: preflight (R4 switches; with the switch-off identity outside a rehearsal); export;
# costs-fixed (no cost search, no walk); stress, criterion 5 = G6 (criterion 4 recorded);
# SkillCorner agreement baselines (8a, 8b); then ONCE the label-free step: bar at the fixed pair
# and prior, criteria 6 and 7 and the four extra bounds (r4bounds, criterion 7x) = G7; criterion
# 8 = G8 (numpy against torch on the 8 open matches with the prior, a second run on the match,
# the learned tests; the fresh clone is part b, FRESH_CLONE_LOG); decide. A failed gate stops
# with exit 3 (c23 ships); a tooling failure exits 4 (fix and re-run, it resumes).
set -euo pipefail
[ $# -eq 2 ] || { sed -n 2,32p "$0"; exit 2; }
abs() { mkdir -p "$1" && (cd "$1" && pwd); }
RUN=$(cd "$1" && pwd); OUT=$(abs "$2"); W=$OUT/work; mkdir -p "$W"
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
case "$OUT/" in "$REPO"/*) echo "OUT_DIR must be outside the repository"; exit 2;; esac
: "${CA:?}"; : "${MTAH_DATA_DIR:?}"; : "${C23_DEV:?}"; : "${PAIR:?PAIR=H,A of Section 4}"
: "${PRIOR:?PRIOR=0, 1 or 2 of Section 4}"; : "${OOF_RATIO:?OOF_RATIO=JSON of r4_oof_ratio.py}"
EXT=${EXT:-$(dirname "$CA")/ext}
TORCH_PY=${TORCH_PY:-$HOME/.venvs/barca-torch/bin/python}
ENGINE_PY=${ENGINE_PY:-$HOME/.venvs/barca/bin/python}
FEATURE_CACHE=${FEATURE_CACHE:-$CA}
TH=${THREADS:-1}
export OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH MKL_NUM_THREADS=$TH VECLIB_MAXIMUM_THREADS=$TH
DRY=${DRY_RUN:-}
E() { if [ -n "$DRY" ]; then echo "  [would run] $ENGINE_PY $*"; else nice -n "${NICE:-15}" "$ENGINE_PY" "$@"; fi; }
T() { if [ -n "$DRY" ]; then echo "  [would run] $TORCH_PY $*"; else nice -n "${NICE:-15}" "$TORCH_PY" "$@"; fi; }
RA="--run $RUN --work $W"
XA="--engine-cache $EXT/views/raw/engine_data --sc-engine-cache $EXT/views/raw/engine_data --ref-cache $EXT"
FINAL=$OUT/learned_v4.npz
LF="--costs-rule fixed"; PF="--variant R4"; MATCH="the competition match (BAR-ATH)"
if [ -n "${REHEARSAL:-}" ]; then
  PF="$PF --rehearsal"; LF="$LF --as-match ${AS_MATCH:?REHEARSAL needs AS_MATCH}"
  BA="--as-match $AS_MATCH"; MATCH="$AS_MATCH (REHEARSAL in place of the competition match)"
else
  PF="$PF --identity"; BA=""
fi
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "$(stamp) $*" | tee -a "$OUT/r4_final.log"; }
wait_guard() { [ -z "${GUARD_LAST:-}" ] || [ -n "$DRY" ] && return 0; while ! grep -q "paused=0" "$GUARD_LAST" 2>/dev/null; do say "battery guard paused, waiting"; sleep 60; done; }
decide() { E -m tools.learned_spike.r3_eval decide --work "$W" --out-dir "$OUT" --variant R4 ${EXPLANATIONS:+--explanations $EXPLANATIONS} "$@" | tee -a "$OUT/r4_final.log"; }
stop() {
  if [ -n "$DRY" ]; then echo "  [gate] on failure: stop ($1), decide, exit 3"; return 0; fi
  if [ -n "${REHEARSAL:-}" ]; then say "REHEARSAL: the real run would stop here: $1"; STOPS="${STOPS:+$STOPS; }$1"; return 0; fi
  say "STOP: $1"; decide --stop "$1"; say "M does not qualify: c23 ships (R4 Section 5)"; exit 3; }
fail() { [ -n "$DRY" ] && return 0; say "TOOLING FAILURE: $1 (not a verdict; fix and re-run, it resumes)"; exit 4; }
gate() { if [ -n "$DRY" ]; then echo "  [would run] $ENGINE_PY -m tools.learned_spike.r3_eval gate --work $W --criteria $*"; return 0; fi
  E -m tools.learned_spike.r3_eval gate --work "$W" ${EXPLANATIONS:+--explanations $EXPLANATIONS} --criteria "$@" | tee -a "$OUT/r4_final.log"; return "${PIPESTATUS[0]}"; }
step() { if [ -n "$DRY" ]; then "$@"; return 0; fi; "$@" 2>>"$OUT/engine_runs.log" | tee -a "$OUT/r4_final.log"; return "${PIPESTATUS[0]}"; }

say "r4_final start${DRY:+ (DRY RUN: nothing runs)}: RUN=$RUN OUT=$OUT pair $PAIR prior $PRIOR, label-free match: $MATCH, commit $(git rev-parse HEAD)"
[ -n "$(git status --porcelain --untracked-files=no -- engine games.py tools/learned_spike)" ] && say "warning: engine/ or tools/learned_spike has uncommitted changes"

wait_guard; say "0 preflight (variant R4)${REHEARSAL:+; no switch-off identity in a rehearsal}"
step E -m tools.learned_spike.r3_eval preflight $RA $PF || stop "preflight failed (not a finished R4 run, or the switch-off identity failed)"

wait_guard; say "1 export (torch venv): fold npz, 32-model npz, final $FINAL"
step T -m tools.learned_spike.evaluate export --stage "$RUN" --work "$W" --final-out "$FINAL" || fail "export failed"
if [ -z "$DRY" ]; then md5 -q "$FINAL" > "$OUT/learned_v4.npz.md5" 2>/dev/null || md5sum "$FINAL" | cut -d' ' -f1 > "$OUT/learned_v4.npz.md5"; say "learned_v4.npz md5 $(cat "$OUT/learned_v4.npz.md5")"; fi

wait_guard; say "2 costs-fixed: pair $PAIR, prior bonus $PRIOR, kickoff rule on"
step E -m tools.learned_spike.r3_eval costs-fixed $RA --pair "$PAIR" --prior-bonus "$PRIOR" || fail "costs-fixed"

wait_guard; say "3 stress and uniform30 views on the 28 external matches: criterion 5 = G6 (criterion 4 recorded)"
step E -m tools.learned_spike.r3_eval stress $RA $XA || fail "stress runs failed"
gate 5 || stop "G6: criterion 5 (missing-ball D guard)"

wait_guard; say "4 seed and copy agreement on SkillCorner (8a and 8b baselines; no competition match)"
step E -m tools.learned_spike.r3_eval agreement $RA $XA --final-npz "$FINAL" || fail "agreement"
if [ -n "${STOP_BEFORE_LABEL_FREE:-}" ]; then
  say "STOP_BEFORE_LABEL_FREE: everything on the external matches is done; the label-free step waits for the go"
  [ -z "$DRY" ] && decide --stop "before the label-free step (STOP_BEFORE_LABEL_FREE): G7 and G8 not run yet"
  exit 0
fi

wait_guard; say "5 the label-free step on $MATCH, once: c23, M, Mfold, Section 8; criteria 6, 7 and 7x = G7"
step E -m tools.learned_spike.r3_eval bar $RA $XA --final-npz "$FINAL" --feature-cache "$FEATURE_CACHE" $LF || fail "the label-free step failed"
step E -m tools.learned_spike.r3_eval r4bounds $RA $XA $BA --oof-ratio "$OOF_RATIO" || fail "r4bounds failed"
gate 6 7 7x || stop "G7: criterion 6, 7 or the four extra bounds (7x)"

wait_guard; say "6 criterion 8 = G8: numpy against torch (8 open matches, with the prior), learned tests, second run"
NETS=$(for s in 0 1 2 3; do printf "%s " "$RUN/final/net_s$s.pt"; done)
if [ -n "$DRY" ]; then
  echo "  [would run] $TORCH_PY -m tools.learned_spike.repro_check run --nets $NETS--weights $FINAL --weights-md5 <md5 of $FINAL> --costs $W/costs_label_free.json --structure --prior-bonus $PRIOR --engine-cache $EXT/views/raw/engine_data --engine-python $ENGINE_PY --work $W/repro_check --out $W/repro_check.json --tactical guarded --switch-off-record docs/plan/calibration/learned_switch_off_md5_c23.json --threads $TH"
  echo "  [would run] $ENGINE_PY -m pytest -q -p no:cacheprovider tests/test_learned*.py tests/test_repro_check.py > $W/tests.log"
else
  T -m tools.learned_spike.repro_check run --nets $NETS --weights "$FINAL" \
    --weights-md5 "$(cat "$OUT/learned_v4.npz.md5")" --costs "$W/costs_label_free.json" --structure \
    --prior-bonus "$PRIOR" --engine-cache "$EXT/views/raw/engine_data" --engine-python "$ENGINE_PY" \
    --work "$W/repro_check" --out "$W/repro_check.json" --tactical guarded \
    --switch-off-record docs/plan/calibration/learned_switch_off_md5_c23.json \
    --threads "$TH" ${GUARD_LAST:+--battery-log $GUARD_LAST} ${REHEARSAL:+${REPRO_GAMES:+--games $REPRO_GAMES}} 2>>"$OUT/engine_runs.log" || say "repro_check reported a failure (recorded)"
  E -m pytest -q -p no:cacheprovider tests/test_learned.py tests/test_learned_evaluate.py \
    tests/test_learned_r3_eval.py tests/test_learned_r4.py tests/test_learned_r4_final.py \
    tests/test_learned_prior.py tests/test_repro_check.py > "$W/tests.log" 2>&1 || true
  tail -1 "$W/tests.log" | tee -a "$OUT/r4_final.log"
fi
step E -m tools.learned_spike.r3_eval repro $RA $XA --final-npz "$FINAL" $LF \
  --repro-check "$W/repro_check.json" --tests-log "$W/tests.log" \
  ${FRESH_CLONE_LOG:+--fresh-clone-log $FRESH_CLONE_LOG} || fail "criterion 8 assembly failed"
g8=0; gate 8 || g8=$?
if [ "$g8" = 4 ] && [ -z "${FRESH_CLONE_LOG:-}" ]; then
  say "criterion 8 is not complete: part b (fresh clone) has not run"; PENDING8=1
elif [ "$g8" != 0 ]; then
  stop "G8: criterion 8 (reproducibility)"
fi

say "7 decide"
NOTE="${PENDING8:+criterion 8 part b (fresh clone) pending}"
[ -n "${STOPS:-}" ] && NOTE="REHEARSAL, the real run would have stopped at: $STOPS${NOTE:+; $NOTE}"
[ -n "$DRY" ] && { echo "  [would run] decide"; exit 0; }
if [ -n "$NOTE" ]; then decide --stop "$NOTE"; else decide; fi
say "done: $OUT/LEARNED_STAGE_RESULT_R4.md"

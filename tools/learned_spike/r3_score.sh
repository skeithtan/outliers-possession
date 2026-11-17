#!/bin/bash
# Round 3 scoring of a finished learned run, one pass (LEARNED_STAGE_PROTOCOL_R3.md, R3b or R3c).
#
# usage: tools/learned_spike/r3_score.sh RUN_DIR OUT_DIR
#   RUN_DIR  the r3_launch.sh output folder (folds/, final/, interim/, oof_p5/), read only
#   OUT_DIR  a new folder outside the repository: OUT_DIR/work (every engine output, the label-free
#            step's files), OUT_DIR/learned_v3.npz (the final weights the engine reads),
#            OUT_DIR/LEARNED_STAGE_RESULT_<VARIANT>.md and .json (the decision record)
#
# Environment (required): CA (build.py cache), MTAH_DATA_DIR (competition data folder): source
#   scratchpad/r3_out/scripts/env.sh. C23_DEV (c23 CSVs of the 8 development matches,
#   scratchpad/r3_out/c23_dev). Optional: EXT (c22.py cache; default $(dirname $CA)/ext),
#   TORCH_PY, ENGINE_PY, GUARD_LAST (battery guard .last file: wait for paused=0 before each step),
#   FRESH_CLONE_LOG (log of tools/repro_fresh_clone.sh, criterion 8 part b), EXPLANATIONS
#   (Section 8c JSON), NO_H=1 (skip the recorded H runs), SEED_FROM=DIR (a previous OUT_DIR/work:
#   its c23 runs are copied; they are keyed by engine source hash and config, so a copy is only
#   used when it is the same run).
# STOP_BEFORE_LABEL_FREE=1: stop after every external-match step (criteria 1 to 5, the views
#   tables and cost rules, 8a/8b baselines, export), before the competition match; re-run the
#   same command without it after the go (every step resumes from its cache).
# Rehearsal: REHEARSAL=1 AS_MATCH=<development match>: problems of the run are recorded, not
#   fatal, and the label-free step runs on AS_MATCH in place of the competition match.
# VARIANT=R3B (default) or R3C (R3_DEVIATION_AUG.md 13: R3b + the aug switch; the preflight checks
#   the switches of that variant, the result is LEARNED_STAGE_RESULT_R3C). Use a fresh OUT_DIR
#   per variant.
# PAIR_12A=H,A: run the label-free step and criteria 6 to 10 at the Section 12a pair chosen by
#   `r3_eval pair12a` (R3_DEVIATION_COSTS.md 12a; only after the rule's pair failed 8e), in
#   work/bar_12a/ beside the step at the rule's pair.
#
# Order and stops (R3 Section 4): preflight and switch-off identity; export; costs (rule B);
# development runs; criterion 3 (stop on fail); stress, criteria 4 and 5 (stop); the label-free
# step ONCE (bar), criteria 6 and 7 (stop); V2 and V3, criteria 9 and 10 (stop); criterion 8
# (stop); the recorded H runs; decide. On a stop, decide writes the result with the reason and the
# script exits 3: M does not qualify and c23 ships. Criteria 1 and 2 are recorded, never a stop.
# Costs: the development criteria use rule B as frozen (nested pairs). The label-free step, the
# criterion 8 checks and the configuration that would ship use the pair of the owner-approved
# rule (R3_DEVIATION_COSTS.md: best worst case over the as-provided and loose-touch views, from
# step 5b; COSTS_RULE=a falls back to the frozen pair), with LEARNED_STRUCTURE on
# (R3_DEVIATION_STRUCTURE.md); the frozen pair is recorded beside it.
# One engine process at a time, nice 15, one thread.
set -euo pipefail
[ $# -eq 2 ] || { sed -n 2,30p "$0"; exit 2; }
abs() { mkdir -p "$1" && (cd "$1" && pwd); }
RUN=$(cd "$1" && pwd); OUT=$(abs "$2"); W=$OUT/work; mkdir -p "$W"
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
case "$OUT/" in "$REPO"/*) echo "OUT_DIR must be outside the repository"; exit 2;; esac
: "${CA:?source scratchpad/r3_out/scripts/env.sh first}"; : "${MTAH_DATA_DIR:?}"; : "${C23_DEV:?}"
EXT=${EXT:-$(dirname "$CA")/ext}
TORCH_PY=${TORCH_PY:-$HOME/.venvs/barca-torch/bin/python}
ENGINE_PY=${ENGINE_PY:-$HOME/.venvs/barca/bin/python}
# THREADS (default 1) for torch and the tool processes; the engine pins its own BLAS threads to 1
# (engine/__init__.py). NICE (default 15; 0 for none).
TH=${THREADS:-1}
export OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH MKL_NUM_THREADS=$TH VECLIB_MAXIMUM_THREADS=$TH
E() { nice -n "${NICE:-15}" "$ENGINE_PY" "$@"; }
T() { nice -n "${NICE:-15}" "$TORCH_PY" "$@"; }
RA="--run $RUN --work $W"
VARIANT=${VARIANT:-R3B}
case "$VARIANT" in R3B|R3C) ;; *) echo "VARIANT must be R3B or R3C"; exit 2;; esac
RESULT="LEARNED_STAGE_RESULT_$VARIANT"
XA="--engine-cache $EXT/views/raw/engine_data --sc-engine-cache $EXT/views/raw/engine_data --ref-cache $EXT"
FINAL=$OUT/learned_v3.npz
LF=""; PF="--identity --variant $VARIANT"
if [ -n "${REHEARSAL:-}" ]; then PF="$PF --rehearsal"; LF="--as-match ${AS_MATCH:?REHEARSAL needs AS_MATCH}"; fi
if [ -n "${PAIR_12A:-}" ]; then LF="$LF --pair $PAIR_12A"; fi
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "$(stamp) $*" | tee -a "$OUT/r3_score.log"; }
wait_guard() { [ -z "${GUARD_LAST:-}" ] && return 0; while ! grep -q "paused=0" "$GUARD_LAST" 2>/dev/null; do say "battery guard paused, waiting"; sleep 60; done; }
decide() { E -m tools.learned_spike.r3_eval decide --work "$W" --out-dir "$OUT" --variant "$VARIANT" ${EXPLANATIONS:+--explanations $EXPLANATIONS} "$@" | tee -a "$OUT/r3_score.log"; }
# stop: a criterion failed where the protocol says to stop: record, decide, exit 3 (c23 ships).
# In a rehearsal the stop is recorded and the script goes on, so that every step runs once.
stop() {
  if [ -n "${REHEARSAL:-}" ]; then say "REHEARSAL: the real run would stop here: $1"; STOPS="${STOPS:+$STOPS; }$1"; return 0; fi
  say "STOP: $1"; decide --stop "$1"; say "M does not qualify: c23 ships (R3 Section 4)"; exit 3; }
# fail: a tool did not finish. Not a verdict: fix the cause and run the same command again.
fail() { say "TOOLING FAILURE: $1 (not a verdict; fix and re-run the same command, it resumes)"; exit 4; }
gate() { E -m tools.learned_spike.r3_eval gate --work "$W" ${EXPLANATIONS:+--explanations $EXPLANATIONS} --criteria "$@" | tee -a "$OUT/r3_score.log"; return "${PIPESTATUS[0]}"; }
# step CMD...: run one step, its stdout to the log, its stderr to engine_runs.log; the exit code
# is the command's (called after `||`, so set -e does not end the script on a failure).
step() { "$@" 2>>"$OUT/engine_runs.log" | tee -a "$OUT/r3_score.log"; return "${PIPESTATUS[0]}"; }

say "r3_score start: VARIANT=$VARIANT RUN=$RUN OUT=$OUT commit $(git rev-parse HEAD)${REHEARSAL:+ REHEARSAL on $AS_MATCH}${PAIR_12A:+ Section 12a pair $PAIR_12A}"
if [ -n "$(git status --porcelain --untracked-files=no -- engine games.py tools/learned_spike)" ]; then
  say "warning: engine/ or tools/learned_spike has uncommitted changes"
fi
if [ -n "${SEED_FROM:-}" ]; then
  E - "$SEED_FROM" "$W" <<'PY'
import json, os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
n = 0
for root, _, files in os.walk(os.path.join(src, "runs")):
    for name in files:
        if not name.endswith(".meta.json"):
            continue
        meta = json.load(open(os.path.join(root, name)))
        if meta.get("system") != "c23":
            continue
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        stem = name[:-len(".meta.json")]
        for f in os.listdir(root):
            if f.startswith(stem + "."):
                shutil.copy2(os.path.join(root, f), os.path.join(dst, rel, f))
        n += 1
print("seeded {} c23 runs".format(n))
PY
fi

wait_guard; say "0 preflight and switch-off identity"
if [ -n "${PAIR_12A:-}" ] && E - "$W/preflight.json" "$VARIANT" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
ok = p.get("pass") and (p.get("identity") or {}).get("pass") and p.get("variant", "R3B") == sys.argv[2]
sys.exit(0 if ok else 1)
PY
then
  # A Section 12a rerun: the preflight and the switch-off identity (which runs c23 on the
  # competition match) passed in the first pass; they are kept, not run again.
  say "PAIR_12A: preflight.json passed with the switch-off identity in the first pass; kept"
else
  step E -m tools.learned_spike.r3_eval preflight $RA $PF || stop "preflight failed (the run is not a finished $VARIANT run, or the switch-off identity failed)"
fi

wait_guard; say "1 export (torch venv): fold npz, 32-model npz, final $FINAL"
step T -m tools.learned_spike.evaluate export --stage "$RUN" --work "$W" --final-out "$FINAL" || fail "export failed"
md5 -q "$FINAL" > "$OUT/learned_v3.npz.md5" 2>/dev/null || md5sum "$FINAL" | cut -d' ' -f1 > "$OUT/learned_v3.npz.md5"
say "learned_v3.npz md5 $(cat "$OUT/learned_v3.npz.md5")"

wait_guard; say "2 costs (rule B, recomputed and compared with final/nested_costs.json)"
step E -m tools.learned_spike.r3_eval costs $RA --cache "$CA" --c23-dev "$C23_DEV" || fail "the nested costs of the run differ from their recomputation"

wait_guard; say "3 development runs: c23, M (nested), M (final), matched view"
step E -m tools.learned_spike.r3_eval run $RA $XA || fail "development runs incomplete"

wait_guard; say "4 scores; criterion 3 (1 and 2 recorded)"
step E -m tools.learned_spike.r3_eval score $RA $XA || fail "scores could not be computed"
gate 3 || stop "criterion 3 (as provided, provider-balanced 5 % bound above 0)"

wait_guard; say "5 stress and uniform30 views; criteria 4 and 5"
step E -m tools.learned_spike.r3_eval stress $RA $XA || fail "stress runs failed"
wait_guard; say "5b recorded only: three reference views x three inputs x the whole cost grid, cost rules a/b/c, 8e union"
step E -m tools.learned_spike.r3_views refs --work "$W" --cache "$CA" || say "views: refs failed (recorded only)"
step E -m tools.learned_spike.r3_views grid --work "$W" --run "$RUN" --cache "$CA" --p5 "$RUN/oof_p5" \
  --npz-dir "$W/npz" --c23-dev "$C23_DEV" --engine-cache "$EXT/views/raw/engine_data" --ref-cache "$EXT" \
  || say "views: grid failed (recorded only)"
gate 4 5 || stop "criterion 4 or 5 (stress guard, missing-ball D guard)"

wait_guard; say "5c seed and copy agreement on SkillCorner (8a and 8b baselines; no competition match)"
step E -m tools.learned_spike.r3_eval agreement $RA $XA --final-npz "$FINAL" || fail "agreement"
if [ -n "${STOP_BEFORE_LABEL_FREE:-}" ]; then
  say "STOP_BEFORE_LABEL_FREE: everything on the external matches is done; the label-free step waits for the go"
  decide --stop "before the label-free step (STOP_BEFORE_LABEL_FREE): criteria 6 to 10 not run yet"
  exit 0
fi

wait_guard; say "6 the label-free step (${AS_MATCH:-BAR-ATH}), once: c23, M, Mfold, Section 8; criteria 6 and 7"
step E -m tools.learned_spike.r3_eval bar $RA $XA --final-npz "$FINAL" --costs-rule "${COSTS_RULE:-b}" --feature-cache "$CA" $LF || fail "the label-free step failed"
if ! gate 6 7; then
  if [ -z "${PAIR_12A:-}" ]; then
    say "if criterion 7 failed on 8e only, Section 12a applies (R3_DEVIATION_COSTS.md 12a): run"
    say "  $ENGINE_PY -m tools.learned_spike.r3_eval pair12a $RA $XA ${AS_MATCH:+--as-match $AS_MATCH}"
    say "  then this command again with PAIR_12A=<the chosen pair> (no second network run is needed for the walk)"
  fi
  stop "criterion 6 or 7 (vetoes, Section 8 bounds)${PAIR_12A:+ at the Section 12a pair $PAIR_12A}"
fi

wait_guard; say "7 V2 and V3; criteria 9 and 10"
step E -m tools.learned_spike.r3_eval v23 $RA $XA --final-npz "$FINAL" --costs-rule "${COSTS_RULE:-b}" $LF || fail "V2/V3 runs failed"
gate 9 10 || stop "criterion 9 or 10 (V2, V3)"

wait_guard; say "8 criterion 8: numpy against torch (8 open matches), learned tests, second run"
NETS=$(for s in 0 1 2 3; do printf "%s " "$RUN/final/net_s$s.pt"; done)
T -m tools.learned_spike.repro_check run --nets $NETS --weights "$FINAL" \
  --weights-md5 "$(cat "$OUT/learned_v3.npz.md5")" --costs "$W/costs_label_free.json" --structure \
  --engine-cache "$EXT/views/raw/engine_data" --engine-python "$ENGINE_PY" --work "$W/repro_check" \
  --out "$W/repro_check.json" --tactical guarded \
  --switch-off-record docs/plan/calibration/learned_switch_off_md5_c23.json \
  --threads "$TH" ${GUARD_LAST:+--battery-log $GUARD_LAST} 2>>"$OUT/engine_runs.log" || say "repro_check reported a failure (recorded)"
E -m pytest -q -p no:cacheprovider tests/test_learned.py tests/test_learned_evaluate.py \
  tests/test_learned_r3.py tests/test_learned_r3_eval.py tests/test_learned_r3_views.py tests/test_repro_check.py > "$W/tests.log" 2>&1 || true
tail -1 "$W/tests.log" | tee -a "$OUT/r3_score.log"
step E -m tools.learned_spike.r3_eval repro $RA $XA --final-npz "$FINAL" --costs-rule "${COSTS_RULE:-b}" $LF \
  --repro-check "$W/repro_check.json" --tests-log "$W/tests.log" \
  ${FRESH_CLONE_LOG:+--fresh-clone-log $FRESH_CLONE_LOG} || fail "criterion 8 assembly failed"
g8=0; gate 8 || g8=$?
if [ "$g8" = 4 ] && [ -z "${FRESH_CLONE_LOG:-}" ]; then
  say "criterion 8 is not complete: part b (fresh clone) has not run; see the adoption runbook"
  PENDING8=1
elif [ "$g8" != 0 ]; then
  stop "criterion 8 (reproducibility gate)"
fi

if [ -z "${NO_H:-}" ]; then
  wait_guard; say "9 H (recorded, not eligible): raw runs with nested H costs, rescore"
  step E -m tools.learned_spike.r3_eval run $RA $XA --systems H H_fin --no-matched || say "H runs incomplete (recorded only)"
  E -m tools.learned_spike.r3_eval score $RA $XA 2>>"$OUT/engine_runs.log" > /dev/null || true
fi

say "10 decide"
NOTE="${PENDING8:+criterion 8 part b (fresh clone) pending}"
[ -n "${STOPS:-}" ] && NOTE="REHEARSAL, the real run would have stopped at: $STOPS${NOTE:+; $NOTE}"
if [ -n "$NOTE" ]; then decide --stop "$NOTE"; else decide; fi
say "done: $OUT/$RESULT.md"

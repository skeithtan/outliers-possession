# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# Round 5 chain: shared header of full_R5.sh, costs_R5.sh, gates_R5.sh, g3b_R5.sh, final_R5.sh. Source it.
# STATUS: Used by the rehearsed full_R5.sh; the other scripts' parts (chosen) not exercised.
# Required environment: RECIPE (AQCVX or AQCVXJ), RUNDIR (training output, e.g. $SP/r5_train), CODE (the run checkout).
# Optional: SCREEN_ROOT (left-out-provider networks; default $OUTROOT of r5_env.sh = $R5W/screen), JOBS (4),
#   THREADS (2), NICE (15), STEPS and EVAL_EVERY (rehearsal only), SEEDS (0 1 2 3; rehearsal only).
# Every script writes RUNDIR/logs/<name>.log and RUNDIR/<NAME>_DONE holding its exit code.
: "${RECIPE:?RECIPE=AQCVX or AQCVXJ}"; : "${RUNDIR:?RUNDIR=training output folder}"; : "${CODE:?CODE=run checkout}"
case "$RECIPE" in AQCVX|AQCVXJ) ;; *) echo "RECIPE must be AQCVX or AQCVXJ, not $RECIPE"; exit 2;; esac
source ${SCRATCH:?}/r5_work/scripts/r5_env.sh
export OUTROOT=${SCREEN_ROOT:-$OUTROOT}
export THREADS=${THREADS:-2} JOBS=${JOBS:-4} NICE=${NICE:-15}
export OMP_NUM_THREADS=$THREADS
SEEDS=${SEEDS:-"0 1 2 3"}
EPY=$HOME/.venvs/barca/bin/python
EXTD=$(dirname "$CA")/ext                 # reference md5 records of kfold/final, as round 4
mkdir -p "$RUNDIR/logs"
cd "$CODE" || { echo "CODE=$CODE is not a folder"; exit 2; }
STEP_ARGS=""; [ -n "${STEPS:-}" ] && STEP_ARGS="--steps $STEPS --eval-every ${EVAL_EVERY:-200}"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') [$CHAIN] $*" | tee -a "$RUNDIR/logs/$CHAIN.log"; }
finish() { rc=$?; echo "$rc" > "$RUNDIR/$(echo "$CHAIN" | tr a-z A-Z)_DONE"; say "exit $rc"; }
guard_wait() { while ! grep -q "paused=0" "$GLOG.last" 2>/dev/null; do sleep 60; done; }
# The training switches of RECIPE exactly as r5_screen_launch.sh used them (its last top-level
# "recipe RECIPE seeds ... switches: ...; commit" line in OUTROOT/RECIPE/logs/launch.log).
recipe_switches() {
  local line
  line=$(grep "^[0-9-]* [0-9:]* recipe $RECIPE seeds" "$OUTROOT/$RECIPE/logs/launch.log" 2>/dev/null | tail -1)
  [ -n "$line" ] || { echo "no launcher record for $RECIPE in $OUTROOT/$RECIPE/logs/launch.log" >&2; return 1; }
  line=${line#*switches: }; echo "${line%%; commit *}"
}
# The pair and bonus chosen by costs_R5.sh (RUNDIR/costs/chosen.env: PAIR_HA, PAIR_AD, PRIOR).
chosen() { [ -f "$RUNDIR/costs/chosen.env" ] || { echo "run costs_R5.sh first ($RUNDIR/costs/chosen.env)"; exit 2; }; source "$RUNDIR/costs/chosen.env"; }

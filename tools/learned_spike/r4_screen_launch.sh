#!/bin/bash
# Round 4 screening launcher (LEARNED_STAGE_PROTOCOL_R4.md Section 2): the three left-out-provider
# networks of one recipe, for one or more seeds, four training processes at a time.
#
# usage: tools/learned_spike/r4_screen_launch.sh RECIPE SEED [SEED ...]
#   RECIPE  letters of the round 4 steps, e.g. ABC (S1), AQC (S1'), then + D, then + E
#           A  --views raw                                  (raw plus the 16 aug copies only)
#           B  --sc-labels $PG                              (SkillCorner possession groups)
#           Q  --sc-alive-dead                              (arm B': SkillCorner alive vs dead
#              only). B and Q exclude each other. The SkillCorner-left-out network has no
#              SkillCorner match, so for a Q recipe it is the network of the same recipe with B:
#              it is trained (once, resumable) under that recipe's folder and linked here.
#           C  --cache $C_TRAIN --val-cache $C_VAL --aug-cache $AUG_C --dfl-still removed
#              (without C: --cache $CA --aug-cache $AUG_R3)
#           D  --ballfree-cache $BF                         (D': ball-free window in 15 % of crops)
#           E  --short-weight sqrt
#           F  the aug copies with ball smoothing and offset (step F): --aug-cache $AUG_F in place
#              of $AUG_C (needs C; same layout and dfl_c_train marker, plus a "ball" record)
#           always: --target raw --mix sc_quarter (round 3 recipe; the pools hold no PFF match)
# Environment (all required unless noted): TPY (torch python), CA (main cache), PG (possession-
#   group folder), C_TRAIN, C_VAL, AUG_C, AUG_R3, BF, OUTROOT (outputs go to OUTROOT/RECIPE/
#   <provider>/seed<k>/), GLOG (log of a RUNNING battery guard), MTAH_DATA_DIR.
#   Optional: THREADS (2), JOBS (4), STEPS and EVAL_EVERY (smoke or rehearsal only).
# Resumable: a network with DONE (same step count) is skipped. When all networks of the call are
# finished it writes OUTROOT/RECIPE/DONE_seeds_<seeds>; a failure writes FAILED_seeds_<seeds>.
# Long: start it with nohup, never under a command timeout. It stops nothing but its children.
set -uo pipefail
RECIPE=$1; shift; SEEDS="$*"
[ -n "$SEEDS" ] || { echo "usage: $0 RECIPE SEED [SEED ...]"; exit 2; }
THREADS=${THREADS:-2}; JOBS=${JOBS:-4}
cd "$(dirname "$0")/../.."
case "$RECIPE" in *[!ABCDEFQ]*) echo "recipe letters A to F and Q only: $RECIPE"; exit 2;; esac
case "$RECIPE" in *F*) case "$RECIPE" in *C*) ;; *) echo "F needs C: $RECIPE"; exit 2;; esac;; esac
case "$RECIPE" in *B*Q*|*Q*B*) echo "B and Q exclude each other: $RECIPE"; exit 2;; esac
SW="--target raw --mix sc_quarter"
case "$RECIPE" in *A*) SW="$SW --views raw";; esac
case "$RECIPE" in *B*) SW="$SW --sc-labels $PG";; esac
case "$RECIPE" in *Q*) SW="$SW --sc-alive-dead";; esac
PROVS=${ONLY_PROVIDERS:-"dfl metrica skillcorner"}
CANON=$(echo "$RECIPE" | tr Q B)
if [ "$CANON" != "$RECIPE" ]; then PROVS=$(echo $PROVS | tr ' ' '\n' | grep -v skillcorner | tr '\n' ' '); fi
case "$RECIPE" in
  *C*) AUGX=$AUG_C; case "$RECIPE" in *F*) AUGX=$AUG_F;; esac
       SW="$SW --cache $C_TRAIN --val-cache $C_VAL --aug-cache $AUGX --dfl-still removed";;
  *) SW="$SW --cache $CA --aug-cache $AUG_R3";;
esac
case "$RECIPE" in *D*) SW="$SW --ballfree-cache $BF";; esac
case "$RECIPE" in *E*) SW="$SW --short-weight sqrt";; esac
STEP_ARGS=""; [ -n "${STEPS:-}" ] && STEP_ARGS="--steps $STEPS --eval-every ${EVAL_EVERY:-200}"
OUT=$OUTROOT/$RECIPE; TAG=$(echo $SEEDS | tr ' ' '_')
mkdir -p "$OUT/logs"; rm -f "$OUT/FAILED_seeds_$TAG"
LOG=$OUT/logs/launch.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
gpid=$(cat "$GLOG.pid" 2>/dev/null || true)
if [ -z "$gpid" ] || ! kill -0 "$gpid" 2>/dev/null; then say "battery guard not running ($GLOG.pid)"; echo "battery guard not running"; exit 2; fi
say "recipe $RECIPE seeds $SEEDS threads $THREADS jobs $JOBS ${STEP_ARGS:-full steps}; switches: $SW; commit $(git rev-parse HEAD), dirty files $(git status --porcelain -- tools engine | wc -l | tr -d ' ')"
J=$OUT/logs/jobs_$TAG; : > "$J"
for s in $SEEDS; do for p in $PROVS; do echo "$p $s" >> "$J"; done; done
export TPY THREADS SW STEP_ARGS OUT LOG GLOG
job() {
  p=$1; s=$2
  while ! grep -q "paused=0" "$GLOG.last" 2>/dev/null; do sleep 60; done
  echo "$(date '+%Y-%m-%d %H:%M:%S') start $p seed $s" >> "$LOG"
  if $TPY -m tools.learned_spike.r4_screen train --provider "$p" --seed "$s" --out "$OUT" \
      --threads "$THREADS" $STEP_ARGS $SW > "$OUT/logs/train_${p}_s$s.log" 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') end $p seed $s: $(tail -1 "$OUT/logs/train_${p}_s$s.log")" >> "$LOG"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') FAILED $p seed $s (see logs/train_${p}_s$s.log)" >> "$LOG"
  fi
}
export -f job
[ -s "$J" ] && xargs -P "$JOBS" -L 1 bash -c 'job $0 $1' < "$J"
if [ "$CANON" != "$RECIPE" ] && [ -z "${ONLY_PROVIDERS:-}" ]; then
  say "skillcorner-left-out network: shared with recipe $CANON"
  ONLY_PROVIDERS=skillcorner bash "$0" "$CANON" $SEEDS || say "FAILED shared skillcorner ($CANON)"
  [ -e "$OUT/skillcorner" ] || ln -s "../$CANON/skillcorner" "$OUT/skillcorner"
fi
CHECK=$PROVS; [ "$CANON" != "$RECIPE" ] && [ -z "${ONLY_PROVIDERS:-}" ] && CHECK="$PROVS skillcorner"
for s in $SEEDS; do for p in $CHECK; do
  [ -f "$OUT/$p/seed$s/DONE" ] || { say "MISSING $p seed $s"; touch "$OUT/FAILED_seeds_$TAG"; exit 1; }
done; done
say "all networks present ($CHECK)"; touch "$OUT/DONE_seeds_$TAG${ONLY_PROVIDERS:+_only_$ONLY_PROVIDERS}"

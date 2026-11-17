#!/bin/bash
# Round 5 screening launcher (LEARNED_STAGE_PROTOCOL_R5.md Section 2): the three left-out-provider
# networks of one recipe, for one or more seeds, four training processes at a time. Round 4's
# launcher (r4_screen_launch.sh) with two more letters; its letters keep their meaning.
#
# usage: [ONLY_PROVIDERS="metrica dfl"] tools/learned_spike/r5_screen_launch.sh RECIPE SEED [SEED ...]
#   e.g.  ONLY_PROVIDERS="metrica dfl" r5_screen_launch.sh AQCVX 0 1 2 3
#         ONLY_PROVIDERS=skillcorner r5_screen_launch.sh AQCVX 0 1
#         ONLY_PROVIDERS=metrica r5_screen_launch.sh AQCVXJ 0 1 2 3
#         ONLY_PROVIDERS=skillcorner r5_screen_launch.sh AQCVXJ 0 1
#         ONLY_PROVIDERS=dfl r5_screen_launch.sh AQCVXJ 0 1 2 3   (links AQCVX's, trains nothing)
#   ONLY_PROVIDERS (default all three) and the seed list restrict a call; SC_ONLY=1 is the same
#   as ONLY_PROVIDERS=skillcorner.
#   RECIPE  round 4 letters A B Q C D E F (see r4_screen_launch.sh), plus
#           V  --feature-version stage-v3 with the v3 caches: --cache $C_TRAIN5 --val-cache
#              $C_VAL5 (needs C; the aug folder must hold stage-v3 features: X, or $AUG5)
#           X  the aug copies with switch X (augment version "x", R5_AUG.md): --aug-cache
#              $AUG5X in place of $AUG_C (needs C; its marker's feature_version must match:
#              stage-v3 with V, none without)
#           J  step J (R5_AUG.md): --cache $C_TRAIN5J --aug-cache $AUG5XJ in place of $C_TRAIN5
#              and $AUG5X (needs V and X, or S; markers dfl_input c_train_j)
#           S  round 6: --feature-version stage50-v2s (stage50-v2 with the team-swap fix) with
#              --cache $C_TRAIN2S --val-cache $C_VAL2S --aug-cache $AUG2S (round 4's copies,
#              augment version r3, at v2s); with J: $C_TRAIN2SJ and $AUG2SJ. Needs C; excludes V,
#              X, F.
#           T  round 7: as S with --feature-version stage50-v2t (attack direction fixed) and
#              $C_TRAIN2T / $C_VAL2T / $AUG2T (with J: $C_TRAIN2TJ, $AUG2TJ). Excludes S, V, X, F.
#              For AQCTJ the DFL-left-out networks are trained under AQCT (once) and linked.
#           L  round 6 arm A2: J keeping every still-ball run (input c_train_j100): --cache
#              $C_TRAIN2SJ100 --aug-cache $AUG2SJ100. Needs S, excludes J. Its DFL-left-out
#              networks are those of the recipe without L (AQCS), only linked, never trained.
#           Shared networks. The SkillCorner-left-out network has no SkillCorner match, so for a
#           Q recipe it is the network of the same recipe with B (trained once, resumable, under
#           that recipe's folder, and linked here), as in round 4. The DFL-left-out network has
#           no DFL match, so for a J recipe it is the network of the same recipe without J: it is
#           only LINKED (never trained from a J call; a missing one is reported) for an X recipe;
#           for an S recipe it is trained under the recipe without J (once) and linked, like Q/B.
#           always: --target raw --mix sc_quarter
# Environment: as r4_screen_launch.sh (TPY, CA, PG, C_TRAIN, C_VAL, AUG_C, AUG_R3, BF, OUTROOT,
#   GLOG, MTAH_DATA_DIR), plus for V: C_TRAIN5, C_VAL5 (and AUG5 if no X); for X: AUG5X; for J:
#   C_TRAIN5J, AUG5XJ; for S: C_TRAIN2S, C_VAL2S, AUG2S (with J: C_TRAIN2SJ, AUG2SJ).
#   Optional: THREADS (2), JOBS (4), STEPS and EVAL_EVERY (smoke only), SC_ONLY=1.
# Resumable: a network with DONE (same step count) is skipped. When all networks of the call are
# finished it writes OUTROOT/RECIPE/DONE_seeds_<seeds>_<providers>; a failure writes
# FAILED_seeds_<seeds>. Long: start it with nohup. It stops nothing but its children.
set -uo pipefail
RECIPE=$1; shift; SEEDS="$*"
[ -n "$SEEDS" ] || { echo "usage: $0 RECIPE SEED [SEED ...]"; exit 2; }
THREADS=${THREADS:-2}; JOBS=${JOBS:-4}
cd "$(dirname "$0")/../.."
case "$RECIPE" in *[!ABCDEFJLQSTVX]*) echo "recipe letters A to F, J, L, Q, S, T, V, X only: $RECIPE"; exit 2;; esac
case "$RECIPE" in *T*) case "$RECIPE" in *[SVXFL]*) echo "T excludes S, V, X, F and L: $RECIPE"; exit 2;; esac;; esac
case "$RECIPE" in *L*) case "$RECIPE" in *S*) ;; *) echo "L needs S: $RECIPE"; exit 2;; esac;; esac
case "$RECIPE" in *J*L*|*L*J*) echo "J and L exclude each other: $RECIPE"; exit 2;; esac
case "$RECIPE" in *J*) case "$RECIPE" in *V*X*|*X*V*|*S*|*T*) ;; *) echo "J needs V and X, or S, or T: $RECIPE"; exit 2;; esac;; esac
case "$RECIPE" in *S*) case "$RECIPE" in *[VXF]*) echo "S excludes V, X and F: $RECIPE"; exit 2;; esac;; esac
for L in F S T V X; do
  case "$RECIPE" in *$L*) case "$RECIPE" in *C*) ;; *) echo "$L needs C: $RECIPE"; exit 2;; esac;; esac
done
case "$RECIPE" in *F*X*|*X*F*) echo "F and X exclude each other: $RECIPE"; exit 2;; esac
case "$RECIPE" in *B*Q*|*Q*B*) echo "B and Q exclude each other: $RECIPE"; exit 2;; esac
SW="--target raw --mix sc_quarter"
case "$RECIPE" in *A*) SW="$SW --views raw";; esac
case "$RECIPE" in *B*) SW="$SW --sc-labels $PG";; esac
case "$RECIPE" in *Q*) SW="$SW --sc-alive-dead";; esac
FV=""
case "$RECIPE" in *V*) FV=stage-v3;; esac
case "$RECIPE" in *S*) FV=${S_FEATURE_VERSION:-stage50-v2s};; esac   # override: smoke only
case "$RECIPE" in *T*) FV=stage50-v2t;; esac
TRAINC=$C_TRAIN; VALC=$C_VAL; AUGX=${AUG_C:-}
case "$RECIPE" in *V*) TRAINC=${C_TRAIN5:?V needs C_TRAIN5}; VALC=${C_VAL5:?V needs C_VAL5}; AUGX=${AUG5:-};; esac
case "$RECIPE" in *F*) AUGX=$AUG_F;; esac
case "$RECIPE" in *X*) AUGX=$AUG5X;; esac
case "$RECIPE" in *S*) TRAINC=${C_TRAIN2S:?S needs C_TRAIN2S}; VALC=${C_VAL2S:?S needs C_VAL2S}; AUGX=${AUG2S:?S needs AUG2S};; esac
case "$RECIPE" in *X*J*|*J*X*) TRAINC=${C_TRAIN5J:?J needs C_TRAIN5J}; AUGX=${AUG5XJ:?J needs AUG5XJ};; esac
case "$RECIPE" in *S*J*|*J*S*) TRAINC=${C_TRAIN2SJ:?SJ needs C_TRAIN2SJ}; AUGX=${AUG2SJ:?SJ needs AUG2SJ};; esac
case "$RECIPE" in *T*) TRAINC=${C_TRAIN2T:?T needs C_TRAIN2T}; VALC=${C_VAL2T:?T needs C_VAL2T}; AUGX=${AUG2T:?T needs AUG2T};; esac
case "$RECIPE" in *T*J*|*J*T*) TRAINC=${C_TRAIN2TJ:?TJ needs C_TRAIN2TJ}; AUGX=${AUG2TJ:?TJ needs AUG2TJ};; esac
case "$RECIPE" in *L*) TRAINC=${C_TRAIN2SJ100:?L needs C_TRAIN2SJ100}; AUGX=${AUG2SJ100:?L needs AUG2SJ100};; esac
case "$RECIPE" in
  *C*) [ -n "$AUGX" ] || { echo "no aug folder for $RECIPE (V without X needs AUG5)"; exit 2; }
       SW="$SW --cache $TRAINC --val-cache $VALC --aug-cache $AUGX --dfl-still removed";;
  *) SW="$SW --cache $CA --aug-cache $AUG_R3";;
esac
case "$RECIPE" in *X*|*S*|*T*)
  AV=x; case "$RECIPE" in *S*|*T*) AV=r3;; esac
  WANT_IN=c_train; case "$RECIPE" in *J*) WANT_IN=c_train_j;; *L*) WANT_IN=c_train_j100;; esac
  got=$($TPY -c "import json,sys; m=json.load(open(sys.argv[1])); print(m.get('aug_version'), m.get('dfl_input'), m.get('feature_version') or '')" "$AUGX/dfl_still.json" 2>/dev/null) \
    || { echo "$AUGX: no marker dfl_still.json (r5_inputs mark --kind aug)"; exit 2; }
  [ "$got" = "$AV $WANT_IN $FV" ] || { echo "$AUGX marker '$got', recipe needs '$AV $WANT_IN $FV'"; exit 2; }
  case "$RECIPE" in *J*|*L*)
    gotc=$($TPY -c "import json,sys; print(json.load(open(sys.argv[1])).get('dfl_input'))" "$TRAINC/dfl_still.json" 2>/dev/null)
    [ "$gotc" = "$WANT_IN" ] || { echo "$TRAINC: marker dfl_input '$gotc', recipe needs $WANT_IN"; exit 2; };;
  esac;;
esac
[ -n "$FV" ] && SW="$SW --feature-version $FV"
case "$RECIPE" in *D*) SW="$SW --ballfree-cache $BF";; esac
case "$RECIPE" in *E*) SW="$SW --short-weight sqrt";; esac
STEP_ARGS=""; [ -n "${STEPS:-}" ] && STEP_ARGS="--steps $STEPS --eval-every ${EVAL_EVERY:-200}"
[ -n "${SC_ONLY:-}" ] && ONLY_PROVIDERS=skillcorner
PROVS=${ONLY_PROVIDERS:-"dfl metrica skillcorner"}
canon() {  # the recipe folder that holds the network of provider $1 for this recipe
  case "$1" in skillcorner) echo "$RECIPE" | tr Q B;; dfl) echo "$RECIPE" | tr -d JL;; *) echo "$RECIPE";; esac
}
DIRECT=""; SHARED=""; LINKED=""
for p in $PROVS; do
  c=$(canon $p)
  if [ "$c" = "$RECIPE" ]; then DIRECT="$DIRECT $p"
  elif [ "$p" = dfl ]; then case "$RECIPE" in *L*) LINKED="$LINKED $p";; *S*|*T*) SHARED="$SHARED $p";; *) LINKED="$LINKED $p";; esac
  else SHARED="$SHARED $p"; fi
done
OUT=$OUTROOT/$RECIPE; TAG=$(echo $SEEDS | tr ' ' '_')
mkdir -p "$OUT/logs"; rm -f "$OUT/FAILED_seeds_$TAG"
LOG=$OUT/logs/launch.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
gpid=$(cat "$GLOG.pid" 2>/dev/null || true)
if [ -z "$gpid" ] || ! kill -0 "$gpid" 2>/dev/null; then say "battery guard not running ($GLOG.pid)"; echo "battery guard not running"; exit 2; fi
say "recipe $RECIPE seeds $SEEDS providers $PROVS (train here:${DIRECT:- none}; shared:${SHARED:- none}; linked:${LINKED:- none}) threads $THREADS jobs $JOBS ${STEP_ARGS:-full steps}; switches: $SW; commit $(git rev-parse HEAD), dirty files $(git status --porcelain -- tools engine | wc -l | tr -d ' ')"
J=$OUT/logs/jobs_$TAG; : > "$J"
for s in $SEEDS; do for p in $DIRECT; do echo "$p $s" >> "$J"; done; done
export TPY THREADS SW STEP_ARGS OUT LOG GLOG
job() {
  p=$1; s=$2
  while ! grep -q "paused=0" "$GLOG.last" 2>/dev/null; do sleep 60; done
  echo "$(date '+%Y-%m-%d %H:%M:%S') start $p seed $s" >> "$LOG"
  if nice -n 15 $TPY -m tools.learned_spike.r4_screen train --provider "$p" --seed "$s" --out "$OUT" \
      --threads "$THREADS" $STEP_ARGS $SW > "$OUT/logs/train_${p}_s$s.log" 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') end $p seed $s: $(tail -1 "$OUT/logs/train_${p}_s$s.log")" >> "$LOG"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') FAILED $p seed $s (see logs/train_${p}_s$s.log)" >> "$LOG"
  fi
}
export -f job
[ -s "$J" ] && xargs -P "$JOBS" -L 1 bash -c 'job $0 $1' < "$J"
for p in $SHARED; do
  c=$(canon $p)
  say "$p-left-out network: shared with recipe $c (trained there)"
  SC_ONLY= ONLY_PROVIDERS=$p bash "$0" "$c" $SEEDS || say "FAILED shared $p ($c)"
  [ -e "$OUT/$p" ] || ln -s "../$c/$p" "$OUT/$p"
done
for p in $LINKED; do
  c=$(canon $p)
  say "$p-left-out network: the one of recipe $c (linked, not trained)"
  [ -e "$OUT/$p" ] || ln -s "../$c/$p" "$OUT/$p"
done
for s in $SEEDS; do for p in $PROVS; do
  [ -f "$OUT/$p/seed$s/DONE" ] || { say "MISSING $p seed $s"; touch "$OUT/FAILED_seeds_$TAG"; exit 1; }
done; done
SUFFIX=$(echo $PROVS | tr ' ' '_')
say "all networks present ($PROVS)"; touch "$OUT/DONE_seeds_${TAG}_$SUFFIX"

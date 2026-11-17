#!/usr/bin/env bash
# Fresh-clone reproduction test (PLAN.md 3.8, card P5-robustness-freeze, procedure step 3).
#
# The script clones this repository into a new directory, builds a venv from the pinned
# requirements.txt, runs `python -m engine.run` there, and compares the md5 of the CSV with the
# expected md5. It never reads or writes the cache/ of this checkout. The clone builds its own
# cache/. It never copies match data into the clone. It points MTAH_DATA_DIR at the data instead.

usage() {
    cat <<'EOF'
Usage: tools/repro_fresh_clone.sh EXPECTED_MD5 [GIT_REF]

Clone this repository, install requirements.txt in a new venv, run the engine on the match
data and compare the md5 of the CSV with EXPECTED_MD5.

Arguments:
  EXPECTED_MD5   md5 of the CSV that the candidate version must produce
  GIT_REF        branch, tag or commit to test (default: HEAD)

Environment:
  MTAH_DATA_DIR  directory that holds tactical/ (default: data/ of this checkout)
  TMP_ROOT       parent of the work directory (default: $TMPDIR, else /tmp)
  KEEP_TMP       set to 1 to keep the work directory after a PASS (kept after a FAIL)
  PIP_FIND_LINKS, PIP_NO_INDEX, PIP_INDEX_URL
                 read by pip, for an install without network access

Uncommitted changes in this checkout are not tested. Only the committed GIT_REF is cloned.

Exit status: 0 on PASS, 1 on FAIL, 2 on a usage or set-up error.
EOF
}

set -u
set -o pipefail

if [ "$#" -eq 0 ]; then
    usage >&2
    exit 2
fi
case "$1" in
    -h|--help) usage; exit 0 ;;
esac
if [ "$#" -gt 2 ]; then
    usage >&2
    exit 2
fi

EXPECTED_MD5="$1"
GIT_REF="${2:-HEAD}"
case "$EXPECTED_MD5" in
    *[!0-9a-f]*|"") echo "error: expected md5 must be 32 lower-case hex digits" >&2; exit 2 ;;
esac
if [ "${#EXPECTED_MD5}" -ne 32 ]; then
    echo "error: expected md5 must be 32 lower-case hex digits" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)" || exit 2
DATA_DIR="${MTAH_DATA_DIR:-$REPO_ROOT/data}"
TMP_ROOT="${TMP_ROOT:-${TMPDIR:-/tmp}}"

if [ ! -d "$DATA_DIR/tactical" ]; then
    echo "error: $DATA_DIR/tactical not found. Set MTAH_DATA_DIR to the data bundle." >&2
    exit 2
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

SHA="$(git -C "$REPO_ROOT" rev-parse --verify "${GIT_REF}^{commit}")" || {
    echo "error: $GIT_REF is not a commit in $REPO_ROOT" >&2
    exit 2
}
if [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]; then
    echo "warning: this checkout has uncommitted changes. The test uses commit $SHA only."
fi

mkdir -p "$TMP_ROOT"
WORK="$(mktemp -d "$TMP_ROOT/repro-fresh-clone.XXXXXX")" || exit 2
PASSED=0
cleanup() {
    if [ "$PASSED" -eq 1 ] && [ "${KEEP_TMP:-0}" != "1" ]; then
        rm -rf "$WORK"
    else
        echo "work directory kept: $WORK"
    fi
}
trap cleanup EXIT

# Print the seconds since the script started. Wall-clock time is allowed here. It is not allowed
# in the engine.
now() { date +%s; }
START="$(now)"

echo "ref        $GIT_REF ($SHA)"
echo "data dir   $DATA_DIR"
echo "work dir   $WORK"

step_start="$(now)"
git clone --quiet --no-checkout "$REPO_ROOT" "$WORK/clone" || exit 2
git -C "$WORK/clone" checkout --quiet --detach "$SHA" || exit 2
CLONE_SECONDS=$(( $(now) - step_start ))

# A fresh clone must not hold match data or a cache. Stop if it does.
if [ -d "$WORK/clone/cache" ]; then
    echo "error: the clone already has a cache/ directory" >&2
    exit 2
fi
if [ -n "$(find "$WORK/clone/data" -mindepth 1 ! -name README.md 2>/dev/null)" ]; then
    echo "error: the clone holds files in data/ besides README.md" >&2
    exit 2
fi

step_start="$(now)"
python3 -m venv "$WORK/venv" || exit 2
PY="$WORK/venv/bin/python"
"$PY" -m pip install --quiet --disable-pip-version-check -r "$WORK/clone/requirements.txt" \
    || { echo "error: pip could not install requirements.txt" >&2; exit 2; }
INSTALL_SECONDS=$(( $(now) - step_start ))
echo "python     $("$PY" -c 'import sys; print(sys.version.split()[0])')"
echo "packages   $("$PY" -m pip freeze | tr '\n' ' ')"

md5_of() {
    "$PY" -c 'import hashlib, sys; print(hashlib.md5(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
}

run_engine() {
    # Run in the clone so that the clone writes its own cache/.
    (cd "$WORK/clone" && MTAH_DATA_DIR="$DATA_DIR" "$PY" -m engine.run --out "$1") \
        > "$1.log" 2>&1
}

# Run 1 starts with no cache/. Run 2 reads the cache that run 1 wrote.
step_start="$(now)"
run_engine "$WORK/repro.csv" || { cat "$WORK/repro.csv.log" >&2; echo "FAIL  engine run 1 exited with an error"; exit 1; }
COLD_SECONDS=$(( $(now) - step_start ))
step_start="$(now)"
run_engine "$WORK/repro_warm.csv" || { cat "$WORK/repro_warm.csv.log" >&2; echo "FAIL  engine run 2 exited with an error"; exit 1; }
WARM_SECONDS=$(( $(now) - step_start ))

ACTUAL_MD5="$(md5_of "$WORK/repro.csv")"
WARM_MD5="$(md5_of "$WORK/repro_warm.csv")"
TOTAL_SECONDS=$(( $(now) - START ))

echo "clone      ${CLONE_SECONDS} s"
echo "install    ${INSTALL_SECONDS} s"
echo "run 1      ${COLD_SECONDS} s (no cache/, includes the parse)"
echo "run 2      ${WARM_SECONDS} s (cache/ present)"
echo "cache      build time = run 1 minus run 2 = $(( COLD_SECONDS - WARM_SECONDS )) s (noisy)"
echo "expected   $EXPECTED_MD5"
echo "actual     $ACTUAL_MD5"
echo "run 2 md5  $WARM_MD5"

if [ "$ACTUAL_MD5" = "$EXPECTED_MD5" ] && [ "$WARM_MD5" = "$ACTUAL_MD5" ]; then
    PASSED=1
    echo "PASS  md5 $ACTUAL_MD5 matches, total ${TOTAL_SECONDS} s"
    exit 0
fi
echo "FAIL  expected $EXPECTED_MD5, got $ACTUAL_MD5 (run 2: $WARM_MD5), total ${TOTAL_SECONDS} s"
exit 1

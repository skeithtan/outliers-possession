#!/bin/bash
# The round 3 sealed DFL judge (tools/learned_spike/r3_sealed.py), one engine process at a time.
#
# usage: UNSEAL_DFL=owner-approved-12nov tools/learned_spike/r3_sealed.sh --out-dir OUT \
#            --variant R3B|R3C --weights NPZ --weights-md5 MD5 --pair HA,AD --prereg FILE \
#            [--expect TEXT ...] [--result-json LEARNED_STAGE_RESULT_<variant>.json]
#        rehearsal: UNSEAL_DFL=rehearsal tools/learned_spike/r3_sealed.sh ... --rehearse-as G1,G2
# Environment: ENGINE_PY (default ~/.venvs/barca/bin/python), THREADS (default 1), NICE (default 5).
# The pre-registration (--prereg) must be committed before the real run; the tool writes it into
# OUT/SEALED_RESULT_<variant>.md and .json before it builds or runs anything.
set -euo pipefail
[ $# -ge 1 ] || { sed -n 2,10p "$0"; exit 2; }
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
ENGINE_PY=${ENGINE_PY:-$HOME/.venvs/barca/bin/python}
TH=${THREADS:-1}
export OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH MKL_NUM_THREADS=$TH VECLIB_MAXIMUM_THREADS=$TH
exec nice -n "${NICE:-5}" "$ENGINE_PY" -m tools.learned_spike.r3_sealed run "$@"

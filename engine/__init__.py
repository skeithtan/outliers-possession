"""Possession engine for More than a Hack 2027, Phase 1 (see docs/plan/PLAN.md, Section 3).

numpy + pandas + stdlib only. Nothing in this package reads the ground truth (team rule).
"""

import os as _os

# One BLAS and OpenMP thread, set before numpy loads: the summation order of the learned stage's
# matrix products must not depend on the machine's core count (LEARNED_STAGE_PROTOCOL §6).
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_name] = "1"

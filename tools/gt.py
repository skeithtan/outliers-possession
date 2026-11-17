"""The public ground truth -- the ONLY reader of evaluation/gt/ outside evaluation/.

Team hard rule: `engine/` never reads evaluation/gt/* and never imports this module. Only
`tools/` and `tests/` may call public_gt_codes() (PLAN.md 3.1; findings R-F14).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import games  # noqa: E402
from engine import data  # noqa: E402  (the match name and frame count only)
import possession_metrics as pm  # noqa: E402  (the scorer's own reader: H=0 A=1 D=2 X=3)

GT_DIR = os.path.join(ROOT, "evaluation", "gt")


def public_gt_path(game=None):
    """Path to the public 5-minute GT CSV shipped with the repository. None means games.GAME."""
    return os.path.join(GT_DIR, "{}_possession_gt_public.csv".format(data.resolve_game(game)))


def public_gt_codes(game=None):
    """Public GT as np.int8 (N,) codes, H=0 A=1 D=2 X=3 (evaluation/possession_metrics.LABELS).

    N is the tactical video length from games.py; row i is video frame i. Fails loudly if the
    file is not exactly frame 0..N-1 in order. X means "not graded" (outside the public window).
    """
    table = pm.read_label_csv(public_gt_path(game))
    n_frames = data.frame_count(game)
    problem = pm.validate_submission(table, n_frames)
    if problem:
        raise ValueError("{}: {}".format(public_gt_path(game), problem))
    codes = pm.label_codes(table)
    assert codes.dtype == np.int8
    return codes

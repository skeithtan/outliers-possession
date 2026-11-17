"""Shared helpers for tools/forensics.py and tools/sanity.py (P0-D). Not part of the submission.

Everything here works on a results CSV and the raw tracking arrays; nothing imports `engine/`
except the data layer (`engine.data`) and, guarded, optional modules that may not exist yet.
The kinematics are computed here with plain numpy on slices so the tools do not wait for
`engine/tracks.py`; when that module lands, `optional_module("tracks")` is the hook to swap in.

Conventions: video frame index everywhere (row i = ATD frame_id - 1), half-open [start, stop)
intervals, metres and seconds for physics. Labels are the scorer's codes H=0 A=1 D=2 X=3.
"""

import importlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import games  # noqa: E402
import possession_metrics as pm  # noqa: E402

H, A, D, X = 0, 1, 2, 3
LETTER = {H: "H", A: "A", D: "D", X: "X"}
REPORTS_DIR = os.path.join(ROOT, "evaluation", "reports")


def optional_module(name):
    """`engine.<name>` if it exists and imports, else None (the tools never require it)."""
    try:
        return importlib.import_module("engine." + name)
    except Exception:  # not written yet, or half-written: diagnostics must still run
        return None


def game_name(game=None):
    """The match name: `game`, or games.GAME when `game` is None (engine.data.resolve_game)."""
    from engine import data  # local: the data layer is the only engine module the tools need
    return data.resolve_game(game)


def fps(game=None):
    """Frame rate of the match as a float, from games.py. `game` None means games.GAME."""
    return float(games.GAMES[game_name(game)]["fps"])


def result_name(path):
    """'evaluation/results/baseline_smooth50.csv' -> 'baseline_smooth50'."""
    return os.path.splitext(os.path.basename(path))[0]


def read_results(path, game=None):
    """A submission CSV as int8 codes (H=0 A=1 D=2 X=3); raises ValueError if it is invalid.

    The CSV must have one row per tactical video frame of `game`. None means games.GAME.
    """
    from engine import data
    table = pm.read_label_csv(path)
    problem = pm.validate_submission(table, data.frame_count(game))
    if problem:
        raise ValueError("{}: {}".format(path, problem))
    return pm.label_codes(table)


def runs_from_blocks(blocks):
    """[(half, first, last_inclusive)] graded blocks -> half-open [(start, stop)] runs."""
    return [(int(first), int(last) + 1) for _, first, last in blocks]


def segment_stats(lengths_frames, fps_value=None):
    """min / median / p90 / max / mean of segment lengths, in seconds (None when empty).

    `fps_value` None means the frame rate of games.GAME.
    """
    if not len(lengths_frames):
        return {"n": 0, "min_s": None, "median_s": None, "p90_s": None, "max_s": None,
                "mean_s": None}
    seconds = np.asarray(lengths_frames, dtype=np.float64) / (fps_value or fps())
    return {"n": int(len(seconds)), "min_s": float(seconds.min()),
            "median_s": float(np.median(seconds)), "p90_s": float(np.percentile(seconds, 90)),
            "max_s": float(seconds.max()), "mean_s": float(seconds.mean())}


def kinematics(tactical, lo, hi, pitch, margin=4):
    """Per-frame ball and player features on video frames [lo, hi) as float64 arrays.

    Returns a dict of arrays of length hi - lo (NaN where undefined), plus 'frames':
      dH, dA   nearest home / away player distance to the ball (m)
      v        ball speed (m/s) from the one-frame position difference (NaN if either frame lacks
               the ball)
      dv_3     |v[t+3] - v[t]| (m/s over 3 frames, PLAN 3.2's unit convention)
      meanv    mean player speed (m/s) over the tracked slots
      ball_missing  bool
    `margin` extra frames are read each side so v and dv_3 are defined at the edges. The frame
    rate is tactical["fps"], else the frame rate of games.GAME.
    """
    length, width = pitch
    n = tactical["bx"].shape[0]
    a, b = max(0, lo - margin), min(n, hi + margin)
    bx = tactical["bx"][a:b].astype(np.float64) * length
    by = tactical["by"][a:b].astype(np.float64) * width
    px = tactical["px"][a:b].astype(np.float64) * length
    py = tactical["py"][a:b].astype(np.float64) * width
    team = tactical["team_of_slot"]

    speed = np.full(b - a, np.nan)
    speed[1:] = np.hypot(np.diff(bx), np.diff(by)) * float(tactical.get("fps") or fps())
    dv3 = np.full(b - a, np.nan)
    dv3[:-3] = np.abs(speed[3:] - speed[:-3])
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    out = {}
    with np.errstate(all="ignore"):
        for name, side in (("dH", 0), ("dA", 1)):
            near = np.where((team[None, :] == side) & np.isfinite(dist), dist, np.inf).min(axis=1)
            out[name] = np.where(np.isinf(near), np.nan, near)
        out["meanv"] = np.nanmean(tactical["ps"][a:b].astype(np.float64), axis=1)
    out["v"], out["dv_3"] = speed, dv3
    out["ball_missing"] = np.isnan(bx)
    cut = slice(lo - a, hi - a)
    result = {key: value[cut] for key, value in out.items()}
    result["frames"] = np.arange(lo, hi)
    return result


def tag_team_codes(tags, n_frames, phase_codes, runs=None):
    """Per-frame Smart Tagging team prior as int8: -1 none, 0 H, 1 A, 2 both teams tagged.

    Only rows whose `code` is in `phase_codes` (compared upper-case) and which carry a team are
    painted. With `runs`, painting is clipped to those half-open spans (the graded frames).
    """
    wanted = {code.upper() for code in phase_codes}
    paint = np.full(n_frames, -1, dtype=np.int8)
    for row in tags.itertuples(index=False):
        if row.code.upper() not in wanted or row.team not in ("H", "A"):
            continue
        side = 0 if row.team == "H" else 1
        start, stop = max(0, int(row.start_frame)), min(n_frames, int(row.end_frame))
        if stop <= start:
            continue
        window = paint[start:stop]
        window[window == -1] = side
        window[(window != side) & (window != -1)] = 2
    if runs is not None:
        keep = np.zeros(n_frames, dtype=bool)
        for start, stop in runs:
            keep[start:stop] = True
        paint[~keep] = -1
    return paint

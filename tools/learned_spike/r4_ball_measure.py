"""How smoothed and how displaced is the ball of each external provider (R4_AUG_BALL.md). Not the engine.

    python -m tools.learned_spike.r4_ball_measure --engine-cache DIR --out OUT.json [--jobs 4]

External matches only (folds.ALL: 3 Metrica, 5 DFL, 20 SkillCorner); build.check refuses the
competition match and the sealed matches. Prints and stores summary statistics only.

Statistics, on the engine's cleaned ball (engine.tracks.clean_ball), alive mask not applied:

- Kick sharpness `ratio` (the statistic of tools/analysis/atd_kernel.kick_stats and of
  ATD_VIEW.md Section 3.2): for each run of frames with dv_9 > 3 m/s (speed rise over 9 frames),
  the largest dv_3 in the run (widened by 9 frames each side) over the largest dv_9 in the run;
  the median over the runs. A raw 25 fps kick reads about 1.0; a Gaussian-smoothed ball reads
  lower (about 0.34 at sigma 9.5).
- Touch displacement `touch_d`: for each such run [a, b), the smallest distance from the ball to
  the nearest visible player over frames a - 10 .. b - 1; the median over the runs (m). `near`:
  the share of runs where that distance is at most 1 m.

Fits (Metrica and DFL only, the matches that the component changes):

- ratio(sigma) and touch_d(sigma) on SIGMAS: the ball smoothed with augment.smooth_runs inside
  each visible run cut at the graded block edges.
- touch_d(sigma, A) on a grid: smoothed, then moved by augment.slow_offset with vector RMS A
  metres (time scale 2 s, fixed seed).
- `resampled`: the ball sampled at 10 Hz and linearly interpolated back to 25 Hz, the path of
  tools/skillcorner_bundle.py, to separate the 10 Hz effect from real smoothing.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.learned_spike import augment, folds  # noqa: E402

SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
GRID_SIGMAS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
GRID_A = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
TAU_FIT_S = 2.0
LEAD = 10


def _runs(mask):
    padded = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return edges[0::2], edges[1::2]


def ball_stats(tac, pitch, fps=25):
    """{ratio, touch_d, near, runs, speed_p99, dv9_p50} of one tactical dict."""
    from engine import tracks
    from engine.config import Config
    ball = tracks.clean_ball(tac["bx"], tac["by"], pitch, Config.default(), fps=fps)
    speed = ball["speed"]
    dv3 = tracks.velocity_step(speed, 3, ball["run_id"])
    dv9 = tracks.velocity_step(speed, 9, ball["run_id"])
    px = np.asarray(tac["px"], dtype=np.float64) * pitch[0]
    py = np.asarray(tac["py"], dtype=np.float64) * pitch[1]
    with np.errstate(invalid="ignore"):
        dist = np.hypot(px - ball["x"][:, None], py - ball["y"][:, None])
    dist = np.where(np.isfinite(dist), dist, np.inf).min(axis=1)
    with np.errstate(invalid="ignore"):
        rise = dv9 > 3.0
    ratio, touch, peak9 = [], [], []
    for a, b in zip(*_runs(rise)):
        seg3 = dv3[max(a - 9, 0):b + 9]
        p9 = np.nanmax(dv9[a:b])
        peak9.append(p9)
        if np.isfinite(seg3).any():
            ratio.append(np.nanmax(seg3) / p9)
        d = dist[max(a - LEAD, 0):b].min()
        if np.isfinite(d):
            touch.append(d)
    touch = np.asarray(touch)
    vis = ball["visible"]
    return {"ratio": float(np.median(ratio)) if ratio else None,
            "touch_d": float(np.median(touch)) if touch.size else None,
            "near": float((touch <= 1.0).mean()) if touch.size else None,
            "runs": len(ratio),
            "speed_p99": float(np.nanquantile(speed[vis], 0.99)),
            "dv9_p50": float(np.median(peak9)) if peak9 else None}


def resample_10hz(v, fps=25, hz=10):
    """A (N,) series sampled at `hz` and linearly interpolated back to fps; NaN if a sample is."""
    n = len(v)
    t10 = np.arange(0, (n - 1) / fps + 1e-9, 1.0 / hz) * fps      # positions in 25 fps frames
    i0 = np.floor(t10).astype(int)
    w = t10 - i0
    i1 = np.minimum(i0 + 1, n - 1)
    s = (1 - w) * v[i0] + w * v[i1]                                 # NaN if either is NaN
    t25 = np.arange(n) * hz / fps                                    # positions in 10 Hz samples
    j0 = np.minimum(np.floor(t25).astype(int), s.size - 1)
    j1 = np.minimum(j0 + 1, s.size - 1)
    u = t25 - j0
    return (1 - u) * s[j0] + u * s[j1]


def _smoothed(tac, blocks, pitch, sigma, rms=0.0, seed=0):
    rates = {"sigma": sigma, "offset_m": rms, "tau_s": TAU_FIT_S}
    bx, by, _ = augment.smooth_offset_ball(tac["bx"], tac["by"], blocks, pitch, rates,
                                           np.random.RandomState(seed))
    return dict(tac, bx=bx, by=by)


def measure_game(args):
    game, engine_cache = args
    from engine import data
    from tools.learned_spike import build
    build.check(game)
    if game not in folds.ALL:
        raise SystemExit(game + ": external matches only")
    tac = build.load_view(game, "raw", engine_cache)
    pitch, blocks = data.pitch_m(game), data.graded_blocks(game)
    out = {"game": game, "raw": ball_stats(tac, pitch)}
    if game.startswith("SC-"):
        return out
    out["sigma"] = {str(s): ball_stats(_smoothed(tac, blocks, pitch, s), pitch) for s in SIGMAS}
    out["grid"] = {"{}|{}".format(s, a): ball_stats(_smoothed(tac, blocks, pitch, s, a), pitch)
                   ["touch_d"] for s in GRID_SIGMAS for a in GRID_A}
    rs = dict(tac, bx=resample_10hz(np.asarray(tac["bx"], np.float64)).astype(np.float32),
              by=resample_10hz(np.asarray(tac["by"], np.float64)).astype(np.float32))
    out["resampled"] = {str(s): ball_stats(_smoothed(rs, blocks, pitch, s), pitch)
                        for s in SIGMAS[:9]}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--engine-cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--games", nargs="+", default=list(folds.ALL))
    a = p.parse_args(argv)
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(measure_game, [(g, a.engine_cache) for g in a.games]))
    with open(a.out, "w") as h:
        json.dump({r["game"]: r for r in res}, h, indent=1, sort_keys=True)
    for r in res:
        print(r["game"], json.dumps(r["raw"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

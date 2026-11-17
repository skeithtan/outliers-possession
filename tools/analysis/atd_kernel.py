"""Smoothing-kernel and kicked-ball measurements of a tracking bundle (P9-A4). Not the engine.

    python tools/analysis/atd_kernel.py --game BAR-ATH
    python tools/analysis/atd_kernel.py --game DFL-J03WMX --data-dir ~/mtah2027/refs \
        --cache-dir ~/mtah2027/cache --players

The kernel measurement repeats the method of RESEARCH-2 M13 on one track set:

1. Convert the positions to metres. Cut each track into runs. A run ends at a missing frame,
   at a step longer than `max_step` metres in one frame (the PLAN 1.1 teleport rule for the
   ball, 3 m), and at the edge of a graded block (engine.data.graded_blocks()).
2. The acceleration magnitude |a| is the second difference of the position times fps squared.
3. A strong isolated peak is a frame t whose |a| is the largest over [t - W, t + W], with the
   whole window inside one run, and whose window holds no |a| of at least half the peak outside
   [t - CORE, t + CORE]. W is 50 frames and CORE is 25 frames.
4. The `n_peaks` strongest such peaks are kept (46, the M13 count, for the ball). The profile
   of a peak is |a| over the window divided by the peak value. The tool takes the median
   profile over the peaks.
5. HWHM is the lag, in frames and linearly interpolated, at which the median profile first falls
   under one half, averaged over the two sides. A Gaussian kernel of width sigma smooths a
   velocity step into an |a| bump of HWHM = sigma * sqrt(2 ln 2), so sigma_hwhm = HWHM / 1.1774.
   The tool also prints the HWHM above the baseline (the median profile over CORE < |lag| <= W)
   and the cumulative 25 / 50 / 75 % lags of the median profile (M13 gives -6 / +1 / +8).

For the players the tool pools the runs of every slot, with `max_step` 1 m per frame, and keeps
the `n_peaks` strongest peaks over all slots.

The kicked-ball statistics use engine.tracks.clean_ball, the cleaning of the engine, on the ball
of the bundle: the ball speed quantiles on visible frames, the rates per visible-ball minute of
the runs with dv_3 > 3 m/s, dv_3 > 1.5 m/s with a player within 1.5 m (Appendix B) and
dv_9 > 3 m/s (PLAN 3.4 DV_RESTART), the quantiles of the largest dv_9 of each dv_9 > 3 run, and
the median ratio of the largest dv_3 to the largest dv_9 inside those runs. A kick on a raw ball
has a ratio near 1. A kick on a ball smoothed with sigma 9.5 has a ratio near 0.34. It also
counts the C8 impulse maxima (engine.events.impulse, |J| >= DV_CONTACT, local maxima over
+-CONTACT_HALF_WINDOW) per visible-ball minute, all of them and those with a player within
R_TOUCH over the window. The alive mask of the engine is not applied.

Prints summary statistics only, never a tracking row. Reads no ground truth. numpy + pandas.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine import data, events, tracks  # noqa: E402
from engine.config import Config  # noqa: E402

WINDOW = 50          # frames each side of a peak (2 s at 25 fps)
CORE = 25            # frames each side that the isolation test leaves out
BALL_MAX_STEP = 3.0  # m per frame, the PLAN 1.1 teleport rule
PLAYER_MAX_STEP = 1.0  # m per frame (25 m/s), a slot swap or a jump of the tracker
BALL_PEAKS = 46      # RESEARCH-2 M13
PLAYER_PEAKS = 400
GAUSS_HWHM = math.sqrt(2.0 * math.log(2.0))   # 1.1774


def run_breaks(n_frames, blocks):
    """bool (N,): True at the first frame of each graded block and the frame after its end."""
    breaks = np.zeros(n_frames + 1, dtype=bool)
    for _, first, last in blocks:
        breaks[first] = True
        breaks[last + 1] = True
    return breaks[:n_frames]


def run_labels(x, y, max_step, breaks):
    """int64 run label per frame (-1 where missing), for one track in metres."""
    visible = np.isfinite(x) & np.isfinite(y)
    step = np.full(x.shape, np.inf)
    step[1:] = np.hypot(np.diff(x), np.diff(y))
    new = visible & (~np.roll(visible, 1) | breaks | ~(step <= max_step))
    new[0] = visible[0]
    label = np.cumsum(new) - 1
    return np.where(visible, label, -1)


def accel_magnitude(x, y, label, fps):
    """|a| in m/s^2 by second difference, NaN unless t - 1, t, t + 1 share one run."""
    acc = np.full(x.shape, np.nan)
    same = (label[1:-1] >= 0) & (label[:-2] == label[1:-1]) & (label[2:] == label[1:-1])
    ax = (x[2:] - 2.0 * x[1:-1] + x[:-2]) * fps * fps
    ay = (y[2:] - 2.0 * y[1:-1] + y[:-2]) * fps * fps
    acc[1:-1] = np.where(same, np.hypot(ax, ay), np.nan)
    return acc


def _candidates(acc):
    """Frames whose |a| is a local maximum over +-1 frame (the peaks can only be among them)."""
    finite = np.where(np.isfinite(acc), acc, -np.inf)
    return np.flatnonzero((finite[1:-1] >= finite[:-2]) & (finite[1:-1] >= finite[2:])
                          & np.isfinite(acc[1:-1])) + 1


def is_isolated_peak(acc, label, t):
    """True if frame t passes method step 3 on one |a| series."""
    lo, hi = t - WINDOW, t + WINDOW
    if lo < 0 or hi >= acc.shape[0]:
        return False
    if label[lo] != label[t] or label[hi] != label[t]:
        return False
    win = acc[lo:hi + 1]
    peak = acc[t]
    if not np.all(np.isfinite(win)) or peak <= 0 or win.max() > peak:
        return False
    outer = np.concatenate([win[:WINDOW - CORE], win[WINDOW + CORE + 1:]])
    return bool(outer.max() < 0.5 * peak)


def profile_stats(profiles):
    """HWHM, sigma and cumulative lags of the median of the normalised profiles."""
    lags = np.arange(-WINDOW, WINDOW + 1)
    med = np.median(np.asarray(profiles), axis=0)

    def half_lag(level):
        sides = []
        for direction in (1, -1):
            prev_lag, prev_val = 0, med[WINDOW]
            found = None
            for k in range(1, WINDOW + 1):
                val = med[WINDOW + direction * k]
                if val < level:
                    found = prev_lag + (prev_val - level) / (prev_val - val)
                    break
                prev_lag, prev_val = k, val
            sides.append(found if found is not None else float("nan"))
        return sides

    left_right = half_lag(0.5)
    hwhm = float(np.mean(left_right))
    outer = np.abs(lags) > CORE
    base = float(np.median(med[outer]))
    hwhm_base = float(np.mean(half_lag(base + 0.5 * (1.0 - base))))
    above = np.clip(med - base, 0.0, None) * (np.abs(lags) <= CORE)
    cum = np.cumsum(above) / above.sum()
    quart = [int(lags[np.searchsorted(cum, q)]) for q in (0.25, 0.5, 0.75)]
    return {"n_peaks": len(profiles), "hwhm": round(hwhm, 2),
            "hwhm_left_right": [round(v, 2) for v in left_right],
            "sigma_hwhm": round(hwhm / GAUSS_HWHM, 2), "baseline": round(base, 3),
            "hwhm_above_baseline": round(hwhm_base, 2),
            "sigma_above_baseline": round(hwhm_base / GAUSS_HWHM, 2),
            "cum_25_50_75": quart}


def kernel_of_tracks(xs, ys, breaks, fps, max_step, n_peaks):
    """Kernel statistics over the columns of (N, S) metre arrays (method steps 1 to 5).

    The candidates of every column are tested in order of decreasing |a| (ties by column, then
    frame) until `n_peaks` pass, so the result is the n_peaks strongest isolated peaks.
    """
    series, labels, values, cols, frames = [], [], [], [], []
    for s in range(xs.shape[1]):
        x, y = xs[:, s], ys[:, s]
        label = run_labels(x, y, max_step, breaks)
        acc = accel_magnitude(x, y, label, fps)
        series.append(acc)
        labels.append(label)
        cand = _candidates(acc)
        values.append(acc[cand])
        cols.append(np.full(cand.shape, s))
        frames.append(cand)
    values, cols, frames = (np.concatenate(v) for v in (values, cols, frames))
    order = np.lexsort((frames, cols, -values))
    chosen, passed = [], 0
    for i in order:
        s, t = int(cols[i]), int(frames[i])
        if is_isolated_peak(series[s], labels[s], t):
            passed += 1
            if len(chosen) < n_peaks:
                chosen.append((float(values[i]), s, t))
    if not chosen:
        return {"n_peaks": 0}
    profiles = [series[s][t - WINDOW:t + WINDOW + 1] / peak for peak, s, t in chosen]
    out = profile_stats(profiles)
    out["n_isolated"] = passed
    out["peak_a_median"] = round(float(np.median([c[0] for c in chosen])), 1)
    out["peak_a_min"] = round(float(min(c[0] for c in chosen)), 1)
    return out


def _runs_of(mask):
    """(starts, stops) of the True runs of a bool array."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return edges[0::2], edges[1::2]


def kick_stats(tactical, pitch, fps):
    """Kicked-ball statistics on the engine's cleaned ball (see the module docstring)."""
    cfg = Config.default()
    ball = tracks.clean_ball(tactical["bx"], tactical["by"], pitch, cfg, fps=fps)
    speed = ball["speed"]
    visible = ball["visible"]
    minutes = visible.sum() / fps / 60.0
    dv3 = tracks.velocity_step(speed, 3, ball["run_id"])
    dv9 = tracks.velocity_step(speed, 9, ball["run_id"])
    length, width = pitch
    px = tactical["px"].astype(np.float64) * length
    py = tactical["py"].astype(np.float64) * width
    with np.errstate(invalid="ignore"):
        dist = np.hypot(px - ball["x"][:, None], py - ball["y"][:, None])
        dmin = np.where(np.isfinite(dist).any(axis=1), np.nanmin(
            np.where(np.isfinite(dist), dist, np.inf), axis=1), np.nan)
        near = dmin <= 1.5
        up3 = np.abs(dv3) > 3.0
        up15 = (np.abs(dv3) > 1.5) & near
        rise9 = dv9 > 3.0
    starts9, stops9 = _runs_of(rise9)
    peak9 = np.array([np.nanmax(dv9[a:b]) for a, b in zip(starts9, stops9)])
    ratio = []
    for a, b in zip(starts9, stops9):
        seg3 = dv3[max(a - 9, 0):b + 9]
        if np.isfinite(seg3).any():
            ratio.append(np.nanmax(seg3) / np.nanmax(dv9[a:b]))
    size = events.impulse(ball, cfg.CONTACT_HALF_WINDOW)
    with np.errstate(invalid="ignore"):
        strong = np.flatnonzero(np.isfinite(size) & (size >= cfg.DV_CONTACT))
    maxima = events.window_maxima(size, cfg.CONTACT_HALF_WINDOW, strong)
    h = int(cfg.CONTACT_HALF_WINDOW)
    near_any = np.where(np.isfinite(dmin), dmin, np.inf)
    attributed = sum(1 for c in maxima
                     if near_any[max(0, c - h):c + h + 1].min() <= cfg.R_TOUCH)
    q = [0.5, 0.9, 0.99]
    return {
        "bump_maxima_per_vmin": round(len(maxima) / minutes, 2),
        "bump_attributed_per_vmin": round(attributed / minutes, 2),
        "impulse_p50_p90_at_maxima": ([round(float(v), 2) for v in
                                       np.quantile(size[maxima], [0.5, 0.9])]
                                      if len(maxima) else None),
        "visible_minutes": round(float(minutes), 1),
        "speed_p50_p90_p99": [round(float(v), 2) for v in np.nanquantile(speed[visible], q)],
        "dv3_gt3_runs_per_vmin": round(len(_runs_of(up3)[0]) / minutes, 2),
        "dv3_gt1p5_near_runs_per_vmin": round(len(_runs_of(up15)[0]) / minutes, 2),
        "dv9_gt3_runs_per_vmin": round(len(starts9) / minutes, 2),
        "dv9_peak_p50_p90": ([round(float(v), 2) for v in np.quantile(peak9, [0.5, 0.9])]
                             if peak9.size else None),
        "dv3_over_dv9_median": round(float(np.median(ratio)), 3) if ratio else None,
        "teleports": int(ball["teleport"].sum()),
    }


def measure(game, data_dir=None, cache_dir=None, players=False):
    """All statistics of one bundle as a dict."""
    tactical = data.load_tactical(game, data_dir=data_dir, cache_dir=cache_dir)
    pitch = data.pitch_m(game)
    fps = float(data.fps(game))
    breaks = run_breaks(tactical["n_frames"], data.graded_blocks(game))
    length, width = pitch
    bx = tactical["bx"].astype(np.float64)[:, None] * length
    by = tactical["by"].astype(np.float64)[:, None] * width
    out = {"game": game, "ball_kernel": kernel_of_tracks(bx, by, breaks, fps, BALL_MAX_STEP,
                                                         BALL_PEAKS),
           "kick": kick_stats(tactical, pitch, fps)}
    if players:
        px = tactical["px"].astype(np.float64) * length
        py = tactical["py"].astype(np.float64) * width
        out["player_kernel"] = kernel_of_tracks(px, py, breaks, fps, PLAYER_MAX_STEP,
                                                PLAYER_PEAKS)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--players", action="store_true")
    args = parser.parse_args(argv)
    result = measure(args.game, args.data_dir and os.path.expanduser(args.data_dir),
                     args.cache_dir and os.path.expanduser(args.cache_dir), args.players)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

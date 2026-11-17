"""Label-free GHOST_NOBODY_M: the valley of the nearest-player distance at a stationary ball.

    MTAH_DATA_DIR=~/mtah2027/data python tools/analysis/ghost_valley.py [--game BAR-ATH]

The rule is pre-registered in docs/plan/calibration/grids.json (bucket b): "valley of the
nearest-player distance histogram over stationary-ball runs >= STATIONARY_S on the full BAR-ATH
match, 0.5 m bins, no labels". This script fixes every free choice of that rule before it reads
the match:

- Frames. The frames of deadball.stationary_mask(): the cleaned ball is visible for a whole
  window of STATIONARY_S and moves less than STATIONARY_MAX_DISP_M in it. Only frames inside a
  graded block (engine.data.graded_blocks) count, so the pre-match and half-time frames do not.
  A frame counts once. A frame without a tracked player (dmin not finite) is left out.
- Value. feats["dmin"] of engine.tracks.prepare(), the distance from the ball to the nearest
  player of either team, in metres.
- Histogram. Bins of 0.5 m from 0 to 30 m: bin i holds dmin in [0.5 i, 0.5 (i + 1)). A value at
  30 m or more goes to an overflow count, which takes no part in the valley search.
- Smoothing. A centred 3-bin moving average. At the two ends the mean is over the bins that
  exist (2 bins).
- Mode. The first local maximum of the smoothed counts: the first bin i whose value is at least
  the value of bin i - 1 (bin 0 has no left neighbour, so the test holds) and strictly greater
  than the value of bin i + 1.
- Valley. The first local minimum after the mode: the first bin j > mode whose value is at most
  the value of bin j - 1 and strictly less than the value of bin j + 1.
- Result. GHOST_NOBODY_M is the centre of the valley bin, 0.5 j + 0.25 m. When no valley exists
  the script prints "no valley" and the field stays undecided.

A secondary histogram is printed for the record only: the same histogram with one value per
stationary run (the median dmin of the run). It never selects.

The script reads the tracking through the engine data layer only. It reads no ground truth file,
no video and no Smart Tagging. It prints bin counts, the mode and the valley, and nothing per
frame. Deterministic: no seed, no clock.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BIN_M = 0.5
MAX_M = 30.0
SEALED = ("DFL-J03WQQ", "DFL-J03WR9")


def histogram(values, bin_m=BIN_M, max_m=MAX_M):
    """(counts, overflow): counts per bin [bin_m i, bin_m (i + 1)) up to max_m, and the rest."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= 0.0)]
    n_bins = int(round(max_m / bin_m))
    index = np.floor(values / bin_m).astype(np.int64)
    inside = index < n_bins
    counts = np.bincount(index[inside], minlength=n_bins)[:n_bins]
    return counts.astype(np.int64), int((~inside).sum())


def smooth3(counts):
    """Centred 3-bin moving average. The two end bins average the 2 bins that exist."""
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.shape[0]
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - 1), min(n, i + 2)
        out[i] = counts[lo:hi].mean()
    return out


def first_mode(smoothed):
    """Index of the first local maximum (>= left, > right), or None."""
    s = np.asarray(smoothed, dtype=np.float64)
    for i in range(s.shape[0] - 1):
        if (i == 0 or s[i] >= s[i - 1]) and s[i] > s[i + 1]:
            return i
    return None


def first_valley(smoothed):
    """(mode, valley): the first local maximum and the first local minimum after it.

    The valley is the first j > mode with s[j] <= s[j - 1] and s[j] < s[j + 1]. Either index is
    None when it does not exist.
    """
    s = np.asarray(smoothed, dtype=np.float64)
    mode = first_mode(s)
    if mode is None:
        return None, None
    for j in range(mode + 1, s.shape[0] - 1):
        if s[j] <= s[j - 1] and s[j] < s[j + 1]:
            return mode, j
    return mode, None


def valley_value(valley, bin_m=BIN_M):
    """Centre of the valley bin in metres, or None."""
    return None if valley is None else bin_m * valley + bin_m / 2.0


def run_medians(values, mask):
    """Median of `values` over each run of True in `mask` (NaN values left out)."""
    from engine.tracks import _runs

    starts, stops = _runs(np.asarray(mask, dtype=bool))
    out = []
    for start, stop in zip(starts, stops):
        seg = values[start:stop]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out.append(float(np.median(seg)))
    return np.array(out)


def measure(game):
    """Summary dict: the frame histogram, its valley and the run-median histogram."""
    from engine import data, deadball, possession, tracks
    from engine.config import Config

    if game in SEALED:
        raise SystemExit("{} is sealed".format(game))
    cfg = Config.default()
    tactical = possession.engine_tactical(cfg, game)
    ball, feats, _ = tracks.prepare(tactical, cfg)
    n = ball["visible"].shape[0]
    graded = np.zeros(n, dtype=bool)
    for _, first, last in data.graded_blocks(game):
        graded[first:last + 1] = True
    dmin = np.asarray(feats["dmin"], dtype=np.float64)
    mask = deadball.stationary_mask(ball, cfg) & graded
    frames = dmin[mask & np.isfinite(dmin)]
    counts, overflow = histogram(frames)
    smoothed = smooth3(counts)
    mode, valley = first_valley(smoothed)
    run_vals = run_medians(dmin, mask)
    run_counts, run_overflow = histogram(run_vals)
    run_mode, run_valley = first_valley(smooth3(run_counts))
    return {"game": game, "stationary_s": cfg.STATIONARY_S,
            "stationary_max_disp_m": cfg.STATIONARY_MAX_DISP_M,
            "graded_frames": int(graded.sum()), "stationary_frames": int(mask.sum()),
            "frames_with_player": int(frames.size), "counts": counts.tolist(),
            "overflow": overflow, "smoothed": [round(float(v), 2) for v in smoothed],
            "mode_bin": mode, "valley_bin": valley, "valley_m": valley_value(valley),
            "runs": int(run_vals.size), "run_counts": run_counts.tolist(),
            "run_overflow": run_overflow, "run_mode_bin": run_mode, "run_valley_bin": run_valley,
            "run_valley_m": valley_value(run_valley)}


def report(result):
    """Plain-text summary: bin counts, smoothed counts, mode and valley."""
    lines = ["game {game}: graded frames {graded_frames}, stationary frames {stationary_frames}, "
             "with a player {frames_with_player}, overflow (>= 30 m) {overflow}".format(**result),
             "", "| bin m | frames | smoothed | runs |", "|---|---|---|---|"]
    for i, (count, smooth, runs) in enumerate(zip(result["counts"], result["smoothed"],
                                                  result["run_counts"])):
        lines.append("| {:.1f}-{:.1f} | {} | {:.2f} | {} |".format(BIN_M * i, BIN_M * (i + 1),
                                                                 count, smooth, runs))
    lines.append("")

    def edge(index):
        return "none" if index is None else "{:.1f}-{:.1f} m".format(BIN_M * index,
                                                                   BIN_M * (index + 1))

    lines.append("frames: mode bin {}, valley bin {}, GHOST_NOBODY_M = {}".format(
        edge(result["mode_bin"]), edge(result["valley_bin"]),
        "no valley" if result["valley_m"] is None else "{:.2f}".format(result["valley_m"])))
    lines.append("runs (record only, {} runs, overflow {}): mode bin {}, valley bin {}".format(
        result["runs"], result["run_overflow"], edge(result["run_mode_bin"]),
        edge(result["run_valley_bin"])))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default="BAR-ATH")
    parser.add_argument("--json", default=None, help="also write the summary JSON here")
    args = parser.parse_args(argv)
    result = measure(args.game)
    if args.json:
        import json
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=1, sort_keys=True)
    sys.stdout.write(report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

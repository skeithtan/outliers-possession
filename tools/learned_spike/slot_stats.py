"""Label-free slot and ball statistics of a tracking bundle. Not part of the submission.

    python -m tools.learned_spike.slot_stats --games BAR-ATH METRICA-S1 ... --out STATS.json
        [--engine-cache DIR] [--sc-engine-cache DIR]

The tool reads the raw tactical view of each match through engine.data. It writes summary
statistics only. It never prints a tracking row, never reads a label and never reads
evaluation/gt. The statistics are allowed on the competition match (team rules: summary
statistics). docs/plan/calibration/STRESS_RATES.md holds the table and the corruption
parameters that come from it.

All statistics count graded frames only (the blocks of games.py), per half.

- Slots used: the slots with at least one visible frame in the half.
- Tracklets: the maximal visible runs of a slot inside a half. The tool gives the births per
  10 minutes, the lifetime quantiles (seconds) per tracklet, and the frame-weighted lifetime
  quantiles. A frame-weighted quantile is the lifetime of the tracklet that a random visible
  player frame belongs to.
- Concurrent slots: the number of visible slots of one team on a frame. The tool gives
  quantiles, the mean, the share of frames above MAX_TEAM and the share below MIN_TEAM.
- Slot gaps: the invisible runs of a slot between two visible runs in one half. A gap is an
  identity jump when the player moves more than V_MAX x gap time + NOISE_M across it.
- Teleports: a step between two consecutive visible frames of a slot above
  V_MAX / fps + NOISE_M. A run of such steps counts as one event.
- Ghosts: two slots of one team, both visible, closer than GHOST_R_M for GHOST_MIN_S or more.
- Ball: the visible share, the quantiles of the gap lengths, and the share of slow frames
  (mean player speed under SLOW_MPS) among the frames with and without the ball.

Python 3.9, numpy and pandas only.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import games  # noqa: E402
from engine import data  # noqa: E402

# Top sprint speed of a footballer is about 10 to 11 m/s. 12 m/s is a safe upper bound.
V_MAX = 12.0
# Margin for the position noise of an automatic tracker, in metres. The p99.9 per-frame step of
# the clean open references is under 0.6 m (STRESS_RATES.md), so 0.5 m on top of the 0.48 m
# physical bound puts the threshold near 1 m per frame at 25 fps.
NOISE_M = 0.5
GHOST_R_M = 0.5
GHOST_MIN_S = 1.0
MAX_TEAM = 11
MIN_TEAM = 9
HIST_MAX = 14
# A collective stop: the mean speed of the visible players under 1.6 m/s (PLAN.md 1.1).
SLOW_MPS = 1.6
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
SEALED = ("DFL-J03WQQ", "DFL-J03WR9")
HOME_DIR = os.path.expanduser("~")
REFS_ROOT = os.path.join(HOME_DIR, "mtah2027", "refs")
SAMPLE_ROOT = os.path.join(HOME_DIR, "mtah2027", "sample-data", "data")
MTAH_ROOT = os.path.join(HOME_DIR, "mtah2027")


def load_raw(game, engine_cache=None, sc_engine_cache=None):
    """The raw tactical dict of a match. The competition match uses the default cache."""
    if game in SEALED:
        raise SystemExit("{} is sealed".format(game))
    if game == games.GAME:
        return data.load_tactical(game)
    from analysis import external_sweep as xs
    root = xs.data_root(game, "raw", REFS_ROOT, SAMPLE_ROOT, MTAH_ROOT)
    cache = sc_engine_cache if game.startswith("SC-") else engine_cache
    return data.load_tactical(game, data_dir=root, cache_dir=cache,
                              use_cache=cache is not None)


def runs(mask):
    """(starts, stops) of the True runs of a 1-D bool array, stops exclusive."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2]


def quantiles(values, weights=None):
    """{"p10": ...} of a 1-D array, NaN keys when empty. Weights give a weighted quantile."""
    values = np.asarray(values, dtype=np.float64)
    out = {}
    if values.size == 0:
        return {"p{:g}".format(100 * q): None for q in QUANTILES}
    if weights is None:
        qs = np.quantile(values, QUANTILES)
    else:
        order = np.argsort(values, kind="mergesort")
        v, w = values[order], np.asarray(weights, dtype=np.float64)[order]
        cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
        qs = np.interp(QUANTILES, cdf, v)
    for q, value in zip(QUANTILES, qs):
        out["p{:g}".format(100 * q)] = round(float(value), 3)
    return out


def metres(tac, game):
    """(px, py, visible) in metres, float64."""
    length, width = data.pitch_m(game)
    px = np.asarray(tac["px"], dtype=np.float64) * length
    py = np.asarray(tac["py"], dtype=np.float64) * width
    return px, py, np.isfinite(px) & np.isfinite(py)


def slot_block_stats(px, py, vis, team, fps):
    """Statistics of one graded block. Arrays are the block rows only."""
    n, n_slots = vis.shape
    minutes = n / fps / 60.0
    jump_m = V_MAX / fps + NOISE_M
    used = np.flatnonzero(vis.any(axis=0))
    lifetimes, gaps, gap_jumps = [], [], 0
    teleports = {0: 0, 1: 0}
    steps_all = []
    for slot in range(n_slots):
        starts, stops = runs(vis[:, slot])
        if starts.size == 0:
            continue
        lifetimes.extend(((stops - starts) / float(fps)).tolist())
        for k in range(1, starts.size):
            g = starts[k] - stops[k - 1]
            gaps.append(g / float(fps))
            a, b = stops[k - 1] - 1, starts[k]
            d = np.hypot(px[b, slot] - px[a, slot], py[b, slot] - py[a, slot])
            if d > V_MAX * g / float(fps) + NOISE_M:
                gap_jumps += 1
        both = vis[1:, slot] & vis[:-1, slot]
        step = np.hypot(np.diff(px[:, slot]), np.diff(py[:, slot]))
        steps_all.append(step[both])
        big = np.zeros(n, dtype=bool)
        big[1:] = both & (step > jump_m)
        teleports[int(team[slot])] += int(runs(big)[0].size)
    lifetimes = np.asarray(lifetimes)
    steps_all = np.concatenate(steps_all) if steps_all else np.zeros(0)
    out = {
        "frames": int(n), "minutes": round(minutes, 2),
        "slots_used": int(used.size),
        "slots_used_team": [int((team[used] == t).sum()) for t in (0, 1)],
        "tracklets": int(lifetimes.size),
        "births_per_10min": round(lifetimes.size / minutes * 10.0, 1),
        "lifetime_s": quantiles(lifetimes),
        "lifetime_s_frame_weighted": quantiles(lifetimes, lifetimes),
        "share_tracklets_under_2s": round(float((lifetimes < 2.0).mean()), 4)
        if lifetimes.size else None,
        "slot_gap_s": quantiles(gaps),
        "slot_gaps": len(gaps),
        "slot_gap_identity_jump_share": round(gap_jumps / float(len(gaps)), 4) if gaps else None,
        "teleport_m_per_frame": round(jump_m, 3),
        "teleports_per_min_team": [round(teleports[t] / minutes, 3) for t in (0, 1)],
        "step_m_p99": round(float(np.quantile(steps_all, 0.99)), 3) if steps_all.size else None,
        "step_m_p999": round(float(np.quantile(steps_all, 0.999)), 3) if steps_all.size else None,
    }
    for t, key in ((0, "home"), (1, "away")):
        count = vis[:, team == t].sum(axis=1)
        out["concurrent_" + key] = {
            "mean": round(float(count.mean()), 3),
            "p5": float(np.quantile(count, 0.05)), "p50": float(np.quantile(count, 0.5)),
            "p95": float(np.quantile(count, 0.95)),
            "share_over_{}".format(MAX_TEAM): round(float((count > MAX_TEAM).mean()), 4),
            "share_under_{}".format(MIN_TEAM): round(float((count < MIN_TEAM).mean()), 4),
            "hist": [round(float((count == k).mean()), 4) for k in range(HIST_MAX)]
            + [round(float((count >= HIST_MAX).mean()), 4)],
        }
    out["ghosts"] = ghost_stats(px, py, vis, team, fps)
    return out


def ghost_stats(px, py, vis, team, fps):
    """Ghost duplicate events: same-team pairs within GHOST_R_M for GHOST_MIN_S or more."""
    n = vis.shape[0]
    min_len = int(round(GHOST_MIN_S * fps))
    events, frames = 0, np.zeros(n, dtype=bool)
    lengths = []
    for t in (0, 1):
        cols = np.flatnonzero(team == t)
        for i in range(cols.size):
            a = cols[i]
            if not vis[:, a].any():
                continue
            for b in cols[i + 1:]:
                both = vis[:, a] & vis[:, b]
                if not both.any():
                    continue
                with np.errstate(invalid="ignore"):
                    near = both & (np.hypot(px[:, a] - px[:, b], py[:, a] - py[:, b]) < GHOST_R_M)
                starts, stops = runs(near)
                keep = (stops - starts) >= min_len
                events += int(keep.sum())
                for s, e in zip(starts[keep], stops[keep]):
                    frames[s:e] = True
                    lengths.append((e - s) / float(fps))
    minutes = n / fps / 60.0
    return {"events": events, "per_10min": round(events / minutes * 10.0, 3),
            "share_frames": round(float(frames.mean()), 5), "length_s": quantiles(lengths)}


def slow_frames(px, py, vis, fps):
    """bool (N,): the mean speed of the visible players is under SLOW_MPS (difference velocity)."""
    from engine import tracks
    vx, vy = tracks._difference_velocity(px, py, vis, fps)
    speed = np.hypot(vx, vy)
    ok = np.isfinite(speed)
    count = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, speed, 0.0).sum(axis=1) / count
    return (count > 0) & (mean < SLOW_MPS)


def ball_stats(bx, blocks, fps, slow=None):
    """Ball visibility and gap lengths on graded frames, and the slow share by ball state."""
    vis_total, total, gaps = 0, 0, []
    long_frames = 0
    slow_missing, slow_visible = [], []
    for _, first, last in blocks:
        b = np.isfinite(np.asarray(bx[first:last + 1], dtype=np.float64))
        if slow is not None:
            s = slow[first:last + 1]
            slow_missing.append(s[~b])
            slow_visible.append(s[b])
        vis_total += int(b.sum())
        total += b.size
        starts, stops = runs(~b)
        lengths = (stops - starts) / float(fps)
        gaps.extend(lengths.tolist())
        long_frames += int((stops - starts)[lengths > 2.0].sum())
    gaps = np.asarray(gaps)
    return {"visible_share": round(vis_total / float(total), 4), "gaps": int(gaps.size),
            "gaps_over_2s": int((gaps > 2.0).sum()),
            "share_in_gaps_over_2s": round(long_frames / float(total), 4),
            "gap_s": quantiles(gaps), "gap_s_frame_weighted": quantiles(gaps, gaps),
            "slow_share_ball_missing": round(float(np.concatenate(slow_missing).mean()), 4)
            if slow_missing and np.concatenate(slow_missing).size else None,
            "slow_share_ball_visible": round(float(np.concatenate(slow_visible).mean()), 4)
            if slow_visible and np.concatenate(slow_visible).size else None}


def match_stats(tac, game):
    """{"halves": {half: block stats}, "ball": ball stats, "n_slots": int}."""
    fps = data.fps(game)
    px, py, vis = metres(tac, game)
    team = np.asarray(tac["team_of_slot"])
    blocks = data.graded_blocks(game)
    halves = {}
    for half, first, last in blocks:
        rows = slice(first, last + 1)
        halves[str(half)] = slot_block_stats(px[rows], py[rows], vis[rows], team, fps)
    return {"game": game, "fps": fps, "n_slots": int(vis.shape[1]),
            "n_slots_team": [int((team == t).sum()) for t in (0, 1)],
            "halves": halves,
            "ball": ball_stats(tac["bx"], blocks, fps, slow_frames(px, py, vis, fps))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--engine-cache")
    parser.add_argument("--sc-engine-cache")
    args = parser.parse_args(argv)
    result = {}
    for game in args.games:
        tac = load_raw(game, args.engine_cache, args.sc_engine_cache)
        result[game] = match_stats(tac, game)
        print(game, file=sys.stderr, flush=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

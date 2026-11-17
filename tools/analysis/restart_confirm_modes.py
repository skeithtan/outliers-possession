"""Label-free RESTART_CONFIRM_S: the M15 accept and reject modes on the full match.

    MTAH_DATA_DIR=~/mtah2027/data python tools/analysis/restart_confirm_modes.py \
        [--windows 2 3 4 5 6 7 8 9 10 11 12] [--json OUT.json]

The rule is pre-registered in docs/plan/calibration/grids.json (bucket b): "midpoint in log time
between p90 of the M15 accept mode and p10 of the reject mode, rounded to whole seconds". M15
is the measurement of RESEARCH-2 1b (tools/analysis/r2_restart_features.py). This script repeats
it on the current engine and prints summary statistics only.

Definitions (those of M15, on the graded frames of the match):

- Candidates. The rows of deadball.restarts(), from possession.prepare().
- Players time. For a candidate at frame f, the number of frames from f to the first frame in
  f .. f + 750 where meanv_all > V_STOP. A candidate with no such frame has the time +inf
  (players do not move within 30 s).
- Accept mode. The candidates that end a stoppage of the M1 state machine
  (possession.dead_intervals(), with the goalkeeper holds as possession.run() passes them).
- Reject mode. The candidates that are not confirmed, lie inside a stoppage interval and come at
  least RESTART_ARM_S after its entry. The confirmation alone rejects them.
- Quantiles. Linear interpolation between order statistics (numpy "linear"). When an
  interpolation touches a +inf value the quantile is +inf.
- Value. W = sqrt(p90(accept) * p10(reject)) / fps seconds, rounded to whole seconds (half up).

The classification depends on the window: a candidate is confirmed when the players move within
RESTART_CONFIRM_S of it (or by the chain rule). So the script runs the detection once for each
window w of `--windows` and computes the value W(w) with the classification at w. The shipped
window must play no role (COMPLIANCE_PROTOCOL.md, Section 1). The selected value is therefore
the fixed point: the w with W(w) = w. When exactly one fixed point exists it is the value. When
none or several exist the script says so and the field stays undecided. The row at the shipped
window is the literal M15 re-run. It is printed for the record only.

The script reads the tracking and the Smart Tagging through the engine only (the engine reads
them for its own decisions). It reads no ground truth file and no video. It prints counts and
quantiles, never a frame. Deterministic: no seed, no clock.
"""

import argparse
import dataclasses
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HORIZON = 750
DEFAULT_WINDOWS = tuple(range(2, 13))


def censored_quantile(values, q):
    """Quantile q (0..1) with linear interpolation. +inf values are allowed.

    The result is +inf when the interpolation uses a +inf order statistic with a non-zero weight.
    Returns None for an empty input.
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    if v.size == 0:
        return None
    pos = q * (v.size - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, v.size - 1)
    w = pos - lo
    if w == 0.0 or hi == lo:
        return float(v[lo])
    if not (np.isfinite(v[lo]) and np.isfinite(v[hi])):
        return float("inf")
    return float(v[lo] + w * (v[hi] - v[lo]))


def log_midpoint_seconds(accept_p90, reject_p10, fps):
    """sqrt(accept_p90 * reject_p10) / fps, or None when either is missing, zero or +inf."""
    if accept_p90 is None or reject_p10 is None:
        return None
    if not (np.isfinite(accept_p90) and np.isfinite(reject_p10)) or accept_p90 <= 0 \
            or reject_p10 <= 0:
        return None
    return math.sqrt(accept_p90 * reject_p10) / float(fps)


def round_half_up(value):
    """Whole number nearest to value, a half rounds up. None stays None."""
    return None if value is None else int(math.floor(value + 0.5))


def fixed_points(values_by_window):
    """Sorted windows w with values_by_window[w] == w."""
    return sorted(w for w, v in values_by_window.items() if v is not None and v == w)


def players_times(moving, frames, horizon=HORIZON):
    """Frames from each candidate to the first moving frame within horizon, +inf if none."""
    out = []
    n = moving.shape[0]
    for f in frames:
        seg = np.flatnonzero(moving[int(f):min(n, int(f) + horizon + 1)])
        out.append(float(seg[0]) if seg.size else float("inf"))
    return np.array(out)


def stoppage_intervals(inputs, cfg):
    """[(enter, leave)] of possession.dead_intervals(), replicated to keep the entries.

    The replica is asserted equal to the engine's mask and restart list.
    """
    from engine import possession
    from engine.tracks import _runs

    onsets, e_dead, blocks = inputs["onsets"], inputs["e_dead"], inputs["blocks"]
    kickoffs, fps, n = inputs["kickoffs"], inputs["fps"], inputs["n_frames"]
    holds = inputs.get("holds")
    candidates = inputs["restarts"]
    high = np.asarray(e_dead) > cfg.DEAD_EVIDENCE_HIGH
    dwell = max(1, int(round(cfg.COLLECTIVE_STOP_S * fps)))
    arm = max(1, int(round(cfg.RESTART_ARM_S * fps)))
    rs, re_ = _runs(high)
    onset_entries = onsets.loc[onsets["mechanism"] != "collective_stop", "frame"].to_numpy()
    entries = np.unique(np.concatenate([rs[(re_ - rs) >= dwell], onset_entries]).astype(np.int64))
    onset_set = set(int(f) for f in onset_entries)
    usable = candidates[candidates["confirmed"]]
    cf = usable["frame"].to_numpy()
    teams = usable["team"].tolist()
    dead = np.zeros(n, dtype=bool)
    rl, iv = [], []
    for half, first, last in blocks:
        stop = last + 1
        pos = first
        kick = kickoffs.get(half)
        if kick is not None and first < kick <= last:
            dead[first:kick] = True
            rl.append((int(kick), None))
            pos = kick
        while pos < stop:
            later = entries[(entries >= pos) & (entries < stop)]
            if later.size == 0:
                break
            enter = int(later[0])
            ok = np.flatnonzero((cf >= enter + arm) & (cf < stop))
            leave = int(cf[ok[0]]) if ok.size else stop
            if holds is not None and holds[0][enter] and enter not in onset_set \
                    and not holds[1][enter:leave].any():
                pos = enter + 1
                continue
            if ok.size == 0:
                dead[enter:stop] = True
                iv.append((enter, stop))
                break
            dead[enter:leave] = True
            rl.append((leave, teams[ok[0]]))
            iv.append((enter, leave))
            pos = leave
    ref_dead, ref_rl = possession.dead_intervals(n, onsets, e_dead, candidates, blocks, kickoffs,
                                                 cfg, fps, holds)
    assert np.array_equal(dead, ref_dead) and rl == ref_rl
    return iv


def classify(inputs, cfg):
    """(accept frames, reject frames) of the graded candidates (see the module docstring)."""
    blocks = inputs["blocks"]
    cand = inputs["restarts"]
    arm = max(1, int(round(cfg.RESTART_ARM_S * inputs["fps"])))
    iv = stoppage_intervals(inputs, cfg)
    leaves = set(leave for _, leave in iv)
    frames = cand["frame"].to_numpy()
    confirmed = cand["confirmed"].to_numpy()
    accept, reject = [], []
    used_leaves = set()
    for f, ok in zip(frames, confirmed):
        f = int(f)
        if not any(first <= f <= last for _, first, last in blocks):
            continue
        if ok and f in leaves and f not in used_leaves:
            accept.append(f)
            used_leaves.add(f)
            continue
        if ok:
            continue
        if any(enter + arm <= f < leave for enter, leave in iv):
            reject.append(f)
    return np.array(accept, dtype=np.int64), np.array(reject, dtype=np.int64)


def measure_window(window, game=None):
    """Summary dict of the two modes with the classification at `window` seconds."""
    from engine import deadball, possession
    from engine.config import Config

    cfg = dataclasses.replace(Config.default(), RESTART_CONFIRM_S=float(window))
    inputs = possession.prepare(cfg, game=game)
    fps = inputs["fps"]
    moving = deadball._above(inputs["feats"]["meanv_all"], cfg.V_STOP)
    accept, reject = classify(inputs, cfg)
    ta, tr = players_times(moving, accept), players_times(moving, reject)
    a90 = censored_quantile(ta, 0.9)
    r10 = censored_quantile(tr, 0.1)
    mid = log_midpoint_seconds(a90, r10, fps)
    return {"window_s": float(window), "fps": fps, "accept_n": int(ta.size),
            "accept_never": int(np.isinf(ta).sum()),
            "accept_q": [censored_quantile(ta, q) for q in (0.1, 0.5, 0.9)],
            "reject_n": int(tr.size), "reject_never": int(np.isinf(tr).sum()),
            "reject_q": [censored_quantile(tr, q) for q in (0.1, 0.5, 0.9)],
            "midpoint_s": mid, "value_s": round_half_up(mid)}


def _fmt(value):
    if value is None:
        return "-"
    if not np.isfinite(value):
        return "inf"
    return "{:.1f}".format(value)


def report(rows, shipped):
    """Markdown table of the windows, then the fixed points."""
    lines = ["| window s | accept n (never) | accept p10/p50/p90 fr | reject n (never) | "
             "reject p10/p50/p90 fr | log midpoint s | rounded |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {:g}{} | {} ({}) | {} | {} ({}) | {} | {} | {} |".format(
            r["window_s"], " (shipped, record only)" if r["window_s"] == shipped else "",
            r["accept_n"], r["accept_never"], "/".join(_fmt(v) for v in r["accept_q"]),
            r["reject_n"], r["reject_never"], "/".join(_fmt(v) for v in r["reject_q"]),
            _fmt(r["midpoint_s"]), "-" if r["value_s"] is None else r["value_s"]))
    points = fixed_points(dict((int(r["window_s"]), r["value_s"]) for r in rows))
    lines.append("")
    lines.append("fixed points W(w) = w: {}".format(points or "none"))
    if len(points) == 1:
        lines.append("RESTART_CONFIRM_S = {}".format(points[0]))
    else:
        lines.append("RESTART_CONFIRM_S undecided ({} fixed points)".format(len(points)))
    return "\n".join(lines) + "\n"


def main(argv=None):
    from engine.config import Config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--game", default=None, help="default games.GAME")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if args.game in ("DFL-J03WQQ", "DFL-J03WR9"):
        raise SystemExit("sealed match refused")
    rows = [measure_window(w, args.game) for w in args.windows]
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=1, sort_keys=True)
    sys.stdout.write(report(rows, Config.default().RESTART_CONFIRM_S))
    return 0


if __name__ == "__main__":
    sys.exit(main())

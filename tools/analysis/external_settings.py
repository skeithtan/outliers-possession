"""Measure the settings of docs/plan/calibration/EXTERNAL_SETTINGS.md on external references.

    python tools/analysis/external_settings.py measure --stage STAGE --game NAME --ref REF.csv
        --data-dir DIR --out OUT.json [--set FIELD=VALUE ...]
    python tools/analysis/external_settings.py select --stage STAGE OUT.json [OUT.json ...]

`measure` writes the summary statistics of one match. `select` applies the rules of the protocol
to the files of all matches and prints the values. The stages follow the order of the protocol:

    stop      V_STOP, players only
    confirm   ONSET_CONFIRM_S and RESTART_CONFIRM_S, players only, needs --set V_STOP=...
    nearest   the lag curve, the distance bands and the pursuit look-back, matched view,
              needs --set V_STOP=... --set ONSET_CONFIRM_S=...
    rise      RESTART_K, sigma 9.5 view

The tool refuses BAR-ATH and the sealed matches. It reads no ground truth file. Write the JSON
files outside the repository. They hold summary statistics only. Deterministic.
"""

import argparse
import dataclasses
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tools", "analysis")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import provenance_recheck as pr  # noqa: E402

V_GRID = tuple(round(0.8 + 0.1 * i, 1) for i in range(23))        # 0.8 to 3.0 m/s
H_GRID = tuple(1.0 + 0.5 * i for i in range(23))                  # 1 to 12 s
LAGS = tuple(range(0, 46))
LOOKBACKS = tuple(range(0, 21))
# Control frames of the restart test: at least this far before the end of the D run, this apart.
CONTROL_GAP_S = 12.0
CONTROL_STEP = 25
# A row is inconclusive when J is at most this on more than half the matches.
J_FLOOR = 0.05
LOOKBACK_SHARE = 0.90
# The band ladder of the recheck: (name, upper edge in metres).
LADDER = (("le1.0", 1.0), ("1-1.5", 1.5), ("1.5-2", 2.0), ("2-3", 3.0), ("3-4", 4.0), ("4-5", 5.0),
          ("5-8", 8.0))


def youden_curve(true_pass, control_pass):
    """list of J = share of true cases that pass minus share of control cases that pass."""
    return [float(a) - float(b) for a, b in zip(true_pass, control_pass)]


def best_on_grid(grid, values):
    """(grid value, J) of the largest J. A tie goes to the smaller grid value."""
    index = int(np.argmax(np.asarray(values, dtype=np.float64)))
    return grid[index], float(values[index])


def round_to(value, step):
    """`value` rounded half up to a multiple of `step`."""
    return round(math.floor(round(value / step, 9) + 0.5) * step, 6)


def speed_thresholds(meanv, labels, grid=V_GRID):
    """(share of D below, share of alive below) for each threshold of the grid."""
    meanv = np.asarray(meanv, dtype=np.float64)
    labels = np.asarray(labels)
    finite = np.isfinite(meanv)
    dead = meanv[finite & (labels == "D")]
    alive = meanv[finite & ((labels == "H") | (labels == "A"))]
    return ([float((dead < t).mean()) for t in grid], [float((alive < t).mean()) for t in grid])


def reference_restarts(labels):
    """int64: the frames t with an alive label and label D at t - 1."""
    labels = np.asarray(labels)
    alive = (labels == "H") | (labels == "A")
    return np.flatnonzero(alive[1:] & (labels[:-1] == "D")) + 1


def control_frames(labels, fps, gap_s=CONTROL_GAP_S, step=CONTROL_STEP):
    """int64: D frames at least `gap_s` before the end of their run, every `step` frames."""
    labels = np.asarray(labels)
    dead = (labels == "D").astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], dead, [0]))))
    gap = int(round(gap_s * fps))
    out = []
    for start, stop in zip(edges[0::2], edges[1::2]):
        out.extend(range(int(start), int(stop) - gap + 1, step))
    return np.asarray(out, dtype=np.int64)


def moving_within(frames, meanv, v_stop, horizons):
    """list of shares: `frames` where meanv rises above v_stop within each horizon in frames."""
    moving = np.asarray(meanv, dtype=np.float64) > v_stop
    index = np.flatnonzero(moving)
    frames = np.asarray(frames, dtype=np.int64)
    if frames.size == 0:
        return [0.0 for _ in horizons]
    slot = np.searchsorted(index, frames, side="left")
    has = slot < index.size
    wait = np.full(frames.shape[0], np.inf)
    wait[has] = index[slot[has]] - frames[has]
    return [float((wait <= h).mean()) for h in horizons]


def lookback_shares(offsets, total, grid=LOOKBACKS):
    """list of shares of `total` with an offset of at most each look-back."""
    offsets = np.asarray(offsets, dtype=np.float64)
    return [float((offsets <= g).sum()) / total if total else 0.0 for g in grid]


def band_rates(curve, lag):
    """{band: agreement rate} of the ladder at one lag."""
    row = curve[str(int(lag))]
    return dict((name, row[name][0] / float(row[name][1])) for name, _ in LADDER if row[name][1])


# -- measure ---------------------------------------------------------------------------------------

def measure_stop(feats, labels):
    dead, alive = speed_thresholds(feats["meanv_all"], labels)
    return {"grid": list(V_GRID), "dead": dead, "alive": alive, "J": youden_curve(dead, alive)}


def measure_confirm(ball, feats, labels, cfg):
    from engine import deadball as db

    fps = float(ball["fps"])
    frames = [db._frames(s, fps) for s in H_GRID]
    starts, stops = db._stop_runs(feats, cfg, fps)
    onsets = pr.reference_onsets(labels)
    delays = pr.stop_delays(onsets, starts, stops)
    recall = [float((delays <= h).mean()) for h in frames]
    false = []
    for h in frames:
        passed, eligible = pr.false_confirm_rate(labels, starts, stops, h)
        false.append(passed / float(eligible) if eligible else 0.0)
    restarts = reference_restarts(labels)
    control = control_frames(labels, fps)
    meanv = feats["meanv_all"]
    true_move = moving_within(restarts, meanv, cfg.V_STOP, frames)
    false_move = moving_within(control, meanv, cfg.V_STOP, frames)
    return {"grid": list(H_GRID),
            "onset": {"n": int(onsets.size), "true": recall, "control": false,
                      "J": youden_curve(recall, false)},
            "restart": {"n": int(restarts.size), "n_control": int(control.size), "true": true_move,
                        "control": false_move, "J": youden_curve(true_move, false_move)}}


def measure_nearest(ball, feats, labels, cfg):
    from engine import deadball as db

    fps = float(ball["fps"])
    owner = pr.owner_codes(labels)
    ghost = db.ghost_mask(ball, feats, cfg)
    usable = ball["visible"] & np.isfinite(feats["dmin"]) & (feats["nearest_team"] >= 0) & ~ghost
    curve = pr.nearest_agreement(owner, feats["nearest_team"], feats["dmin"], usable, LAGS)
    _, _, top, rate, _ = pr.plateau(curve, "all")
    horizon = db._frames(cfg.ONSET_CONFIRM_S, fps)
    chase = db._frames(cfg.COLLECTIVE_STOP_S, fps)
    rows = pr.pursuit_candidates(ball, feats, cfg, horizon, chase)
    onsets = pr.reference_onsets(labels)
    offsets, total = [], 0
    for row in rows:
        if not pr.is_owned(row, onsets):
            continue
        team = str(np.asarray(labels)[row["start"]])
        if team not in pr.ALIVE:
            continue
        total += 1
        off = pr.lookback_offset(row["start"], team, feats["dH"], feats["dA"], ball["visible"],
                                 cfg.CONTESTED_MARGIN_M)
        if off is not None:
            offsets.append(off)
    return {"best_lag": int(top), "best_rate": float(rate), "curve": curve,
            "lookback": {"grid": list(LOOKBACKS), "n": total,
                         "share": lookback_shares(offsets, total)}}


def measure_rise(ball, feats, labels, cfg):
    rises = pr.handover_rises(ball, feats, pr.reference_handovers(labels), cfg)
    return {"kicks": rises["kicks"], "gain_len_q": rises["gain_len_q"]}


def measure(stage, game, ref_path, data_dir, overrides):
    from engine.config import Config

    if game in pr.REFUSED:
        raise SystemExit("{} is refused: see the module docstring".format(game))
    cfg = dataclasses.replace(Config.default(), **overrides)
    tactical, ball, feats = pr.load_match(game, data_dir, cfg)
    labels = pr.read_labels(ref_path, int(tactical["n_frames"]))
    fn = {"stop": lambda: measure_stop(feats, labels),
          "confirm": lambda: measure_confirm(ball, feats, labels, cfg),
          "nearest": lambda: measure_nearest(ball, feats, labels, cfg),
          "rise": lambda: measure_rise(ball, feats, labels, cfg)}[stage]
    return {"game": game, "stage": stage, "overrides": overrides, "result": fn()}


# -- select ----------------------------------------------------------------------------------------

def select_grid(rows, step):
    """The median over matches of the best grid value, with the inconclusive test.

    `rows` is a list of (game, grid, J curve).
    """
    best = [(game,) + best_on_grid(grid, curve) for game, grid, curve in rows]
    values = [b[1] for b in best]
    weak = sum(1 for b in best if b[2] <= J_FLOOR)
    return {"per_match": best, "median": float(np.median(values)),
            "value": round_to(float(np.median(values)), step), "weak_matches": weak,
            "inconclusive": weak > len(best) / 2.0}


def select_nearest(payloads):
    lags = [p["result"]["best_lag"] for p in payloads]
    lag = int(round_to(float(np.median(lags)), 1))
    rates = [band_rates(p["result"]["curve"], lag) for p in payloads]
    drops = []
    for (name, edge), (nxt, _) in zip(LADDER[:-1], LADDER[1:]):
        drops.append((edge, float(np.median([r[name] - r[nxt] for r in rates]))))
    near_close = max(drops, key=lambda d: d[1])[0]
    close_band, far_band = "le{:.1f}".format(near_close), "{:g}-5".format(near_close)
    close = [p["result"]["curve"][str(lag)][close_band] for p in payloads]
    far = [p["result"]["curve"][str(lag)][far_band] for p in payloads]
    p_close = float(np.median([a / float(b) for a, b in close]))
    p_far = float(np.median([a / float(b) for a, b in far]))
    grid = payloads[0]["result"]["lookback"]["grid"]
    shares = np.array([p["result"]["lookback"]["share"] for p in payloads])
    medians = np.median(shares, axis=0)
    reached = [g for g, m in zip(grid, medians) if m >= LOOKBACK_SHARE]
    return {"best_lags": dict((p["game"], p["result"]["best_lag"]) for p in payloads),
            "NEAREST_LAG": lag, "band_drops": drops, "NEAR_CLOSE_M": near_close,
            "P_NEAR_CLOSE": round_to(p_close, 0.001), "EVENT_CONF_MAX": round_to(p_close, 0.001),
            "P_NEAR_FAR": round_to(p_far, 0.001),
            "lookback_median": dict((int(g), round(float(m), 3)) for g, m in zip(grid, medians)),
            "lookback_n": dict((p["game"], p["result"]["lookback"]["n"]) for p in payloads),
            "PURSUIT_KICK_LOOKBACK": reached[0] if reached else None}


def select(stage, payloads):
    if stage == "stop":
        rows = [(p["game"], p["result"]["grid"], p["result"]["J"]) for p in payloads]
        return {"V_STOP": select_grid(rows, 0.1)}
    if stage == "confirm":
        out = {}
        for field, key in (("ONSET_CONFIRM_S", "onset"), ("RESTART_CONFIRM_S", "restart")):
            rows = [(p["game"], p["result"]["grid"], p["result"][key]["J"]) for p in payloads]
            out[field] = select_grid(rows, 0.5)
        return out
    if stage == "nearest":
        return select_nearest(payloads)
    p90 = [p["result"]["gain_len_q"]["p90"] for p in payloads]
    return {"per_match": dict((p["game"], p["result"]["gain_len_q"]["p90"]) for p in payloads),
            "kicks": dict((p["game"], p["result"]["kicks"]) for p in payloads),
            "RESTART_K": int(round_to(float(np.median(p90)), 1))}


def _override(text):
    name, _, value = text.partition("=")
    return name, float(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--stage", required=True, choices=("stop", "confirm", "nearest", "rise"))
    m.add_argument("--game", required=True)
    m.add_argument("--ref", required=True)
    m.add_argument("--data-dir", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--set", action="append", default=[], type=_override)
    s = sub.add_parser("select")
    s.add_argument("--stage", required=True, choices=("stop", "confirm", "nearest", "rise"))
    s.add_argument("inputs", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "measure":
        payload = measure(args.stage, args.game, args.ref, args.data_dir, dict(args.set))
        with open(args.out, "w") as handle:
            json.dump(payload, handle, sort_keys=True)
        return 0
    payloads = []
    for path in args.inputs:
        with open(path) as handle:
            payloads.append(json.load(handle))
    print(json.dumps(select(args.stage, payloads), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Shared helpers for the r2 restart-confirmation study (research direction b).

Not engine code. Loads the engine inputs once (pickled in the scratch directory), classifies the
restart candidates of deadball.restarts() against the M1 state machine, measures per-candidate
features, and reruns possession.run() with a replaced `confirmed` column.

The engine is never edited: a variant replaces inputs["restarts"]["confirmed"] and calls
possession.run(cfg, inputs=...). That column is read by dead_intervals(), shift_stoppage(),
kickoff_teams() and the decoder's restart switch costs, so one replacement changes every use.

Only the PUBLIC ground truth is read (tools/gt.public_gt_codes), and only by public_report().
"""

import os
import tempfile
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine import data, possession, submission  # noqa: E402
from engine.config import Config  # noqa: E402
from engine.tracks import _runs, velocity_step  # noqa: E402

SCRATCH = os.environ.get(
    "R2_SCRATCH",
    os.path.join(tempfile.gettempdir(), "r2_research", "b_restart"))
FPS = 25.0
V_STOP = 1.6
V_NEAR = 1.5
PUBLIC_RESTARTS = (12989, 16400, 17838, 18389)
PUBLIC = (11185, 18686)


def engine_hash():
    """md5 over engine/*.py, so a cache built on another engine version is never reused."""
    import hashlib
    h = hashlib.md5()
    edir = os.path.join(ROOT, "engine")
    for name in sorted(os.listdir(edir)):
        if name.endswith(".py"):
            with open(os.path.join(edir, name), "rb") as fh:
                h.update(name.encode() + fh.read())
    return h.hexdigest()


def load(cfg=None, tag="base", prepare=None):
    """possession.prepare(cfg) plus tactical player positions, cached in SCRATCH/inputs_<tag>.pkl.

    `prepare` replaces possession.prepare (a variant with a monkeypatched detector); the cache
    key is the detection key, the engine source hash and the tag.
    """
    cfg = cfg or Config.default()
    path = os.path.join(SCRATCH, "inputs_%s.pkl" % tag)
    key = (possession.detection_key(cfg), engine_hash(), tag)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            inputs = pickle.load(fh)
        if inputs.get("_cache_key") == key:
            return inputs
    inputs = (prepare or possession.prepare)(cfg)
    tac = data.load_tactical()
    length, width = data.pitch_m()
    inputs["px"] = np.asarray(tac["px"], dtype=np.float64) * length
    inputs["py"] = np.asarray(tac["py"], dtype=np.float64) * width
    inputs["ps"] = np.asarray(tac["ps"], dtype=np.float64)
    inputs["ball"]["dv_9"] = velocity_step(inputs["ball"]["speed"], 9, inputs["ball"]["run_id"])
    inputs["_cache_key"] = key
    os.makedirs(SCRATCH, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(inputs, fh, protocol=4)
    return inputs


def graded(frame, blocks):
    """True if `frame` is inside a graded block."""
    return any(first <= frame <= last for _, first, last in blocks)


def stoppage_intervals(inputs, cfg, candidates):
    """Replica of possession.dead_intervals that also returns the intervals.

    Returns (dead, restart_list, intervals) with intervals = [(enter, leave, cand_row or -1)];
    leave = block stop when no restart follows. Asserted equal to the engine's mask and list.
    """
    onsets, e_dead, blocks = inputs["onsets"], inputs["e_dead"], inputs["blocks"]
    kickoffs, fps, n = inputs["kickoffs"], inputs["fps"], inputs["n_frames"]
    high = np.asarray(e_dead) > cfg.DEAD_EVIDENCE_HIGH
    dwell = max(1, int(round(cfg.COLLECTIVE_STOP_S * fps)))
    arm = max(1, int(round(cfg.RESTART_ARM_S * fps)))
    rs, re_ = _runs(high)
    entries = np.unique(np.concatenate([
        rs[(re_ - rs) >= dwell],
        onsets.loc[onsets["mechanism"] != "collective_stop", "frame"].to_numpy()]).astype(np.int64))
    conf = candidates["confirmed"].to_numpy()
    rows = np.flatnonzero(conf)
    cf = candidates["frame"].to_numpy()[rows]
    teams = candidates["team"].tolist()
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
            if ok.size == 0:
                dead[enter:stop] = True
                iv.append((enter, stop, -1))
                break
            leave = int(cf[ok[0]])
            dead[enter:leave] = True
            rl.append((leave, teams[rows[ok[0]]]))
            iv.append((enter, leave, int(rows[ok[0]])))
            pos = leave
    ref_dead, ref_rl = possession.dead_intervals(n, onsets, e_dead, candidates, blocks, kickoffs,
                                                 cfg, fps)
    assert np.array_equal(dead, ref_dead) and rl == ref_rl
    return dead, rl, iv


def classify(inputs, cfg):
    """DataFrame of graded candidates with a `cls` column and its stoppage entry.

    exit         the candidate ends a stoppage of the current state machine
    confirm_rej  unconfirmed, inside a stoppage, at least RESTART_ARM_S after its entry (so the
                 confirmation alone rejected it)
    not_armed    inside a stoppage but within RESTART_ARM_S of its entry
    alive        outside every stoppage interval (confirmed or not)
    """
    cand = inputs["restarts"].copy()
    blocks = inputs["blocks"]
    dead, rl, iv = stoppage_intervals(inputs, cfg, cand)
    arm = int(round(cfg.RESTART_ARM_S * inputs["fps"]))
    exit_rows = {r: e for e, _, r in iv if r >= 0}
    cls, entry = [], []
    for i, f in enumerate(cand["frame"].to_numpy()):
        f = int(f)
        if i in exit_rows:
            cls.append("exit")
            entry.append(exit_rows[i])
            continue
        hit = [e for e, l, _ in iv if e <= f < l]
        if hit:
            entry.append(hit[0])
            cls.append("confirm_rej" if f >= hit[0] + arm else "not_armed")
        else:
            cls.append("alive")
            entry.append(-1)
    cand["cls"] = cls
    cand["entry"] = entry
    cand["row"] = np.arange(len(cand))
    cand = cand[[graded(int(f), blocks) for f in cand["frame"]]].reset_index(drop=True)
    return cand, dead, rl, iv


def label_exits(labels, blocks):
    """[(frame, to_label)] D->H/A changes of a label vector inside graded blocks."""
    out = []
    lab = np.asarray(labels)
    for _, first, last in blocks:
        seg = lab[first:last + 1]
        idx = np.flatnonzero((seg[:-1] == "D") & (seg[1:] != "D")) + 1
        out += [(int(first + i), str(seg[i])) for i in idx]
    return out


def first_after(mask, f, horizon):
    """Frames from f to the first True of mask in [f, f + horizon], NaN if none."""
    seg = np.flatnonzero(mask[f:f + horizon + 1])
    return float(seg[0]) if seg.size else np.nan


def ball_rest_frame(speed, visible, f, start_after, rest_speed=1.0, rest_frames=25):
    """First frame >= start_after where the visible ball stays below rest_speed for rest_frames."""
    n = speed.shape[0]
    run = 0
    for t in range(start_after, min(n, f + 750)):
        if visible[t] and np.isfinite(speed[t]) and speed[t] < rest_speed:
            run += 1
            if run >= rest_frames:
                return t - rest_frames + 1
        else:
            run = 0
    return -1


def peak_end(speed, f, k=9):
    """End of the ball's rise: first local maximum of speed at or after f (within 2 s), else
    argmax over [f, f + 2 s]."""
    n = speed.shape[0]
    hi = min(n - 1, f + 50)
    for t in range(f, hi):
        if np.isfinite(speed[t]) and np.isfinite(speed[t + 1]) and speed[t + 1] < speed[t] \
                and speed[t] > speed[f] + 0.5:
            return t
    seg = speed[f:hi + 1]
    return f + int(np.nanargmax(seg)) if np.isfinite(seg).any() else f


def features(inputs, cand):
    """Per-candidate features (one row per candidate frame)."""
    ball, feats = inputs["ball"], inputs["feats"]
    x, y, sp, vis = ball["x"], ball["y"], ball["speed"], ball["visible"]
    moving_all = np.isfinite(feats["meanv_all"]) & (np.nan_to_num(feats["meanv_all"]) > V_STOP)
    moving_near = np.isfinite(feats["meanv_near"]) & (np.nan_to_num(feats["meanv_near"]) > V_NEAR)
    tags = inputs["tagging"]
    sp_rows = tags[tags["code"] == "SET PIECES"]
    px, py = inputs["px"], inputs["py"]
    rows = []
    for f in cand["frame"].to_numpy():
        f = int(f)
        r = {"frame": f, "vis": bool(vis[f])}
        r["t_all"] = first_after(moving_all, f, 750)
        r["t_near"] = first_after(moving_near, f, 750)
        for T in (1, 2, 4, 6):
            hi = f + int(T * FPS)
            seg = np.arange(f, hi + 1)
            ok = vis[seg]
            if vis[f] and ok.any():
                d = np.hypot(x[seg][ok] - x[f], y[seg][ok] - y[f])
                r["disp%d" % T] = float(d.max())
                r["net%d" % T] = float(d[-1]) if ok[-1] else np.nan
            else:
                r["disp%d" % T] = np.nan
                r["net%d" % T] = np.nan
        seg = sp[f:f + 51]
        r["peak2"] = float(np.nanmax(seg)) if np.isfinite(seg).any() else np.nan
        r["sp0"] = float(sp[f]) if np.isfinite(sp[f]) else np.nan
        r["dv9"] = float(ball["dv_9"][f]) if np.isfinite(ball["dv_9"][f]) else np.nan
        if vis[f]:
            d = np.hypot(px[f] - x[f], py[f] - y[f])
            r["n5"] = int(np.sum(np.isfinite(d) & (np.nan_to_num(d, nan=99) <= 5.0)))
        else:
            r["n5"] = -1
        inside = sp_rows[(sp_rows["start_frame"] <= f) & (f < sp_rows["end_frame"])]
        near = sp_rows[(sp_rows["start_frame"] <= f) & (f < sp_rows["end_frame"] + 125)]
        r["sp_tag"] = bool(len(inside))
        r["sp_tag_tail"] = bool(len(near))
        r["sp_type"] = str(near["type"].iloc[0]) if len(near) else ""
        pk = peak_end(sp, f)
        r["peak_end"] = pk
        r["t_all_from_peak"] = first_after(moving_all, pk, 750)
        rest = ball_rest_frame(sp, vis, f, pk)
        r["rest_after"] = (rest - f) if rest >= 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def pct(series, qs=(10, 50, 90)):
    """'p10/p50/p90 (n, nan)' string of a numeric series."""
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    ok = v[np.isfinite(v)]
    if ok.size == 0:
        return "- (n=0, nan=%d)" % v.size
    q = np.percentile(ok, qs)
    return "/".join("%.1f" % a for a in q) + " (n=%d, nan=%d)" % (ok.size, v.size - ok.size)


def run_with_confirmed(inputs, cfg, confirmed):
    """possession.run() with inputs["restarts"]["confirmed"] replaced. Returns (labels, parts)."""
    alt = dict(inputs)
    table = inputs["restarts"].copy()
    table["confirmed"] = np.asarray(confirmed, dtype=bool)
    alt["restarts"] = table
    return possession.run(cfg, return_parts=True, inputs=alt)


def write_and_sanity(labels, name):
    """Write evaluation/results/<name>.csv and return (path, md5, sanity stats)."""
    import sanity
    path = os.path.join(ROOT, "evaluation", "results", name + ".csv")
    submission.write_submission(labels, path)
    stats = sanity.sanity_stats(labels)
    return path, submission.md5_of(path), stats


def public_report(labels, tol=25):
    """Public-window transitions: (pairs, missed, spurious) at tol, and spellF1 mean."""
    import possession_metrics as pm
    from gt import public_gt_codes
    gt = public_gt_codes()
    pred = pm.label_codes(pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}))
    res = pm.evaluate(gt, pred, bootstrap_n=0)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    pairs, missed, spurious = pm.match_transitions(pm.transitions(gt, runs),
                                                   pm.transitions(filled, runs), tol)
    return pairs, missed, spurious, res

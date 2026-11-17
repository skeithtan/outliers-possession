"""r2 task 2: alternative restart confirmations, measured on the full match and the public window.

    python tools/analysis/r2_restart_variants.py [--write]

Each variant recomputes the `confirmed` column of deadball.restarts() for ALL candidates and
reruns possession.run() with it (r2_restart_common.run_with_confirmed). --write also writes
evaluation/results/rx_b_<variant>.csv (sanity numbers are computed either way).
Deterministic. Reads the public GT only (tools/gt.py).

Variants (window lengths in frames at 25 fps):
  cur      meanv_all > V_STOP in [f, f + 150]                       (engine, md5 check)
  A        ball max displacement >= 10 m within 4 s of f            (replaces the players test)
  A_or     cur OR A
  B        the ball is still moving 4 s after f, or a further touch/release/control/dribble event
           comes before it stops (stop = visible speed < 1 m/s for 1 s, or ball off the pitch /
           missing for 1 s)
  C        meanv_all > V_STOP in [f, peak + 150], peak = end of the ball's rise
  D        meanv_all > V_STOP in [f + delta_kind, f + delta_kind + 150] (DELTA_RESTART_KICK /
           DELTA_RESTART_REAPPEAR, the shifted edge)
  W8, W10  meanv_all > V_STOP in [f, f + 200] / [f, f + 250]        (window-length sensitivity)
  H        cur, or (kick and the next cur-confirmed candidate within CHAIN frames): play goes on.
           CHAIN = 87 frames = median gap between consecutive kick candidates in alive play.
Suffix _k9: the same with DELTA_RESTART_KICK = 9 (the smallest kick offset that brings an exit
at 18355 inside +-25 of 18389).
"""

import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_restart_common as c  # noqa: E402
from engine import deadball  # noqa: E402
from engine.config import Config  # noqa: E402

CHAIN = 87
REPLAYS_PUBLIC = (16058, 17425)


def moving_mask(feats):
    v = feats["meanv_all"]
    return np.isfinite(v) & (np.nan_to_num(v) > c.V_STOP)


def window_confirm(frames, moving, lo_off, hi_off):
    """moving.any() over [f + lo_off[i], f + hi_off[i]] per candidate."""
    n = moving.shape[0]
    out = np.zeros(len(frames), dtype=bool)
    for i, f in enumerate(frames):
        lo, hi = max(0, f + int(lo_off[i])), min(n, f + int(hi_off[i]) + 1)
        out[i] = bool(moving[lo:hi].any())
    return out


def stop_frame(ball, f, start, pitch, horizon=750):
    """First frame >= start where play visibly stops: speed < 1 m/s for 25 frames, or the ball
    is off the pitch or missing for 25 frames. -1 if none within horizon."""
    sp, vis = ball["speed"], ball["visible"]
    out = deadball.outside_pitch(ball, pitch, 0.5)
    n = sp.shape[0]
    rest = miss = 0
    for t in range(start, min(n, f + horizon)):
        slow = vis[t] and np.isfinite(sp[t]) and sp[t] < 1.0
        gone = (not vis[t]) or out[t]
        rest = rest + 1 if slow else 0
        miss = miss + 1 if gone else 0
        if rest >= 25:
            return t - 24
        if miss >= 25:
            return t - 24
    return -1


def variants(inp, cfg):
    """{name: (confirmed bool array over inp['restarts'], cfg)}."""
    table = inp["restarts"]
    frames = table["frame"].to_numpy(dtype=np.int64)
    kinds = table["kind"].to_numpy(dtype=object)
    ball, feats = inp["ball"], inp["feats"]
    moving = moving_mask(feats)
    n = len(frames)
    zero = np.zeros(n)
    cur = window_confirm(frames, moving, zero, zero + 150)
    assert np.array_equal(cur, table["confirmed"].to_numpy())
    out = {"cur": cur}
    # A: displacement
    x, y, vis = ball["x"], ball["y"], ball["visible"]
    disp = np.zeros(n, dtype=bool)
    for i, f in enumerate(frames):
        seg = np.arange(f, min(len(x), f + 101))
        ok = vis[seg]
        if vis[f] and ok.any():
            disp[i] = np.hypot(x[seg][ok] - x[f], y[seg][ok] - y[f]).max() >= 10.0
    out["A"] = disp
    out["A_or"] = cur | disp
    # B: play continues
    ev = inp["events"]
    evf = ev[ev["kind"].isin(["touch", "release", "control", "dribble", "contested"])][
        "frame"].to_numpy()
    pitch = c.data.pitch_m()
    b = np.zeros(n, dtype=bool)
    peaks = np.array([c.peak_end(ball["speed"], int(f)) for f in frames])
    for i, f in enumerate(frames):
        s = stop_frame(ball, f, int(peaks[i]), pitch)
        if s < 0 or s > f + 100:
            b[i] = True
            continue
        later = evf[(evf > peaks[i] + 2) & (evf < s)]
        b[i] = bool(later.size)
    out["B"] = b
    # C: from the end of the rise
    out["C"] = window_confirm(frames, moving, zero, (peaks - frames) + 150)
    # D: from the shifted edge
    delta = np.where(kinds == "kick", cfg.DELTA_RESTART_KICK, cfg.DELTA_RESTART_REAPPEAR)
    out["D"] = window_confirm(frames, moving, delta, delta + 150)
    out["W8"] = window_confirm(frames, moving, zero, zero + 200)
    out["W10"] = window_confirm(frames, moving, zero, zero + 250)
    # H: chain to the next confirmed candidate
    h = cur.copy()
    conf_frames = frames[cur]
    for i in range(n):
        if h[i] or kinds[i] != "kick":
            continue
        nxt = conf_frames[conf_frames > frames[i]]
        h[i] = bool(nxt.size and nxt[0] - frames[i] <= CHAIN)
    out["H"] = h
    return out


def exits_compare(base, new, tol=25):
    """(kept_exact, kept_tol, removed, added) of two exit-frame lists."""
    b, nw = np.array(sorted(base)), np.array(sorted(new))
    kept_exact = len(set(base) & set(new))
    kept_tol = int(sum(np.any(np.abs(nw - f) <= tol) for f in b)) if nw.size else 0
    removed = [int(f) for f in b if not (nw.size and np.any(np.abs(nw - f) <= tol))]
    added = [int(f) for f in nw if not (b.size and np.any(np.abs(b - f) <= tol))]
    return kept_exact, kept_tol, removed, added


def main(write=False):
    cfg = Config.default()
    inp = c.load(cfg)
    cand, _, _, _ = c.classify(inp, cfg)
    feat = pd.read_csv(os.path.join(c.SCRATCH, "features.csv"))
    replay_like = set(feat[(feat["cls"] == "confirm_rej") & (feat["disp6"] < 10)
                           & ~(feat["t_all"] <= 150)]["frame"].astype(int))
    rej = set(feat[feat["cls"] == "confirm_rej"]["frame"].astype(int))
    vs = variants(inp, cfg)
    base_lab, base_parts = c.run_with_confirmed(inp, cfg, vs["cur"])
    base_ex = [f for f, _ in c.label_exits(base_lab, inp["blocks"])]
    base_sm = [f for f, _ in base_parts["restart_list"]]
    bp, bm, bs, bres = c.public_report(base_lab)
    base_pairs = {g[0] for g, _ in bp}
    frames = inp["restarts"]["frame"].to_numpy()
    rows = []
    runs = [(k, vs[k], cfg) for k in ("cur", "A", "A_or", "B", "C", "D", "W8", "W10", "H")]
    cfg9 = dataclasses.replace(cfg, DELTA_RESTART_KICK=9)
    runs += [("cur_k9", vs["cur"], cfg9), ("C_k9", vs["C"], cfg9), ("H_k9", vs["H"], cfg9),
             ("W8_k9", vs["W8"], cfg9)]
    for name, conf, cf in runs:
        lab, parts = c.run_with_confirmed(inp, cf, conf)
        ex = [f for f, _ in c.label_exits(lab, inp["blocks"])]
        sm = [f for f, _ in parts["restart_list"]]
        pairs, missed, spur, res = c.public_report(lab)
        pub = {g[0]: p[0] - g[0] for g, p in pairs}
        lost = sorted(base_pairs - set(pub))
        gained = sorted(set(pub) - base_pairs)
        if write:
            path, md5, st = c.write_and_sanity(lab, "rx_b_" + name)
        else:
            import sanity
            st, md5 = sanity.sanity_stats(lab), ""
        newly = set(frames[conf & ~vs["cur"]].astype(int))
        unconf = set(frames[~conf & vs["cur"]].astype(int))
        k_ex, k_tol, rem, add = exits_compare(base_ex, ex)
        k_sm, _, rem_sm, add_sm = exits_compare(base_sm, sm, tol=0)
        h = st["halves"]
        row = {
            "variant": name, "md5": md5,
            "n_confirmed": int(conf.sum()), "newly_confirmed": len(newly),
            "unconfirmed_vs_cur": len(unconf),
            "rej_confirmed": len(newly & rej),
            "replay_like_confirmed": sorted(newly & replay_like),
            "public_replays_confirmed": sorted(newly & set(REPLAYS_PUBLIC)),
            "sm_exits": len(sm), "sm_kept_exact": k_sm, "sm_removed": len(rem_sm),
            "sm_added": len(add_sm),
            "label_exits": len(ex), "label_kept_exact": k_ex, "label_kept_tol25": k_tol,
            "label_removed": rem, "label_added": add,
            "pub_lat": {g: pub.get(g) for g in c.PUBLIC_RESTARTS},
            "pub_matched": len(pairs), "pub_lost": lost, "pub_gained": gained,
            "spellF1": round(res["spell_f1_mean"], 4), "macroF1": round(res["macro_f1"], 4),
            "D1": round(100 * h[1]["d_share"], 2), "D2": round(100 * h[2]["d_share"], 2),
            "Dtot": round(100 * st["total"]["d_share"], 2),
            "Dseg": st["total"]["d_segments"], "poss": st["total"]["possessions"],
            "violations": st["violations"],
        }
        rows.append(row)
        print(json.dumps(row, default=int))
    with open(os.path.join(c.SCRATCH, "variants.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=int)


if __name__ == "__main__":
    main(write="--write" in sys.argv)

"""r2 task 3: the 18389 goal-kick case and the full-match exits on a kick of a fast ball.

    python tools/analysis/r2_restart_cases.py

Prints (1) a per-frame trace 18300..18520 of the 18355 kick: speed, dv_9, displacement from the
candidate, meanv_all, meanv_near, per-team mean speed, nearest H / A distance, and where the
150-frame confirmation window ends; (2) for every exit of the current state machine whose kick
starts on a ball already >= 3 m/s: the ball's path in the 4 s before (visibility, distance to the
nearest touchline, the frame the ball last started moving from < 2 m/s), Smart Tagging SET PIECES
type, the stoppage entry and the ball's path after. No ground truth.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_restart_common as c  # noqa: E402
from engine.config import Config  # noqa: E402


def trace_18389(inp):
    b, f = inp["ball"], inp["feats"]
    f0 = 18355
    print("== 18355 kick (GT restart 18389). window [18355, 18505]; columns: frame vis x y speed "
          "dv9 disp_from_18355 meanv_all meanv_near meanvH meanvA dH dA")
    for t in list(range(18300, 18424, 4)) + list(range(18424, 18524, 8)):
        d = np.hypot(b["x"][t] - b["x"][f0], b["y"][t] - b["y"][f0])
        mark = "  <- window end" if t <= 18505 < t + 8 and t >= 18424 else ""
        print("%d %d %5.1f %5.1f %5.2f %5.2f %5.1f %4.2f %4.2f %4.2f %4.2f %5.1f %5.1f%s" % (
            t, b["visible"][t], b["x"][t], b["y"][t], b["speed"][t], b["dv_9"][t], d,
            f["meanv_all"][t], f["meanv_near"][t], f["meanvH"][t], f["meanvA"][t], f["dH"][t],
            f["dA"][t], mark))
    mv = np.isfinite(f["meanv_all"]) & (np.nan_to_num(f["meanv_all"]) > c.V_STOP)
    first = f0 + int(np.flatnonzero(mv[f0:])[0])
    print("first meanv_all > 1.6 after 18355: %d (+%d from 18355, +%d from 18389)" % (
        first, first - f0, first - 18389))
    for name, lo, hi in (("18355..18505", 18355, 18506), ("18389..18539", 18389, 18540)):
        print("  max meanv_all %s = %.2f, max meanvH %.2f, max meanvA %.2f" % (
            name, np.nanmax(f["meanv_all"][lo:hi]), np.nanmax(f["meanvH"][lo:hi]),
            np.nanmax(f["meanvA"][lo:hi])))
    ev = inp["events"]
    print(ev[(ev["frame"] >= 18230) & (ev["frame"] < 18520)].to_string(index=False))


def fast_kick_exits(inp, cfg):
    cand, dead, rl, iv = c.classify(inp, cfg)
    b, f = inp["ball"], inp["feats"]
    feat = pd.read_csv(os.path.join(c.SCRATCH, "features.csv"))
    ex = feat[(feat["cls"] == "exit") & (feat["kind"] == "kick") & (feat["sp0"] >= 3)]
    tg = inp["tagging"]
    sp_rows = tg[tg["code"] == "SET PIECES"]
    width = c.data.pitch_m()[1]
    length = c.data.pitch_m()[0]
    print("\n== exits on a kick of a ball already >= 3 m/s: n=%d" % len(ex))
    for _, r in ex.iterrows():
        fr = int(r["frame"])
        lo = fr - 100
        vis = b["visible"][lo:fr + 1]
        sp = b["speed"][lo:fr + 1]
        # last start of motion: last frame before fr with speed < 2 m/s (visible)
        slow = np.flatnonzero(np.isfinite(sp) & (sp < 2.0))
        start_motion = lo + int(slow[-1]) if slow.size else None
        gap_end = None
        miss = np.flatnonzero(~vis)
        if miss.size:
            gap_end = lo + int(miss[-1]) + 1
        tl = min(b["y"][fr], width - b["y"][fr])
        gl = min(b["x"][fr], length - b["x"][fr])
        ref = start_motion if start_motion is not None else gap_end
        ref_xy = (b["x"][ref], b["y"][ref]) if ref is not None else (np.nan, np.nan)
        ref_tl = min(ref_xy[1], width - ref_xy[1]) if ref is not None else np.nan
        tag = sp_rows[(sp_rows["start_frame"] - 125 <= fr) & (fr < sp_rows["end_frame"] + 125)]
        tagtxt = "%s [%d,%d)" % (tag["type"].iloc[0], tag["start_frame"].iloc[0],
                                 tag["end_frame"].iloc[0]) if len(tag) else "-"
        entry = int(r["entry"])
        rej = feat[(feat["entry"] == entry) & (feat["cls"] == "confirm_rej")]["frame"].tolist()
        print("%6d sp0 %.1f dv9 %.1f | entry %d (%.1f s before) | ball %s; last <2 m/s at %s "
              "(%s), touchline %.1f m there, %.1f m at kick, goal line %.1f m | t_all %d | "
              "disp2/6 %.0f/%.0f m | tag %s | rejected in stoppage %s" % (
                  fr, r["sp0"], r["dv9"], entry, (fr - entry) / 25.0,
                  "seen throughout" if vis.all() else "gap ends %s" % gap_end,
                  start_motion, "%+d" % (start_motion - fr) if start_motion else "-", ref_tl,
                  tl, gl, r["t_all"], r["disp2"], r["disp6"], tagtxt, rej))


def out_entry_reappears(inp, cfg):
    """Exits whose stoppage entry is an `out` onset on (+-1 frame of) a reappear candidate:
    is the reappearing ball outside the pitch and heading in (a throw-in / corner being taken)?"""
    cand, dead, rl, iv = c.classify(inp, cfg)
    b, f = inp["ball"], inp["feats"]
    length, width = c.data.pitch_m()
    on = inp["onsets"]
    outs = set(on[on["mechanism"] == "out"]["frame"].astype(int))
    reap = inp["restarts"][inp["restarts"]["kind"] == "reappear"]["frame"].astype(int).tolist()
    tg = inp["tagging"]
    sp_rows = tg[tg["code"] == "SET PIECES"]
    print("\n== exits whose entry is an out onset on a reappear frame")
    n_in = 0
    for e, l, r in iv:
        if r < 0 or e not in outs:
            continue
        rf = [x for x in reap if abs(x - e) <= 1]
        if not rf:
            continue
        t = rf[0]
        x, y, vx, vy = b["x"][t], b["y"][t], b["vx"][t], b["vy"][t]
        # inward = velocity points toward the pitch centre
        inward = (vx * (length / 2 - x) + vy * (width / 2 - y)) > 0
        outside = x < 0 or x > length or y < 0 or y > width
        mv = np.isfinite(f["meanv_all"][t:l]) & (np.nan_to_num(f["meanv_all"][t:l]) > c.V_STOP)
        first_mv = int(np.flatnonzero(mv)[0]) if mv.any() else -1
        tag = sp_rows[(sp_rows["start_frame"] - 125 <= t) & (t < sp_rows["end_frame"] + 125)]
        n_in += int(outside and inward)
        print("  entry %d exit %d (D %d frames): ball (%.1f, %.1f) %s, speed %.1f, %s; players "
              "move +%s; tag %s" % (e, l, l - e, x, y, "outside" if outside else "inside",
                                    b["speed"][t], "inward" if inward else "outward", first_mv,
                                    tag["type"].iloc[0] if len(tag) else "-"))
    print("  outside and heading inward: %d" % n_in)


def main():
    cfg = Config.default()
    inp = c.load(cfg)
    trace_18389(inp)
    fast_kick_exits(inp, cfg)
    out_entry_reappears(inp, cfg)


if __name__ == "__main__":
    main()

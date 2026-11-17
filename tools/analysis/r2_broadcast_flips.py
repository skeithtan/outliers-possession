"""R2-a step 3: the four public H<->A flips with no new-team player within 1.5 m of the ATD ball.

For each flip f (11351, 11553, 13785, 15766): the public GT team before/after, and over
[f-60, f+60] the tactical ball (xy, speed), the broadcast ball (xy, both with and without the
replay mask), their distance, and the nearest H and A player to each ball. Summary per case:
minimum new-team distance to each ball in [f-40, f+6], any player within 3 m of either ball,
airborne by speed (> 15 m/s), and tactical track stuck (speed < 0.3 m/s for >= 25 frames).
Writes the per-frame table to <out_dir>/flip_<f>.csv (scratch only). Uses the public GT only.
Usage: python tools/analysis/r2_broadcast_flips.py <out_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE)))  # tools/
import r2_broadcast_common as C  # noqa: E402
from gt import public_gt_codes  # noqa: E402

OUT = sys.argv[1]
FLIPS = [11351, 11553, 13785, 15766]
LET = "HADX"
d = C.load(OUT)
gt = public_gt_codes()
tx, ty = d["ball"]["x"], d["ball"]["y"]
tv = d["ball"]["visible"].astype(bool) & np.isfinite(tx)
tx, ty = np.where(tv, tx, np.nan), np.where(tv, ty, np.nan)
tsp = d["ball"]["speed"]
bx, by = d["bb"]["bb_x"], d["bb"]["bb_y"]
rx, ry = d["bb"]["bb_raw_x"], d["bb"]["bb_raw_y"]
labels = d["labels"]
dT, sT = C.nearest_per_team(d["px"], d["py"], d["team_of_slot"], tx, ty)
dB, sB = C.nearest_per_team(d["px"], d["py"], d["team_of_slot"], bx, by)
dR, sR = C.nearest_per_team(d["px"], d["py"], d["team_of_slot"], rx, ry)
bsp = np.full(len(bx), np.nan)
bsp[1:-1] = np.hypot(bx[2:] - bx[:-2], by[2:] - by[:-2]) * 12.5

for f in FLIPS:
    before, after = LET[gt[f - 1]], LET[gt[f]]
    new = "HA".index(after)
    w = np.arange(f - 60, f + 61)
    tab = pd.DataFrame({
        "frame": w, "rel": w - f, "gt": [LET[gt[i]] for i in w], "engine": [str(labels[i]) for i in w],
        "Tx": tx[w], "Ty": ty[w], "Tspeed": tsp[w], "Bx": bx[w], "By": by[w], "Bspeed": bsp[w],
        "Braw_x": rx[w], "Braw_y": ry[w], "chamfer": d["bb"]["chamfer"][w],
        "T_B_dist": np.hypot(tx[w] - bx[w], ty[w] - by[w]),
        "T_dH": dT[w, 0], "T_sH": sT[w, 0], "T_dA": dT[w, 1], "T_sA": sT[w, 1],
        "B_dH": dB[w, 0], "B_sH": sB[w, 0], "B_dA": dB[w, 1], "B_sA": sB[w, 1],
        "Braw_dH": dR[w, 0], "Braw_dA": dR[w, 1]})
    tab.to_csv(os.path.join(OUT, "flip_%d.csv" % f), index=False, float_format="%.2f")
    print("\n=== flip %d: GT %s -> %s (new team %s) ===" % (f, before, after, after))
    print("every 5th frame (rel, Tx, Ty, Tspd, Bx, By, |T-B|, T:dH/sH dA/sA, B:dH/sH dA/sA, chamfer):")
    for _, r in tab[tab["rel"] % 5 == 0].iterrows():
        print("  %+4d  T(%6.1f,%5.1f) %5.1f  B(%6.1f,%5.1f) |%5.1f|  T: H %5.1f/%2d A %5.1f/%2d  B: H %5.1f/%2d A %5.1f/%2d  ch %.1f"
              % (r.rel, r.Tx, r.Ty, r.Tspeed, r.Bx, r.By, r.T_B_dist, r.T_dH, r.T_sH, r.T_dA, r.T_sA,
                 r.B_dH, r.B_sH, r.B_dA, r.B_sA, r.chamfer))
    s = tab[(tab.rel >= -40) & (tab.rel <= 6)]
    nk = "H" if new == 0 else "A"
    print("summary [-40,+6]: T visible %d/47, B visible %d/47 (raw %d/47)"
          % (np.isfinite(s.Tx).sum(), np.isfinite(s.Bx).sum(), np.isfinite(s.Braw_x).sum()))
    print("  min new-team (%s) dist to T ball %.2f m (frame %s); to B ball %.2f m (frame %s); to raw B %.2f m"
          % (nk, np.nanmin(s["T_d" + nk]), s.frame.values[np.nanargmin(s["T_d" + nk].values)] if np.isfinite(s["T_d" + nk]).any() else "-",
             np.nanmin(s["B_d" + nk]) if np.isfinite(s["B_d" + nk]).any() else np.nan,
             s.frame.values[np.nanargmin(s["B_d" + nk].values)] if np.isfinite(s["B_d" + nk]).any() else "-",
             np.nanmin(s["Braw_d" + nk]) if np.isfinite(s["Braw_d" + nk]).any() else np.nan))
    print("  new-team within 1.5 m of B ball on %d frames; of T ball on %d frames"
          % ((s["B_d" + nk] <= 1.5).sum(), (s["T_d" + nk] <= 1.5).sum()))
    full = tab
    anyT = np.fmin(full.T_dH, full.T_dA)
    anyB = np.fmin(full.B_dH, full.B_dA)
    print("  [-60,+60]: any player within 3 m of T ball on %d frames, of B ball on %d frames; new-team within 3 m of T %d, of B %d"
          % ((anyT <= 3).sum(), (anyB <= 3).sum(), (full["T_d" + nk] <= 3).sum(), (full["B_d" + nk] <= 3).sum()))
    print("  T speed max %.1f m/s (frames > 15 m/s: %d), B speed max %.1f; |T-B| median %.2f max %.2f m"
          % (np.nanmax(full.Tspeed), (full.Tspeed > 15).sum(), np.nanmax(full.Bspeed),
             np.nanmedian(full.T_B_dist), np.nanmax(full.T_B_dist)))
    box = np.hypot(np.nanmax(s.Tx) - np.nanmin(s.Tx), np.nanmax(s.Ty) - np.nanmin(s.Ty))
    print("  T ball in [-40,+6]: bounding-box diagonal %.2f m, median speed %.1f m/s, nearest player of either team min %.2f m"
          " (stuck = diagonal < 2 m with no player within 3 m); T missing frames %d/121"
          % (box, np.nanmedian(s.Tspeed), np.nanmin(np.fmin(s.T_dH, s.T_dA)), np.isnan(full.Tx).sum()))
    ch = np.flatnonzero(np.diff(gt[f - 300:f + 300])) + f - 299
    print("  public GT changes within +-300:", [(int(c), LET[gt[c - 1]], LET[gt[c]]) for c in ch])
    ch = np.flatnonzero(labels[f - 300:f + 300][1:] != labels[f - 300:f + 300][:-1]) + f - 299
    print("  engine changes within +-300:", [(int(c), str(labels[c - 1]), str(labels[c])) for c in ch])

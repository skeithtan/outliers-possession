"""R2-a step 2: agreement of the tactical (engine-cleaned) ball and the aligned broadcast ball.

Graded frames only, by half. Distance distribution where both are visible; frames with only one
ball; and the same with the broadcast replay mask off (bb_raw). Also the distance by chamfer bin.
Usage: python tools/analysis/r2_broadcast_agree.py <out_dir>
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_broadcast_common as C  # noqa: E402

OUT = sys.argv[1]
d = C.load(OUT)
tx, ty, tvis = d["ball"]["x"], d["ball"]["y"], d["ball"]["visible"].astype(bool)
tvis = tvis & np.isfinite(tx)
bb = d["bb"]


def report(name, bx, by):
    bvis = np.isfinite(bx)
    print("--- broadcast ball: %s ---" % name)
    for h in (1, 2, 0):
        g = d["graded"] if h == 0 else (d["half"] == h)
        both = g & tvis & bvis
        dist = np.hypot(tx[both] - bx[both], ty[both] - by[both])
        print("%s: graded %d | both %d (%.1f%%) | T only %d (%.1f%%) | B only %d (%.1f%%) | neither %d (%.1f%%)"
              % ("all " if h == 0 else "H%d  " % h, g.sum(), both.sum(), 100 * both.sum() / g.sum(),
                 (g & tvis & ~bvis).sum(), 100 * (g & tvis & ~bvis).sum() / g.sum(),
                 (g & ~tvis & bvis).sum(), 100 * (g & ~tvis & bvis).sum() / g.sum(),
                 (g & ~tvis & ~bvis).sum(), 100 * (g & ~tvis & ~bvis).sum() / g.sum()))
        print("      |T-B| p50 %.2f p90 %.2f p99 %.2f m; >3 m %.1f%%; >5 m %.1f%%; >10 m %.1f%%"
              % (np.percentile(dist, 50), np.percentile(dist, 90), np.percentile(dist, 99),
                 100 * np.mean(dist > 3), 100 * np.mean(dist > 5), 100 * np.mean(dist > 10)))


report("replay-masked (bb_x)", bb["bb_x"], bb["bb_y"])
report("valid homography, no replay mask (bb_raw_x)", bb["bb_raw_x"], bb["bb_raw_y"])

print("--- |T-B| by per-frame player chamfer (graded, both visible, no replay mask) ---")
bx, by = bb["bb_raw_x"], bb["bb_raw_y"]
both = d["graded"] & tvis & np.isfinite(bx)
dist = np.hypot(tx - bx, ty - by)
ch = bb["chamfer"]
for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 6), (6, 1e9)):
    s = both & (ch >= lo) & (ch < hi)
    if s.sum():
        print("  chamfer [%g,%g) m: n=%d |T-B| p50 %.2f p90 %.2f >3 m %.1f%%"
              % (lo, hi, s.sum(), np.median(dist[s]), np.percentile(dist[s], 90), 100 * np.mean(dist[s] > 3)))
s = both & ~np.isfinite(ch)
print("  chamfer NaN: n=%d" % s.sum())

print("--- alive vs stoppage (engine stoppage mask), replay-masked ---")
bx, by = bb["bb_x"], bb["bb_y"]
for name, m in (("alive", ~d["stoppage"]), ("stoppage", d["stoppage"])):
    s = d["graded"] & tvis & np.isfinite(bx) & m
    dd = np.hypot(tx[s] - bx[s], ty[s] - by[s])
    print("  %s: both %d, p50 %.2f p90 %.2f >3 m %.1f%% >5 m %.1f%%"
          % (name, s.sum(), np.median(dd), np.percentile(dd, 90), 100 * np.mean(dd > 3), 100 * np.mean(dd > 5)))
    g = d["graded"] & m
    print("     B only %d (%.1f%% of %s graded), T only %d"
          % ((g & ~tvis & np.isfinite(bx)).sum(), 100 * (g & ~tvis & np.isfinite(bx)).sum() / g.sum(), name,
             (g & tvis & ~np.isfinite(bx)).sum()))

print("--- disagreement episodes (|T-B| > 3 m, replay-masked, graded) ---")
bx, by = bb["bb_x"], bb["bb_y"]
both = d["graded"] & tvis & np.isfinite(bx)
dist = np.hypot(tx - bx, ty - by)
txm, tym = np.where(tvis, tx, np.nan), np.where(tvis, ty, np.nan)
dT, _ = C.nearest_per_team(d["px"], d["py"], d["team_of_slot"], txm, tym)
dB, _ = C.nearest_per_team(d["px"], d["py"], d["team_of_slot"], bx, by)
mT, mB = np.fmin(dT[:, 0], dT[:, 1]), np.fmin(dB[:, 0], dB[:, 1])
bsp = np.full(len(bx), np.nan)
bsp[1:-1] = np.hypot(bx[2:] - bx[:-2], by[2:] - by[:-2]) * 12.5
tsp = d["ball"]["speed"]
for name, s in (("|T-B| < 1", both & (dist < 1)), ("1-3", both & (dist >= 1) & (dist < 3)), ("> 3", both & (dist > 3))):
    print("  %s m: n=%d, nearest player to T p50 %.2f, to B p50 %.2f; T speed p50 %.1f, B speed p50 %.1f m/s"
          % (name, s.sum(), np.nanmedian(mT[s]), np.nanmedian(mB[s]), np.nanmedian(tsp[s]), np.nanmedian(bsp[s])))
s = both & (dist > 3)
print("  > 3 m: B ball nearer a player than T ball on %.0f%%; T speed > 15 on %.0f%%, B speed > 15 on %.0f%%; dy=B-T < 0 on %.0f%% (median dy %.1f m)"
      % (100 * np.nanmean(mB[s] < mT[s]), 100 * np.nanmean(tsp[s] > 15), 100 * np.nanmean(bsp[s] > 15),
         100 * np.mean((by - ty)[s] < 0), np.median((by - ty)[s])))
p = np.concatenate(([0], s.astype(int), [0]))
e = np.flatnonzero(np.diff(p))
st, en = e[::2], e[1::2]
ln = en - st
fixed = 0
for a, b in zip(st, en):
    best = np.nanmedian(dist[a:b])
    for lag in range(-25, 26):
        if lag == 0 or a + lag < 0 or b + lag > len(bx):
            continue
        idx = np.arange(a, b)
        dd = np.hypot(tx[idx] - bx[idx + lag], ty[idx] - by[idx + lag])
        if np.isfinite(dd).sum() >= 0.8 * (b - a):
            best = min(best, np.nanmedian(dd))
    fixed += best < 1.5
print("  episodes %d, length p50 %d p90 %d frames; fixed (median < 1.5 m) by a lag within +-25 frames: %d"
      % (len(ln), np.median(ln), np.percentile(ln, 90), fixed))

"""R2-a step 1: align the broadcast ATD to the tactical index and save the broadcast ball on it.

Method (re-derived, not hard-coded):
  1. Coarse offset per half: cross-correlate the 1 Hz team-centroid tracks (x and y of each team)
     of tactical and broadcast over every integer-second offset, for the 4 mirror hypotheses
     (m0 none, m1 x -> 1-x, m2 y -> 1-y, m3 both). Report the best and second-best peaks.
  2. Fine offset per half: median tactical-broadcast ball distance (both visible, broadcast
     homography valid) for offsets coarse +- 50 frames, under each mirror hypothesis.
  3. Drift check: best fine offset inside each 5-minute window of the graded frames.
  4. Replay check: per-frame one-sided chamfer (broadcast players -> tactical players, per team)
     at the constant offset. Frames with chamfer > CHAMFER_MAX m are marked `replay`.
Output: <OUT>/bb_on_tactical.npz with bb_x, bb_y (metres, NaN when missing / invalid / replay),
plus bb_raw_x/y (no replay mask), b_frame (broadcast frame or -1), chamfer, valid flags.
Deterministic: no randomness. Summary statistics printed only.
Usage: python tools/analysis/r2_broadcast_align.py <out_dir>
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine import data  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
L, W = data.pitch_m()
FPS = 25
CHAMFER_MAX = 3.0  # m; at the right offset the live-camera chamfer is ~0.5 m (PLAN 1.5)

T = data.load_tactical()
B = data.load_broadcast()
NT, NB = T["n_frames"], B["n_frames"]
blocks = data.graded_blocks()
# Half of a tactical frame: split at the middle of the half-time gap.
split = (blocks[0][2] + blocks[1][1]) // 2


def mirror_xy(x, y, m):
    if m in (1, 3):
        x = 1 - x
    if m in (2, 3):
        y = 1 - y
    return x, y


def centroids(src, m=0):
    """(N, 4) per-frame centroid x_H, y_H, x_A, y_A in metres (NaN if < 5 players)."""
    px, py = mirror_xy(src["px"], src["py"], m)
    out = np.full((src["n_frames"], 4), np.nan)
    for t in (0, 1):
        cols = src["team_of_slot"] == t
        vis = np.isfinite(px[:, cols])
        n = vis.sum(1)
        with np.errstate(all="ignore"):
            cx = np.nanmean(px[:, cols], 1) * L
            cy = np.nanmean(py[:, cols], 1) * W
        ok = n >= 5
        out[ok, 2 * t] = cx[ok]
        out[ok, 2 * t + 1] = cy[ok]
    return out


def per_second(c):
    n = c.shape[0] // FPS
    with np.errstate(all="ignore"):
        return np.nanmean(c[: n * FPS].reshape(n, FPS, 4), 1)


print("=== 1. coarse offset: 1 Hz team-centroid match (mean abs centroid error, m) ===")
Tc = per_second(centroids(T))
coarse = {}
for half, lo, hi in blocks:
    tsec = np.arange(lo // FPS + 1, hi // FPS)
    res = []
    for m in range(4):
        Bc = per_second(centroids(B, m))
        best = []
        for off_s in range(-Bc.shape[0], Tc.shape[0]):
            bsec = tsec - off_s
            ok = (bsec >= 0) & (bsec < Bc.shape[0])
            if ok.sum() < 0.5 * len(tsec):
                continue
            e = np.abs(Tc[tsec[ok]] - Bc[bsec[ok]])
            good = np.isfinite(e).all(1)
            if good.sum() < 300:
                continue
            best.append((np.mean(e[good]), off_s, good.sum()))
        best.sort()
        top = best[0]
        second = [b for b in best if abs(b[1] - top[1]) > 10][0]
        res.append((top[0], m, top[1], top[2], second[0], second[1]))
        print("half %d mirror m%d: best offset %6d s err %.2f m (n=%d s); 2nd peak %6d s err %.2f m (ratio %.1f)"
              % (half, m, top[1], top[0], top[2], second[1], second[0], second[0] / top[0]))
    res.sort()
    coarse[half] = (res[0][1], res[0][2] * FPS)
    print("half %d: chosen mirror m%d, coarse offset %d frames" % (half, res[0][1], res[0][2] * FPS))

print("\n=== 2. fine offset: ball agreement (median |T - B| m, both visible, homography valid) ===")
Tbx = T["bx"].astype(float) * L
Tby = T["by"].astype(float) * W
fine = {}
for half, lo, hi in blocks:
    m0, c = coarse[half]
    t = np.arange(lo, hi + 1)
    out = []
    for m in range(4):
        bxn, byn = mirror_xy(B["bx"].astype(float), B["by"].astype(float), m)
        bxm, bym = bxn * L, byn * W
        bestm = None
        for off in range(c - 50, c + 51):
            b = t - off
            ok = (b >= 0) & (b < NB)
            tt, bb = t[ok], b[ok]
            both = np.isfinite(Tbx[tt]) & np.isfinite(bxm[bb]) & B["valid"][bb]
            if both.sum() < 1000:
                continue
            e = np.hypot(Tbx[tt][both] - bxm[bb][both], Tby[tt][both] - bym[bb][both])
            r = (np.median(e), off, both.sum(), np.mean(e < 2))
            if bestm is None or r[0] < bestm[0]:
                bestm = r
        out.append((bestm[0], m, bestm[1], bestm[2], bestm[3]))
        print("half %d mirror m%d: best offset %d median %.2f m (<2 m %.0f%% of %d)"
              % (half, m, bestm[1], bestm[0], 100 * bestm[3], bestm[2]))
    out.sort()
    fine[half] = (out[0][1], out[0][2])
    # neighbours of the best offset, mirror chosen
    m, off = fine[half]
    bxn, byn = mirror_xy(B["bx"].astype(float), B["by"].astype(float), m)
    line = []
    for o in range(off - 3, off + 4):
        b = t - o
        ok = (b >= 0) & (b < NB)
        tt, bb = t[ok], b[ok]
        both = np.isfinite(Tbx[tt]) & np.isfinite(bxn[bb]) & B["valid"][bb]
        e = np.hypot(Tbx[tt][both] - bxn[bb][both] * L, Tby[tt][both] - byn[bb][both] * W)
        line.append("%d:%.2f" % (o, np.median(e)))
    print("half %d: chosen m%d offset %d; neighbourhood medians %s" % (half, m, off, " ".join(line)))

print("\n=== 3. drift: best offset per 5-minute window (ball median) ===")
for half, lo, hi in blocks:
    m, off = fine[half]
    bxn, byn = mirror_xy(B["bx"].astype(float), B["by"].astype(float), m)
    offs = []
    for wlo in range(lo, hi + 1, 7500):
        t = np.arange(wlo, min(wlo + 7500, hi + 1))
        best = None
        for o in range(off - 25, off + 26):
            b = t - o
            ok = (b >= 0) & (b < NB)
            tt, bb = t[ok], b[ok]
            both = np.isfinite(Tbx[tt]) & np.isfinite(bxn[bb]) & B["valid"][bb]
            if both.sum() < 200:
                continue
            e = np.median(np.hypot(Tbx[tt][both] - bxn[bb][both] * L, Tby[tt][both] - byn[bb][both] * W))
            if best is None or e < best[0]:
                best = (e, o, both.sum())
        if best:
            offs.append(best[1])
            print("  half %d window %6d-%6d: best %d (median %.2f m, n=%d)" % (half, t[0], t[-1], best[1], best[0], best[2]))
    print("  half %d: window offsets min %d max %d" % (half, min(offs), max(offs)))

print("\n=== 4. build arrays on the tactical index + replay check ===")
b_frame = np.full(NT, -1, np.int64)
mir = np.zeros(NT, np.int8)
for half, lo, hi in blocks:
    m, off = fine[half]
    t = np.arange(0, split) if half == 1 else np.arange(split, NT)
    b = t - off
    ok = (b >= 0) & (b < NB)
    b_frame[t[ok]] = b[ok]
    mir[t] = m
has = b_frame >= 0
bi = np.where(has, b_frame, 0)
bxn = B["bx"][bi].astype(float)
byn = B["by"][bi].astype(float)
pxB = B["px"][bi].astype(float)
pyB = B["py"][bi].astype(float)
for m in range(4):
    sel = mir == m
    if sel.any():
        bxn[sel], byn[sel] = mirror_xy(bxn[sel], byn[sel], m)
        pxB[sel], pyB[sel] = mirror_xy(pxB[sel], pyB[sel], m)
valid = has & B["valid"][bi]
raw_x = np.where(has & np.isfinite(bxn), bxn * L, np.nan)
raw_y = np.where(has & np.isfinite(byn), byn * W, np.nan)

# one-sided chamfer per frame, per team, mean of the two teams
pxT = T["px"].astype(float) * L
pyT = T["py"].astype(float) * W
pxB *= L
pyB *= W
cham = np.full(NT, np.nan)
CH = 20000
parts = []
for t in (0, 1):
    cT = np.flatnonzero(T["team_of_slot"] == t)
    cB = np.flatnonzero(B["team_of_slot"] == t)
    res = np.full(NT, np.nan)
    for s in range(0, NT, CH):
        e = min(s + CH, NT)
        dx = pxB[s:e, cB][:, :, None] - pxT[s:e, cT][:, None, :]
        dy = pyB[s:e, cB][:, :, None] - pyT[s:e, cT][:, None, :]
        d = np.hypot(dx, dy)
        with np.errstate(all="ignore"):
            dm = np.nanmin(np.where(np.isfinite(d), d, np.inf), 2)
        dm[~np.isfinite(dm)] = np.nan
        nb = np.isfinite(dm).sum(1)
        with np.errstate(all="ignore"):
            mm = np.nanmean(dm, 1)
        mm[nb < 3] = np.nan
        res[s:e] = mm
    parts.append(res)
with np.errstate(all="ignore"):
    cham = np.nanmean(np.stack(parts), 0)
cham[~has] = np.nan

graded = np.zeros(NT, bool)
for _, lo, hi in blocks:
    graded[lo:hi + 1] = True
okc = graded & valid & np.isfinite(cham)
print("chamfer on graded valid frames: p50 %.2f p90 %.2f p99 %.2f m; share > %.0f m: %.2f%%"
      % (np.percentile(cham[okc], 50), np.percentile(cham[okc], 90), np.percentile(cham[okc], 99),
         CHAMFER_MAX, 100 * np.mean(cham[okc] > CHAMFER_MAX)))
replay = valid & np.isfinite(cham) & (cham > CHAMFER_MAX)
use = valid & ~replay
bb_x = np.where(use, raw_x, np.nan)
bb_y = np.where(use, raw_y, np.nan)
print("graded frames: mapped %.1f%%, homography valid %.1f%%, broadcast ball (valid, not replay) %.1f%%, replay-flagged %.2f%%"
      % (100 * has[graded].mean(), 100 * valid[graded].mean(), 100 * np.isfinite(bb_x[graded]).mean(),
         100 * replay[graded].mean()))
path = os.path.join(OUT, "bb_on_tactical.npz")
np.savez_compressed(path, bb_x=bb_x, bb_y=bb_y, bb_raw_x=np.where(valid, raw_x, np.nan),
                    bb_raw_y=np.where(valid, raw_y, np.nan), b_frame=b_frame, valid=valid,
                    chamfer=cham, replay=replay,
                    offsets=np.array([fine[1][1], fine[2][1]]), mirrors=np.array([fine[1][0], fine[2][0]]))
print("saved", path)

"""With piecewise-constant offsets, how much of the tactical ball gap does the broadcast ball fill?"""
import os, sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "cache", "atd_tactical.npz")
sys.path.insert(0, ROOT)
from games import asset_paths
L, W = 105.3, 68.0

p = asset_paths(source="broadcast")
bx, by = [], []
with open(p["atd_raw"]) as fh:
    for line in fh:
        b = line.rstrip("\n").rsplit(":", 1)[1].split(",")
        bx.append(float(b[0])); by.append(float(b[1]))
Bbx, Bby = np.array(bx), np.array(by)
hom = pd.read_csv(p["homography"], usecols=["valid", "reprojection_error"])
Bvalid = hom["valid"].values.astype(bool)
print("broadcast ball visible %.1f%%, valid homography %.1f%%, ball visible & valid %.1f%%" % (
    100 * np.mean(~np.isnan(Bbx)), 100 * Bvalid.mean(), 100 * np.mean(~np.isnan(Bbx) & Bvalid)))

d = np.load(CACHE)
Tbx, Tby = d["bx"].astype(float), d["by"].astype(float)
N = len(Tbx)

def mapped(off, lo, hi):
    t = np.arange(lo, hi)
    b = t - off
    ok = (b >= 0) & (b < len(Bbx))
    return t[ok], b[ok]

# refine offsets by ball agreement where both visible
for half, (lo, hi), guess in ((1, (11185, 90652), 4014), (2, (110808, 190198), 26919)):
    best = None
    for off in range(guess - 6, guess + 7):
        t, b = mapped(off, lo, hi)
        both = ~np.isnan(Tbx[t]) & ~np.isnan(Bbx[b])
        err = np.sqrt(((Tbx[t][both] - Bbx[b][both]) * L) ** 2 + ((Tby[t][both] - Bby[b][both]) * W) ** 2)
        med = np.median(err)
        if best is None or med < best[1]:
            best = (off, med, np.mean(err < 2.0), both.sum())
    off, med, frac2, n = best
    print("\nhalf %d: best offset %d  median ball disagreement %.2f m  (<2m on %.0f%% of %d co-visible frames)" % (
        half, off, med, 100 * frac2, n))
    t, b = mapped(off, lo, hi)
    tgap = np.isnan(Tbx[t])
    bfill = ~np.isnan(Bbx[b]) & Bvalid[b]
    print("  tactical graded frames %d, ball missing %d (%.1f%%)" % (len(t), tgap.sum(), 100 * tgap.mean()))
    print("  of the missing, broadcast ball present (valid homography) on %d (%.1f%%)" % (
        (tgap & bfill).sum(), 100 * (tgap & bfill).sum() / tgap.sum()))
    # distribution of remaining gap lengths after fill
    still = tgap & ~bfill
    edges = np.diff(np.concatenate(([0], still.astype(int), [0])))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    print("  remaining gaps: n=%d median=%d p90=%d max=%d frames; frames in gaps >2s: %.1f%% of graded" % (
        len(runs), np.median(runs), np.percentile(runs, 90), runs.max(), 100 * runs[runs > 50].sum() / len(t)))
    # Does the broadcast see the ball when tactical doesn't, in a *different* way (e.g. constant replays)? Check
    # ball-vs-player consistency: speed of the broadcast ball at filled frames
    bb = b[tgap & bfill]
    sp = np.sqrt((np.diff(Bbx[bb]) * L) ** 2 + (np.diff(Bby[bb]) * W) ** 2) * 25
    print("  broadcast-fill ball speed median %.1f m/s (sanity, should be football-like)" % np.median(sp[np.diff(bb) == 1]))

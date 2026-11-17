"""Ground the plan: what does the public GT look like against the ATD? Not part of the repo."""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "cache", "atd_tactical.npz")
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/starter"); sys.path.insert(0, ROOT + "/evaluation")
import load_smart_tagging as st

d = np.load(CACHE)
px, py, ps, bx, by, bs, team = d["px"], d["py"], d["ps"], d["bx"], d["by"], d["bs"], d["team_of_slot"]
L, W = 105.3, 68.0
N = len(bx)

gt = pd.read_csv(ROOT + "/evaluation/gt/BAR-ATH_possession_gt_public.csv")["label"].values
win = np.arange(11185, 18685)
g = gt[win]

# --- segments
def segs(labels, offset):
    br = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    b = np.concatenate(([0], br, [len(labels)]))
    return [(offset + b[i], offset + b[i + 1], labels[b[i]]) for i in range(len(b) - 1)]

S = segs(g, 11185)
print("GT segments in public window (%d):" % len(S))
for a, b, l in S:
    print("  %6d-%6d  %s  %5.1fs" % (a, b, l, (b - a) / 25))

# --- ball visibility by label
vis = ~np.isnan(bx)
print("\nball visible by GT label:")
for l in "HAD":
    m = g == l
    print("  %s: %5d frames, ball visible %.1f%%" % (l, m.sum(), 100 * vis[win][m].mean()))

# --- gap lengths of missing ball inside window
miss = ~vis[win]
runs = segs(miss.astype(int), 0)
gaps = [(b - a) for a, b, l in runs if l == 1]
print("\nmissing-ball runs in window: n=%d  median=%d  p90=%d  max=%d frames" % (
    len(gaps), np.median(gaps), np.percentile(gaps, 90), max(gaps)))
print("  frames in gaps <=12: %d, 13-50: %d, >50: %d" % (
    sum(x for x in gaps if x <= 12), sum(x for x in gaps if 12 < x <= 50), sum(x for x in gaps if x > 50)))

# --- nearest player team & distance
dx = (px - bx[:, None]) * L
dy = (py - by[:, None]) * W
dist = np.sqrt(dx ** 2 + dy ** 2)  # N x 46, NaN where either missing
dist_home = np.where(team[None, :] == 0, dist, np.nan)
dist_away = np.where(team[None, :] == 1, dist, np.nan)
with np.errstate(all="ignore"):
    dmin_h = np.nanmin(dist_home, axis=1)
    dmin_a = np.nanmin(dist_away, axis=1)
nearest_team = np.where(np.isnan(dmin_h) & np.isnan(dmin_a), -1,
                        np.where(np.nan_to_num(dmin_h, nan=1e9) <= np.nan_to_num(dmin_a, nan=1e9), 0, 1))
nearest_lab = np.array(["H", "A"])[np.clip(nearest_team, 0, 1)]
dmin = np.fmin(np.nan_to_num(dmin_h, nan=1e9), np.nan_to_num(dmin_a, nan=1e9))

alive = (g == "H") | (g == "A")
m = alive & vis[win]
print("\nnearest-player team == GT team on alive frames with ball visible: %.1f%% (%d frames)" % (
    100 * (nearest_lab[win][m] == g[m]).mean(), m.sum()))
# by nearest distance bucket
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)]:
    mm = m & (dmin[win] >= lo) & (dmin[win] < hi)
    if mm.sum():
        print("  dmin in [%g,%g): %5d frames, agree %.1f%%" % (lo, hi, mm.sum(), 100 * (nearest_lab[win][mm] == g[mm]).mean()))

# --- dead-ball characteristics
print("\nDead ball frames: ball position / speed / players")
D = g == "D"
bxw, byw = bx[win] * L, by[win] * W
out = (bxw < 0) | (bxw > L) | (byw < 0) | (byw > W)
print("  ball visible & out of pitch bounds on D frames: %.1f%% of visible D frames" % (100 * out[D & vis[win]].mean()))
print("  ball visible & out of bounds on alive frames: %.1f%%" % (100 * out[alive & vis[win]].mean()))
print("  ball speed median D=%.2f alive=%.2f" % (np.nanmedian(bs[win][D]), np.nanmedian(bs[win][alive])))
nplayers = (~np.isnan(px)).sum(axis=1)
print("  n tracked players mean D=%.1f alive=%.1f" % (nplayers[win][D].mean(), nplayers[win][alive].mean()))
mean_speed = np.nanmean(ps, axis=1)
print("  mean player speed D=%.2f alive=%.2f" % (np.nanmean(mean_speed[win][D]), np.nanmean(mean_speed[win][alive])))
# team centroid spread
for l in "HAD":
    mm = g == l
    hx = np.nanmean(np.where(team[None, :] == 0, px, np.nan), axis=1)[win][mm]
    ax = np.nanmean(np.where(team[None, :] == 1, px, np.nan), axis=1)[win][mm]
    print("  %s: team x-centroid sep (m) mean %.1f" % (l, np.nanmean(np.abs(hx - ax)) * L))

# --- around GT transitions: when does nearest-team flip? what is ball speed / dmin?
print("\nTransitions (GT) and the raw nearest-team signal around them (frames rel. to GT change):")
trans = [(S[i][0], S[i - 1][2], S[i][2]) for i in range(1, len(S))]
for f, a, b in trans:
    rel = np.arange(-50, 51)
    idx = f + rel
    nt = nearest_lab[idx].copy()
    nt[~vis[idx]] = "."
    s = "".join(nt)
    print("  %6d %s->%s | %s | %s" % (f, a, b, s[:50], s[50:]))
    # first frame after transition where nearest team == new team for 5 consecutive frames (ball visible)
    if b in "HA":
        ok = (nearest_lab == b) & vis
        hit = None
        for k in range(-25, 100):
            if ok[f + k:f + k + 5].all():
                hit = k; break
        print("           first 5-frame run of nearest==%s at rel %s; ball speed at t0=%.1f dmin=%.1f" % (
            b, hit, bs[f] if vis[f] else float("nan"), dmin[f] if vis[f] else float("nan")))

# --- Smart tagging in window
ev = st.load()
print("\nSmart Tagging instances overlapping public window:")
for e in ev:
    if e["end_frame"] >= 11185 and e["start_frame"] <= 18684:
        print("  %6d-%6d %-28s %s" % (e["start_frame"], e["end_frame"], e["code"],
                                      {k: v for k, v in e["tags"].items() if k != "Half"}))

# --- global ball visibility stats over graded frames; gap distribution
graded = np.zeros(N, bool); graded[11185:90652] = True; graded[110808:190198] = True
mg = ~vis & graded
runs_all = segs(mg.astype(int), 0)
gaps_all = np.array([b - a for a, b, l in runs_all if l == 1])
print("\nGraded missing-ball gaps over full match: n=%d median=%d p75=%d p90=%d p99=%d max=%d" % (
    len(gaps_all), np.median(gaps_all), np.percentile(gaps_all, 75), np.percentile(gaps_all, 90),
    np.percentile(gaps_all, 99), gaps_all.max()))
print("  share of missing frames in gaps >2s: %.1f%%" % (100 * gaps_all[gaps_all > 50].sum() / gaps_all.sum()))
print("  ball speed > 30 m/s (likely spurious) share of visible graded: %.2f%%" % (100 * np.nanmean(bs[graded & vis] > 30)))
# ball x-jumps between consecutive visible frames (teleports)
jump = np.sqrt((np.diff(bx) * L) ** 2 + (np.diff(by) * W) ** 2)
print("  consecutive-visible ball jumps > 3m/frame: %d" % np.nansum(jump > 3))

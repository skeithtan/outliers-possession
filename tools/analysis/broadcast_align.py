"""Alignment v2: one-sided chamfer (broadcast -> tactical), mirror hypotheses, temporal window."""
import os, sys, time
import numpy as np
import xml.etree.ElementTree as ET
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "cache", "atd_tactical.npz")
sys.path.insert(0, ROOT)
from games import asset_paths
L, W = 105.3, 68.0

def parse(source):
    p = asset_paths(source=source)
    root = ET.parse(p["atd_meta"]).getroot()
    tids = [t.get("id") for t in root.iter("Team")]
    names = [t.findtext("Name") for t in root.iter("Team")]
    team = np.array([0 if pl.get("teamId") == tids[0] else 1 for pl in root.iter("Player")], np.int8)
    rows = []
    with open(p["atd_raw"]) as fh:
        for line in fh:
            fid, players, ball = line.rstrip("\n").split(":")
            v = np.array(players.replace(";", ",").split(","), np.float32).reshape(-1, 3)
            b = ball.split(",")
            rows.append((v[:, 0], v[:, 1], float(b[0]), float(b[1])))
    n = len(rows); ns = len(team)
    px = np.full((n, ns), np.nan, np.float32); py = np.full((n, ns), np.nan, np.float32)
    bx = np.full(n, np.nan, np.float32); by = np.full(n, np.nan, np.float32)
    for i, (x, y, bxx, byy) in enumerate(rows):
        px[i, :len(x)] = x; py[i, :len(y)] = y; bx[i] = bxx; by[i] = byy
    return px, py, bx, by, team, names

Bpx, Bpy, Bbx, Bby, Bteam, Bnames = parse("broadcast")
print("broadcast teams:", Bnames)
d = np.load(CACHE)
Tpx, Tpy, Tbx, Tby, Tteam = d["px"], d["py"], d["bx"], d["by"], d["team_of_slot"]

# --- orientation check: home-team centroid x over the match (first vs second half), both sources
def centroid_x(px, team, t):
    with np.errstate(all="ignore"):
        return np.nanmean(np.where(team[None, :] == t, px, np.nan), axis=1)
tc = centroid_x(Tpx, Tteam, 0); bc = centroid_x(Bpx, Bteam, 0)
print("tactical home centroid x: first 40%% of frames %.2f, last 40%% %.2f" % (np.nanmean(tc[:80000]), np.nanmean(tc[-80000:])))
print("broadcast home centroid x: first 40%% of frames %.2f, last 40%% %.2f" % (np.nanmean(bc[:65000]), np.nanmean(bc[-65000:])))
print("broadcast players visible per frame: mean %.1f" % (~np.isnan(Bpx)).sum(1).mean())

# --- kd-trees for tactical frames (per team) built lazily
def tpts(i, t):
    m = (Tteam == t) & ~np.isnan(Tpx[i])
    return np.stack([Tpx[i, m] * L, Tpy[i, m] * W], 1)

def bpts(i, t, mirror):
    m = (Bteam == t) & ~np.isnan(Bpx[i])
    x, y = Bpx[i, m], Bpy[i, m]
    if mirror in (1, 3): x = 1 - x
    if mirror in (2, 3): y = 1 - y
    return np.stack([x * L, y * W], 1)

def one_sided(P_b, P_t):
    if len(P_b) < 3 or len(P_t) < 3:
        return np.inf
    D = np.sqrt(((P_b[:, None, :] - P_t[None, :, :]) ** 2).sum(-1))
    return D.min(1).mean()

# Reference broadcast frames: pick frames with many visible players and a ball, spread through the match
rng = np.random.RandomState(1)
nvis = (~np.isnan(Bpx)).sum(1)
cands = np.flatnonzero((nvis >= 14) & ~np.isnan(Bbx))
sample = np.sort(rng.choice(cands, 8, replace=False))
STEP = 5
tidx = np.arange(0, len(Tbx), STEP)
print("\nFor each broadcast frame: best tactical frame under each mirror hypothesis (window of 3 frames at +-25)")
print("%8s | %s" % ("bcast", "  ".join("m%d: frame d_best ratio" % m for m in range(4))))
t0 = time.time()
results = []
for bf in sample:
    line = []
    for mirror in range(4):
        scores = np.full(len(tidx), np.inf)
        Pb = [(bpts(bf + k, 0, mirror), bpts(bf + k, 1, mirror)) for k in (-25, 0, 25) if 0 <= bf + k < len(Bbx)]
        for j, tf in enumerate(tidx):
            if tf - 25 < 0 or tf + 25 >= len(Tbx):
                continue
            tot = 0.0
            for (b0, b1), k in zip(Pb, (-25, 0, 25)):
                tot += one_sided(b0, tpts(tf + k, 0)) + one_sided(b1, tpts(tf + k, 1))
            scores[j] = tot / (2 * len(Pb))
        order = np.argsort(scores)
        best = order[0]
        far = order[np.abs(tidx[order] - tidx[best]) > 125]
        second = far[0]
        line.append("m%d: %6d %5.2f %4.2f" % (mirror, tidx[best], scores[best], scores[second] / scores[best]))
        results.append((bf, mirror, tidx[best], scores[best], scores[second] / scores[best]))
    print("%8d | %s" % (bf, "  ".join(line)))
print("took %.0fs" % (time.time() - t0))

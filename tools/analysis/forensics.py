"""Per-transition forensics on the public window: when does the GT switch relative to contact?"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "cache", "atd_tactical.npz")
d = np.load(CACHE)
px, py, bx, by, team = d["px"], d["py"], d["bx"], d["by"], d["team_of_slot"]
L, W = 105.3, 68.0
gt = pd.read_csv(ROOT + "/evaluation/gt/BAR-ATH_possession_gt_public.csv")["label"].values

# derived ball speed (m/s) from visible consecutive frames
BX, BY = bx * L, by * W
vx = np.full_like(BX, np.nan); vx[1:] = (BX[1:] - BX[:-1]) * 25
vy = np.full_like(BY, np.nan); vy[1:] = (BY[1:] - BY[:-1]) * 25
speed = np.sqrt(vx ** 2 + vy ** 2)
# smoothed over 3 frames
sp = pd.Series(speed).rolling(3, center=True, min_periods=1).median().values

dx = (px - bx[:, None]) * L
dy = (py - by[:, None]) * W
dist = np.sqrt(dx ** 2 + dy ** 2)
with np.errstate(all="ignore"):
    dH = np.nanmin(np.where(team[None, :] == 0, dist, np.nan), axis=1)
    dA = np.nanmin(np.where(team[None, :] == 1, dist, np.nan), axis=1)

win = np.arange(11185, 18685)
g = gt[win]
br = np.flatnonzero(g[1:] != g[:-1]) + 1
S = []
b = np.concatenate(([0], br, [len(g)]))
for i in range(len(b) - 1):
    S.append((11185 + b[i], 11185 + b[i + 1], g[b[i]]))

def row(f):
    return "%6d  dH=%5.1f dA=%5.1f  v=%5.1f  %s" % (
        f, dH[f], dA[f], sp[f], gt[f])

print("Legend: dH/dA = nearest home/away player distance to ball (m); v = derived ball speed m/s\n")
for i in range(1, len(S)):
    f, prev, new = S[i][0], S[i - 1][2], S[i][2]
    print("=" * 70)
    print("GT %s -> %s at %d   (prev seg %.1fs, new seg %.1fs)" % (
        prev, new, f, (S[i-1][1]-S[i-1][0])/25, (S[i][1]-S[i][0])/25))
    # contact events of the new team: dNew < 1.2 m
    if new in "HA":
        dn = dH if new == "H" else dA
        close = np.flatnonzero(dn[f - 75:f + 50] < 1.2) + f - 75
        print("  frames with new-team player <1.2m from ball, rel to GT switch:",
              sorted(set(((close - f) // 5 * 5).tolist()))[:20])
    for k in range(-30, 16, 3):
        print("  " + ("->" if k == 0 else "  ") + row(f + k))

# ball-speed statistics: at GT switch frame vs. 20 frames before
sw = [S[i][0] for i in range(1, len(S)) if S[i][2] in "HA" and S[i-1][2] in "HA"]
print("\nspeed at switch: ", np.round([sp[f] for f in sw], 1))
print("speed 20f before:", np.round([sp[f - 20] for f in sw], 1))
print("speed 20f after: ", np.round([sp[f + 20] for f in sw], 1))

# dead-ball onsets: ball position (x,y in m) and whether out of bounds within +-25 frames
print("\nDead-ball onsets:")
for i in range(1, len(S)):
    if S[i][2] == "D":
        f = S[i][0]
        seg = slice(f - 25, f + 50)
        xs, ys = BX[seg], BY[seg]
        vis = ~np.isnan(xs)
        print("  %d: ball visible %d/75; x range %.1f..%.1f  y range %.1f..%.1f  ; stationary(<1m/s) frames after onset: %d" % (
            f, vis.sum(), np.nanmin(xs), np.nanmax(xs), np.nanmin(ys), np.nanmax(ys),
            int(np.nansum(sp[f:f + 50] < 1.0))))

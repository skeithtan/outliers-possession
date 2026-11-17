"""R2-a step 4 (direction d): the H dribble at 15654 and the H release at 15676.

Reports, without proposing a label:
  * tactical and broadcast ball over 15600-15800 (every 4th frame) and their distance;
  * the engine events and the public GT changes in the window (GT read via tools/gt.py);
  * the H slot of the dribble: team_of_slot, tracklet birth/death, slot confidence, and its
    nearest broadcast player (with the broadcast team) as an independent team check; any A slot
    within 1.5 m of it (double track); its path over the tracklet;
  * visible H and A track counts over 15600-15800 (over-count > 11);
  * every player within 8 m of the tactical ball at 15654 and 15676;
  * Smart Tagging rows overlapping 15600-15800.
Usage: python tools/analysis/r2_broadcast_case15654.py <out_dir>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import r2_broadcast_common as C  # noqa: E402
from gt import public_gt_codes  # noqa: E402
from engine import data, tracks  # noqa: E402
from engine.config import Config  # noqa: E402

OUT = sys.argv[1]
LO, HI = 15600, 15800
d = C.load(OUT)
tac = data.load_tactical()
B = data.load_broadcast()
L, W = data.pitch_m()
conf = tracks.slot_confidence(tac, Config.default())
tx, ty = d["ball"]["x"], d["ball"]["y"]
tv = d["ball"]["visible"].astype(bool) & np.isfinite(tx)
tx, ty = np.where(tv, tx, np.nan), np.where(tv, ty, np.nan)
bx, by = d["bb"]["bb_x"], d["bb"]["bb_y"]
px, py, team = d["px"], d["py"], d["team_of_slot"]
ps = tac["ps"]
vis = np.isfinite(px)
gt = public_gt_codes()

print("=== engine events and GT changes, %d-%d ===" % (LO - 100, HI + 100))
ev = d["events"]
print(ev[(ev.frame >= LO - 100) & (ev.frame <= HI + 100)].to_string(index=False))
ch = np.flatnonzero(np.diff(gt[LO - 300:HI + 100])) + LO - 299
print("public GT changes:", [(int(c), "HADX"[gt[c - 1]], "HADX"[gt[c]]) for c in ch])
lab = d["labels"]
ch = np.flatnonzero(lab[LO - 300:HI + 100][1:] != lab[LO - 300:HI + 100][:-1]) + LO - 299
print("engine changes:", [(int(c), str(lab[c - 1]), str(lab[c])) for c in ch])

dT, sT = C.nearest_per_team(px, py, team, tx, ty)
dB, sB = C.nearest_per_team(px, py, team, bx, by)
print("\n=== ball tracks %d-%d (every 4th frame) ===" % (LO, HI))
print(" frame   T(x,y)  Tspd   B(x,y)  |T-B|  chamfer  T:H d/slot  A d/slot   B:H d/slot  A d/slot   nvisH nvisA")
for f in range(LO, HI + 1, 4):
    print("%6d (%5.1f,%4.1f) %5.1f (%5.1f,%4.1f) %5.2f  %4.2f   H %5.2f/%2d A %5.2f/%2d   H %5.2f/%2d A %5.2f/%2d   %2d %2d"
          % (f, tx[f], ty[f], d["ball"]["speed"][f], bx[f], by[f], np.hypot(tx[f] - bx[f], ty[f] - by[f]),
             d["bb"]["chamfer"][f], dT[f, 0], sT[f, 0], dT[f, 1], sT[f, 1], dB[f, 0], sB[f, 0], dB[f, 1], sB[f, 1],
             vis[f, team == 0].sum(), vis[f, team == 1].sum()))
w = np.arange(LO, HI + 1)
dd = np.hypot(tx[w] - bx[w], ty[w] - by[w])
print("|T-B| over %d-%d: n both %d, median %.2f, p90 %.2f, max %.2f m" % (LO, HI, np.isfinite(dd).sum(),
      np.nanmedian(dd), np.nanpercentile(dd, 90), np.nanmax(dd)))
for a, b in ((15640, 15700), (15700, 15770)):
    s = slice(a - LO, b - LO)
    print("  %d-%d: median |T-B| %.2f m; min H dist to B ball %.2f, min A dist to B ball %.2f; to T ball H %.2f A %.2f"
          % (a, b, np.nanmedian(dd[s]), np.nanmin(dB[a:b, 0]), np.nanmin(dB[a:b, 1]), np.nanmin(dT[a:b, 0]), np.nanmin(dT[a:b, 1])))

print("\n=== visible track counts %d-%d ===" % (LO, HI))
nH = vis[LO:HI + 1][:, team == 0].sum(1)
nA = vis[LO:HI + 1][:, team == 1].sum(1)
print("H tracks: min %d max %d, frames with > 11: %d; A tracks: min %d max %d, frames with > 11: %d"
      % (nH.min(), nH.max(), (nH > 11).sum(), nA.min(), nA.max(), (nA > 11).sum()))


def tracklet(slot, f):
    v = vis[:, slot]
    a = f
    while a > 0 and v[a - 1]:
        a -= 1
    b = f
    while b + 1 < len(v) and v[b + 1]:
        b += 1
    return a, b


# broadcast players on the tactical index
bf = d["bb"]["b_frame"]


def broadcast_match(f, x, y):
    """Nearest broadcast player to (x, y) at tactical frame f: (dist, broadcast slot, team)."""
    if bf[f] < 0 or not d["bb"]["valid"][f]:
        return (np.nan, -1, -1)
    bxp = B["px"][bf[f]].astype(float) * L
    byp = B["py"][bf[f]].astype(float) * W
    dist = np.hypot(bxp - x, byp - y)
    if not np.isfinite(dist).any():
        return (np.nan, -1, -1)
    j = int(np.nanargmin(dist))
    return (float(dist[j]), j, int(B["team_of_slot"][j]))


for f in (15654, 15676):
    print("\n=== players within 8 m of the tactical ball at %d (ball (%.1f, %.1f), broadcast ball (%.1f, %.1f)) ==="
          % (f, tx[f], ty[f], bx[f], by[f]))
    dist = np.hypot(px[f] - tx[f], py[f] - ty[f])
    order = np.argsort(np.where(np.isfinite(dist), dist, np.inf))
    print(" slot team  dist_T  dist_B   x     y   speed  conf  tracklet       bcast match (d, bslot, team)")
    for s in order:
        if not np.isfinite(dist[s]) or dist[s] > 8:
            break
        a, b = tracklet(s, f)
        m = broadcast_match(f, px[f, s], py[f, s])
        print("  %2d   %s   %5.2f  %6.2f  %5.1f %5.1f  %4.1f  %.2f  %6d-%6d   (%.2f, %d, %s)"
              % (s, "HA"[team[s]], dist[s], np.hypot(px[f, s] - bx[f], py[f, s] - by[f]), px[f, s], py[f, s],
                 ps[f, s], conf[f, s], a, b, m[0], m[1], "HA"[m[2]] if m[2] >= 0 else "-"))

slot = int(d["feats"]["nearest_slot_H"][15654])
print("\n=== dribble H slot %d ===" % slot)
a, b = tracklet(slot, 15654)
print("team_of_slot %s; tracklet %d-%d (%.1f s); slot confidence at 15654 %.2f, min over tracklet %.2f"
      % ("HA"[team[slot]], a, b, (b - a + 1) / 25.0, conf[15654, slot], conf[a:b + 1, slot].min()))
print("slot %d visible tracklets in the match: %d; total visible frames %d" % (
    slot, len(tracks._runs(vis[:, slot])[0]), vis[:, slot].sum()))
# double track: any A slot within 1.5 m of this slot during the tracklet
dA = np.full(b - a + 1, np.inf)
for s in np.flatnonzero(team == 1):
    dd2 = np.hypot(px[a:b + 1, s] - px[a:b + 1, slot], py[a:b + 1, s] - py[a:b + 1, slot])
    dA = np.fmin(dA, np.where(np.isfinite(dd2), dd2, np.inf))
print("nearest A slot to slot %d over its tracklet: min %.2f m, frames within 1.5 m: %d" % (slot, dA.min(), (dA < 1.5).sum()))
# broadcast team of the matched player along the tracklet
teams = []
dists = []
for f in range(max(a, LO), min(b, HI) + 1):
    m = broadcast_match(f, px[f, slot], py[f, slot])
    if m[2] >= 0 and m[0] < 1.5:
        teams.append(m[2])
        dists.append(m[0])
teams = np.array(teams)
print("broadcast match of slot %d (%d-%d, within 1.5 m): %d frames, broadcast team H on %d, A on %d; median match dist %.2f m"
      % (slot, max(a, LO), min(b, HI), len(teams), (teams == 0).sum(), (teams == 1).sum(), np.median(dists) if dists else np.nan))
# whole-match agreement of tactical team with broadcast team (baseline for the check above)
agree = n = 0
for f in range(11185, 90652, 50):
    if bf[f] < 0 or not d["bb"]["valid"][f] or not (d["bb"]["chamfer"][f] < 1):
        continue
    for s in np.flatnonzero(vis[f]):
        m = broadcast_match(f, px[f, s], py[f, s])
        if m[2] >= 0 and m[0] < 1.0:
            n += 1
            agree += int(m[2] == team[s])
print("baseline (H1, every 50th frame, chamfer < 1, match < 1 m): tactical team == broadcast team on %.1f%% of %d player matches"
      % (100.0 * agree / n, n))
print("slot %d path (every 25 frames over its tracklet, clipped to %d-%d):" % (slot, LO - 200, HI + 200))
for f in range(max(a, LO - 200), min(b, HI + 200) + 1, 25):
    print("   %6d (%5.1f, %4.1f) speed %.1f  dist to T ball %.2f" % (f, px[f, slot], py[f, slot], ps[f, slot],
                                                                    np.hypot(px[f, slot] - tx[f], py[f, slot] - ty[f])))

print("\n=== Smart Tagging rows overlapping %d-%d ===" % (LO, HI))
tg = data.load_tagging()
ov = tg[(tg.start_frame < HI + 1) & (tg.end_frame > LO)]
print(ov[["code", "team", "type", "start_frame", "end_frame"]].to_string(index=False))

print("\n=== slot identity checks (goalkeeper hypothesis) ===")
prev = a - 1
while prev > 0 and not vis[prev, slot]:
    prev -= 1
gap = a - prev
dist_jump = np.hypot(px[a, slot] - px[prev, slot], py[a, slot] - py[prev, slot])
print("slot %d last seen before %d at frame %d (%.1f, %.1f); reborn at (%.1f, %.1f): %.1f m in %d frames = %.1f m/s"
      % (slot, a, prev, px[prev, slot], py[prev, slot], px[a, slot], py[a, slot], dist_jump, gap, dist_jump / gap * 25))
x0 = px[prev - 2500:prev + 1, slot]
print("slot %d over the 100 s before frame %d: x median %.1f, x range %.1f-%.1f (visible %d frames)"
      % (slot, prev, np.nanmedian(x0), np.nanmin(x0), np.nanmax(x0), np.isfinite(x0).sum()))
step = np.hypot(np.diff(px[a:min(b, a + 1000) + 1, slot]), np.diff(py[a:min(b, a + 1000) + 1, slot]))
big = np.flatnonzero(step > 2.0)
if big.size:
    j0, j1 = a + big[0], a + big[-1] + 1
    print("slot %d inside its tracklet: steps > 2 m/frame on %d frames in %d-%d; position (%.1f, %.1f) -> (%.1f, %.1f), %.1f m"
          % (slot, big.size, j0, j1, px[j0, slot], py[j0, slot], px[j1, slot], py[j1, slot],
             np.hypot(px[j1, slot] - px[j0, slot], py[j1, slot] - py[j0, slot])))
    x1 = px[j1:j1 + 2500, slot]
    print("slot %d over the 100 s after %d: x median %.1f, range %.1f-%.1f" % (slot, j1, np.nanmedian(x1), np.nanmin(x1), np.nanmax(x1)))
for t in (0, 1):
    cols = team == t
    print("team %s: tracks with x < 30 at 15654: %d; min x %.1f; max x %.1f" % (
        "HA"[t], (px[15654, cols] < 30).sum(), np.nanmin(px[15654, cols]), np.nanmax(px[15654, cols])))
# frozen A slot near the ball
for s in np.flatnonzero((team == 1) & vis[15654]):
    if np.hypot(px[15654, s] - tx[15654], py[15654, s] - ty[15654]) < 8:
        mv = np.nansum(np.hypot(np.diff(px[LO:HI + 1, s]), np.diff(py[LO:HI + 1, s])))
        m = broadcast_match(15654, px[15654, s], py[15654, s])
        print("A slot %d near the ball: moved %.2f m over %d-%d, speed NaN on %d/%d frames; nearest broadcast player %.2f m (%s)"
              % (s, mv, LO, HI, np.isnan(ps[LO:HI + 1, s]).sum(), HI - LO + 1, m[0], "HA"[m[2]] if m[2] >= 0 else "-"))
print("whole match: visible player-frames with NaN speed %.1f%%" % (100 * np.mean(np.isnan(ps[vis]))))

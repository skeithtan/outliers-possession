"""R2-a (direction d, generalised): goalkeeper slots that appear in the opponent's half.

Case 15654 showed the H goalkeeper slot sitting in the A penalty area for ~13 s, 0.6 m from the
ball, while the A slot near it is frozen (NaN speed). This script measures the pattern over the
whole match:
  * goalkeeper slot of each team and half = the slot with the most visible frames inside its own
    penalty area (own goal side from the team's median x in that half);
  * frames where that slot is more than 52.65 m (half the pitch) from its own goal line;
  * engine event rows (all kinds) on those frames whose team is the slot's team and whose
    nearest player of that team is the slot; and the engine switches within 50 frames of them;
  * frozen tracks: visible player-frames with NaN speed and < 0.05 m movement over 25 frames,
    within 3 m of the ball, by team.
Usage: python tools/analysis/r2_gk_swap.py <out_dir>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_broadcast_common as C  # noqa: E402
from engine import data  # noqa: E402

OUT = sys.argv[1]
d = C.load(OUT)
L, W = data.pitch_m()
px, py, team = d["px"], d["py"], d["team_of_slot"]
ps = data.load_tactical()["ps"]
vis = np.isfinite(px)
lab = np.asarray(d["labels"]).astype(str)
ev = d["events"]
N = px.shape[0]
fr_all = ev["frame"].to_numpy()

for h, lo, hi in d["blocks"]:
    for t in (0, 1):
        cols = np.flatnonzero(team == t)
        with np.errstate(all="ignore"):
            mx = np.nanmedian(px[lo:hi + 1][:, cols])
        own_left = mx < L / 2  # the team plays deeper on its own side on average
        # more robust: use the goalkeeper-like slot: most frames within 16.5 m of either goal line
        nl = np.sum(vis[lo:hi + 1][:, cols] & (px[lo:hi + 1][:, cols] < 16.5), axis=0)
        nr = np.sum(vis[lo:hi + 1][:, cols] & (px[lo:hi + 1][:, cols] > L - 16.5), axis=0)
        side_left = nl.max() > nr.max()
        gk = int(cols[np.argmax(nl if side_left else nr)])
        own_x = 0.0 if side_left else L
        dgoal = np.abs(px[lo:hi + 1, gk] - own_x)
        away = vis[lo:hi + 1, gk] & (dgoal > L / 2)
        inbox = vis[lo:hi + 1, gk] & (dgoal > L - 16.5)
        print("half %d team %s: own goal at x=%.0f (team median x %.1f); goalkeeper-like slot %d (%d frames in own box)"
              % (h, "HA"[t], own_x, mx, gk, (nl if side_left else nr).max()))
        p = np.concatenate(([0], away.astype(int), [0]))
        e = np.flatnonzero(np.diff(p))
        runs = [(lo + a, lo + b) for a, b in zip(e[::2], e[1::2])]
        print("   slot %d beyond half-way: %d frames in %d runs; inside the opponent penalty area: %d frames; runs >= 50 frames: %s"
              % (gk, away.sum(), len(runs), inbox.sum(), [(a, b) for a, b in runs if b - a >= 50][:12]))
        ids = []
        for a, b in runs:
            sel = (fr_all >= a) & (fr_all < b) & (ev["team"].to_numpy() == "HA"[t])
            for i in np.flatnonzero(sel):
                f = int(fr_all[i])
                if d["feats"]["nearest_slot_" + "HA"[t]][f] == gk:
                    ids.append((f, ev["kind"].iloc[i]))
        print("   engine event rows credited to team %s with slot %d nearest while it is beyond half-way: %d %s"
              % ("HA"[t], gk, len(ids), ids[:12]))
        sw = [f for f, _ in ids if np.any(lab[max(f - 50, 0):f + 51][1:] != lab[max(f - 50, 0):f + 51][:-1])]
        print("   of those, with an engine label change within +-50 frames: %d" % len(sw))

print("\n=== frozen tracks near the ball (graded) ===")
tx, ty = d["ball"]["x"], d["ball"]["y"]
mv = np.full(px.shape, np.nan)
mv[:-25] = np.hypot(px[25:] - px[:-25], py[25:] - py[:-25])
frozen = vis & np.isnan(ps) & (mv < 0.05)
near = np.hypot(px - tx[:, None], py - ty[:, None]) < 3
g = d["graded"]
for t in (0, 1):
    cols = team == t
    print("team %s: frozen visible player-frames %d (%.2f%% of visible graded), of which within 3 m of the ball %d"
          % ("HA"[t], (frozen[g][:, cols]).sum(), 100 * frozen[g][:, cols].sum() / vis[g][:, cols].sum(),
             (frozen & near)[g][:, cols].sum()))

print("\n=== per-row signature: other goalkeeper slot frozen? broadcast team of the live slot? public GT? ===")
sys.path.insert(0, os.path.dirname(HERE))
from gt import public_gt_codes  # noqa: E402
gt = public_gt_codes()
B = data.load_broadcast()
bf = d["bb"]["b_frame"]
gk_slot = {}
for h, lo, hi in d["blocks"]:
    for t in (0, 1):
        cols = np.flatnonzero(team == t)
        nl = np.sum(vis[lo:hi + 1][:, cols] & (px[lo:hi + 1][:, cols] < 16.5), axis=0)
        nr = np.sum(vis[lo:hi + 1][:, cols] & (px[lo:hi + 1][:, cols] > L - 16.5), axis=0)
        left = nl.max() > nr.max()
        gk_slot[(h, t)] = (int(cols[np.argmax(nl if left else nr)]), 0.0 if left else L)
n_rows = n_frozen = n_bsame = n_bcov = 0
for i in range(len(ev)):
    f = int(ev["frame"].iloc[i])
    t_letter = ev["team"].iloc[i]
    if t_letter not in ("H", "A") or not d["graded"][f]:
        continue
    t = "HA".index(t_letter)
    h = int(d["half"][f])
    gk, own_x = gk_slot[(h, t)]
    if d["feats"]["nearest_slot_" + t_letter][f] != gk or not vis[f, gk] or abs(px[f, gk] - own_x) <= L / 2:
        continue
    n_rows += 1
    ogk, _ = gk_slot[(h, 1 - t)]
    ofro = vis[f, ogk] and np.isnan(ps[f, ogk])
    od = np.hypot(px[f, ogk] - px[f, gk], py[f, ogk] - py[f, gk]) if vis[f, ogk] else np.nan
    n_frozen += bool(ofro)
    bteam = "-"
    if bf[f] >= 0 and d["bb"]["valid"][f]:
        bxp = B["px"][bf[f]].astype(float) * L
        byp = B["py"][bf[f]].astype(float) * W
        dd = np.hypot(bxp - px[f, gk], byp - py[f, gk])
        if np.isfinite(dd).any() and np.nanmin(dd) < 1.5:
            bteam = "HA"[int(B["team_of_slot"][int(np.nanargmin(dd))])]
            n_bcov += 1
            n_bsame += bteam == t_letter
    g = "HADX"[gt[f]]
    print("  %6d %-9s team %s slot %d at (%.1f, %.1f) | other GK slot %d frozen %s, %.1f m away | broadcast team %s | engine %s | public GT %s"
          % (f, ev["kind"].iloc[i], t_letter, gk, px[f, gk], py[f, gk], ogk, bool(ofro), od, bteam, lab[f], g))
print("rows %d; other GK slot frozen (visible, NaN speed) on %d; broadcast matched on %d, broadcast team == tactical team on %d"
      % (n_rows, n_frozen, n_bcov, n_bsame))

"""R2-a step 5/6: full-match value of the broadcast ball at the engine's hand-overs.

  A. Every direct H<->A switch of the current engine output on graded frames: tactical-broadcast
     ball distance at the switch frame and its median over +-5 frames; whether a new-team player
     comes within 1.5 m of each ball in [f-40, f+6].
  B. Every dribble / control row of the event table (after the engine's shift): the same
     distances; whether the row's team is within 1.5 m of the broadcast ball at the row frame;
     whether the other team is nearer to the broadcast ball; an identity-jump flag for the row's
     slot (a step > 2 m/frame, or a re-birth > 30 m away at > 10 m/s, within 500 frames).
  C. Graded alive frames (engine label H or A) with both balls visible: nearest team to each
     ball, how often they differ, and how often each agrees with the engine owner / label.
  D. Upper bound on hand-overs the broadcast ball could add: runs of >= 25 frames where the
     broadcast ball is > 3 m from the tactical ball and a player of the team the engine does NOT
     label is within 1.5 m of the broadcast ball while no such player is within 1.5 m of the
     tactical ball.
Usage: python tools/analysis/r2_broadcast_fullmatch.py <out_dir>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_broadcast_common as C  # noqa: E402

OUT = sys.argv[1]
d = C.load(OUT)
N = len(d["labels"])
lab = np.asarray(d["labels"]).astype(str)
code = np.full(N, -1, np.int8)
code[lab == "H"] = 0
code[lab == "A"] = 1
tx, ty = d["ball"]["x"], d["ball"]["y"]
tv = d["ball"]["visible"].astype(bool) & np.isfinite(tx)
tx, ty = np.where(tv, tx, np.nan), np.where(tv, ty, np.nan)
bx, by = d["bb"]["bb_x"], d["bb"]["bb_y"]
bv = np.isfinite(bx)
dist = np.hypot(tx - bx, ty - by)
px, py, team = d["px"], d["py"], d["team_of_slot"]
dT, sT = C.nearest_per_team(px, py, team, tx, ty)
dB, sB = C.nearest_per_team(px, py, team, bx, by)
graded = d["graded"]
print("config hash", d["config_hash"])


def med5(f):
    w = dist[max(f - 5, 0):f + 6]
    return np.nanmedian(w) if np.isfinite(w).sum() >= 3 else np.nan


def table(name, frames, teams):
    frames = np.asarray(frames)
    at = dist[frames]
    m5 = np.array([med5(int(f)) for f in frames])
    both = np.isfinite(at)
    print("%s: n=%d | both balls visible at the frame %d (%.0f%%) | |T-B| > 3 m %d, > 5 m %d"
          % (name, len(frames), both.sum(), 100 * both.mean(), (at > 3).sum(), (at > 5).sum()))
    ok5 = np.isfinite(m5)
    print("   median over +-5 (>= 3 co-visible frames): n=%d | > 3 m %d (%.1f%%), > 5 m %d (%.1f%%) | p50 %.2f p90 %.2f m"
          % (ok5.sum(), (m5 > 3).sum(), 100 * np.mean(m5[ok5] > 3), (m5 > 5).sum(), 100 * np.mean(m5[ok5] > 5),
             np.median(m5[ok5]), np.percentile(m5[ok5], 90)))
    return at, m5


print("\n=== A. engine H<->A switches (graded) ===")
sw = np.flatnonzero((code[1:] >= 0) & (code[:-1] >= 0) & (code[1:] != code[:-1])) + 1
sw = sw[graded[sw]]
at, m5 = table("switches", sw, code[sw])
nT = nB = sup = con = cov = 0
for f in sw:
    k = code[f]
    w = slice(f - 40, f + 7)
    t_hit = np.nanmin(dT[w, k]) <= 1.5 if np.isfinite(dT[w, k]).any() else False
    b_cov = np.isfinite(dB[w, k]).sum() >= 10
    b_hit = np.nanmin(dB[w, k]) <= 1.5 if np.isfinite(dB[w, k]).any() else False
    nT += t_hit
    nB += b_hit
    cov += b_cov
    sup += (b_hit and not t_hit)
    con += (t_hit and not b_hit and b_cov)
print("   new team within 1.5 m in [f-40, f+6]: of T ball %d/%d, of B ball %d/%d (B ball on >= 10 frames: %d)"
      % (nT, len(sw), nB, len(sw), cov))
print("   B supports where T does not: %d | T supports where B (covered) does not: %d" % (sup, con))

print("\n=== B. dribble / control rows ===")
ev = d["events"]
rows = ev[ev["kind"].isin(["dribble", "control"]) & ev["team"].isin(["H", "A"])]
rows = rows[graded[rows["frame"].to_numpy()]]
fr = rows["frame"].to_numpy()
tm = np.array([{"H": 0, "A": 1}[t] for t in rows["team"]])
for kind in ("dribble", "control"):
    s = (rows["kind"] == kind).to_numpy()
    table(kind, fr[s], tm[s])
at, m5 = table("dribble+control", fr, tm)
own_b = np.array([dB[f, k] for f, k in zip(fr, tm)])
oth_b = np.array([dB[f, 1 - k] for f, k in zip(fr, tm)])
okb = np.isfinite(own_b)
print("   at the row frame with B ball: n=%d | row team within 1.5 m of B ball %d (%.0f%%) | other team nearer to B ball %d (%.0f%%)"
      % (okb.sum(), (own_b[okb] <= 1.5).sum(), 100 * np.mean(own_b[okb] <= 1.5), (oth_b[okb] < own_b[okb]).sum(),
         100 * np.mean(oth_b[okb] < own_b[okb])))
far = okb & (at > 3)
print("   rows with |T-B| > 3 m at the frame: %d; of those the other team nearer to B ball: %d" % (far.sum(), (far & (oth_b < own_b)).sum()))

# identity-jump flag for each row's slot
vis = np.isfinite(px)
step = np.full(px.shape, np.nan)
step[1:] = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0))


def identity_jump(f, slot, span=500):
    lo, hi = max(f - span, 1), min(f + span, N - 1)
    if np.nanmax(np.where(np.isfinite(step[lo:hi, slot]), step[lo:hi, slot], 0)) > 2.0:
        return True
    v = vis[lo:hi, slot]
    births = np.flatnonzero(v[1:] & ~v[:-1]) + lo + 1
    for b0 in births:
        p = b0 - 1
        while p > 0 and not vis[p, slot]:
            p -= 1
        if p <= 0:
            continue
        jump = np.hypot(px[b0, slot] - px[p, slot], py[b0, slot] - py[p, slot])
        if jump > 30 and jump / (b0 - p) * 25 > 10:
            return True
    return False


slots = np.array([d["feats"]["nearest_slot_" + "HA"[k]][f] for f, k in zip(fr, tm)])
flag = np.array([identity_jump(int(f), int(s)) if s >= 0 else False for f, s in zip(fr, slots)])
print("   rows whose slot has an identity jump within +-500 frames: %d / %d" % (flag.sum(), len(fr)))
ex = [(int(f), "HA"[k], int(s)) for f, k, s, g in zip(fr, tm, slots, flag) if g]
print("   first 15:", ex[:15])

print("\n=== C. nearest team on graded alive frames, both balls visible ===")
alive = graded & (code >= 0) & ~d["stoppage"]
both = alive & tv & bv & np.isfinite(dT).all(1) & np.isfinite(dB).all(1)
nt_T = np.where(dT[:, 0] <= dT[:, 1], 0, 1)
nt_B = np.where(dB[:, 0] <= dB[:, 1], 0, 1)
owner = d["owner"]
print("frames %d | nearest team differs T vs B on %d (%.1f%%)" % (both.sum(), (both & (nt_T != nt_B)).sum(),
                                                               100 * np.mean(nt_T[both] != nt_B[both])))
close = both & (np.fmin(np.fmin(dT[:, 0], dT[:, 1]), np.fmin(dB[:, 0], dB[:, 1])) <= 3)
print("  with a player within 3 m of either ball: %d frames, differs on %.1f%%" % (close.sum(), 100 * np.mean(nt_T[close] != nt_B[close])))
far3 = both & (dist > 3)
print("  with |T-B| > 3 m: %d frames, differs on %.1f%%" % (far3.sum(), 100 * np.mean(nt_T[far3] != nt_B[far3])))
known = both & (owner >= 0)
print("  owner known on %d: T nearest == owner %.1f%%, B nearest == owner %.1f%%"
      % (known.sum(), 100 * np.mean(nt_T[known] == owner[known]), 100 * np.mean(nt_B[known] == owner[known])))
print("  label: T nearest == label %.1f%%, B nearest == label %.1f%%"
      % (100 * np.mean(nt_T[both] == code[both]), 100 * np.mean(nt_B[both] == code[both])))
dif = both & (nt_T != nt_B)
print("  where they differ (%d): T nearest == label %.1f%%, B nearest == label %.1f%%; median |T-B| %.2f m"
      % (dif.sum(), 100 * np.mean(nt_T[dif] == code[dif]), 100 * np.mean(nt_B[dif] == code[dif]), np.median(dist[dif])))

print("\n=== D. upper bound: broadcast-only control by the team the engine does not label ===")
other = np.where(code >= 0, 1 - code, 0)
b_other = np.array([dB[i, other[i]] for i in range(N)])
t_other = np.array([dT[i, other[i]] for i in range(N)])
cand = alive & bv & (dist > 3) & (b_other <= 1.5) & ~(np.nan_to_num(t_other, nan=99) <= 1.5)
p = np.concatenate(([0], cand.astype(int), [0]))
e = np.flatnonzero(np.diff(p))
st, en = e[::2], e[1::2]
# merge runs separated by <= 5 frames
runs = []
for s0, e0 in zip(st, en):
    if runs and s0 - runs[-1][1] <= 5:
        runs[-1][1] = e0
    else:
        runs.append([s0, e0])
runs = np.array(runs) if runs else np.zeros((0, 2), int)
ln = runs[:, 1] - runs[:, 0] if len(runs) else np.array([])
long_runs = runs[ln >= 25]
print("candidate frames %d; runs %d; runs >= 25 frames: %d (H1 %d, H2 %d)"
      % (cand.sum(), len(runs), len(long_runs), (long_runs[:, 0] < 100000).sum(), (long_runs[:, 0] >= 100000).sum()))
print("  run starts (>= 25 frames, first 40):", [int(r[0]) for r in long_runs[:40]])
print("  in the public window 11185-18685:", [(int(r[0]), int(r[1])) for r in long_runs if 11185 <= r[0] <= 18685])
med_ch = [np.nanmedian(d["bb"]["chamfer"][r[0]:r[1]]) for r in long_runs]
print("  chamfer median over those runs: p50 %.2f p90 %.2f m" % (np.median(med_ch), np.percentile(med_ch, 90)) if len(med_ch) else "")
# is the tactical ball stuck in those runs? range of T positions
stuck = 0
for r in long_runs:
    w = slice(r[0], r[1])
    if np.isfinite(tx[w]).sum() >= 10:
        spd = np.nanmedian(d["ball"]["speed"][w])
        anyp = np.nanmedian(np.fmin(dT[w, 0], dT[w, 1]))
        stuck += (spd < 2.0 and anyp > 3)
        g = d.get("gt_public")
        print("    run %d-%d: |T-B| median %.1f m, T speed median %.1f, nearest player to T median %.1f m, to B median %.1f m, engine %s, broadcast suggests %s"
              % (r[0], r[1], np.nanmedian(dist[w]), spd, anyp, np.nanmedian(np.fmin(dB[w, 0], dB[w, 1])),
                 "".join(sorted(set(lab[w]))), "HA"[int(np.median(other[w]))]))
print("  of those, tactical ball slow and isolated (median speed < 2 m/s, median nearest player > 3 m): %d" % stuck)

print("\n=== E. public window only: which ball's nearest team agrees with the public GT? ===")
sys.path.insert(0, os.path.dirname(HERE))
from gt import public_gt_codes  # noqa: E402
gt = public_gt_codes()
pub = (gt <= 1) & tv & bv & np.isfinite(dT).all(1) & np.isfinite(dB).all(1)
print("public GT H/A frames with both balls: %d | T nearest == GT %.1f%% | B nearest == GT %.1f%% | engine label == GT %.1f%%"
      % (pub.sum(), 100 * np.mean(nt_T[pub] == gt[pub]), 100 * np.mean(nt_B[pub] == gt[pub]), 100 * np.mean(code[pub] == gt[pub])))
dif = pub & (nt_T != nt_B)
print("  frames where the nearest teams differ: %d | T right %d | B right %d" % (dif.sum(), (nt_T[dif] == gt[dif]).sum(), (nt_B[dif] == gt[dif]).sum()))
for thr in (1.5, 3.0):
    cT = pub & (np.fmin(dT[:, 0], dT[:, 1]) <= thr)
    cB = pub & (np.fmin(dB[:, 0], dB[:, 1]) <= thr)
    print("  nearest player within %.1f m: T ball %d frames, nearest team == GT %.1f%% | B ball %d frames, == GT %.1f%%"
          % (thr, cT.sum(), 100 * np.mean(nt_T[cT] == gt[cT]), cB.sum(), 100 * np.mean(nt_B[cB] == gt[cB])))
far3 = pub & (dist > 3)
print("  |T-B| > 3 m: %d frames | T nearest == GT %.1f%% | B nearest == GT %.1f%%"
      % (far3.sum(), 100 * np.mean(nt_T[far3] == gt[far3]), 100 * np.mean(nt_B[far3] == gt[far3])))

print("\n=== F. narrow rule candidate: tactical ghost ball with a live broadcast ball ===")
spd = d["ball"]["speed"]
ghost = (graded & ~d["stoppage"] & tv & bv & (spd < 2.0) & (np.fmin(dT[:, 0], dT[:, 1]) > 3)
         & (dist > 5) & (np.fmin(dB[:, 0], dB[:, 1]) < 1.5) & (d["bb"]["chamfer"] < 1.5))
p = np.concatenate(([0], ghost.astype(int), [0]))
e = np.flatnonzero(np.diff(p))
runs = []
for s0, e0 in zip(e[::2], e[1::2]):
    if runs and s0 - runs[-1][1] <= 5:
        runs[-1][1] = e0
    else:
        runs.append([s0, e0])
long_runs = [r for r in runs if r[1] - r[0] >= 25]
print("ghost frames %d; runs %d; runs >= 25 frames %d" % (ghost.sum(), len(runs), len(long_runs)))
for r in long_runs:
    w = slice(r[0], r[1])
    bt = np.where(dB[w, 0] <= dB[w, 1], "H", "A")
    print("   %d-%d (%d frames): |T-B| median %.1f m; broadcast-nearest team H %d / A %d; engine %s; public GT %s"
          % (r[0], r[1], r[1] - r[0], np.nanmedian(dist[w]), (bt == "H").sum(), (bt == "A").sum(),
             "".join(sorted(set(lab[w]))), "".join(sorted(set("HADX"[c] for c in gt[w])))))

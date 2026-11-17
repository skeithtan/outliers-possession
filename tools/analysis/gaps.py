"""What are the long ball gaps? Where does the ball vanish, and what are the players doing meanwhile?"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "cache", "atd_tactical.npz")
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/starter")
import load_smart_tagging as st
L, W = 105.3, 68.0
d = np.load(CACHE)
px, py, ps, bx, by, team = d["px"], d["py"], d["ps"], d["bx"], d["by"], d["team_of_slot"]
N = len(bx)
graded = np.zeros(N, bool); graded[11185:90652] = True; graded[110808:190198] = True
vis = ~np.isnan(bx)
miss = ~vis & graded
edges = np.diff(np.concatenate(([0], miss.astype(int), [0])))
starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
lens = stops - starts
long = lens > 50
print("long gaps (>2s): %d, covering %d graded frames (%.1f%%)" % (long.sum(), lens[long].sum(), 100 * lens[long].sum() / graded.sum()))

mean_speed = np.nanmean(ps, axis=1)
with np.errstate(all="ignore"):
    hx = np.nanmean(np.where(team[None, :] == 0, px, np.nan), 1) * L
    ax = np.nanmean(np.where(team[None, :] == 1, px, np.nan), 1) * L
sep = np.abs(hx - ax)

# set piece windows from smart tagging
ev = st.load()
sp_mask = np.zeros(N, bool)
for e in ev:
    if e["code"].lower() == "set pieces":
        sp_mask[max(0, e["start_frame"] - 250):e["end_frame"] + 50] = True

def zone(x, y):
    if np.isnan(x): return "unknown"
    if x < 16.5 or x > L - 16.5:
        return "penalty-area-x" if 13.8 < y < W - 13.8 else "corner/goal-line-flank"
    return "middle"

rows = []
for s, e, n in zip(starts[long], stops[long], lens[long]):
    lx, ly = bx[s - 1] * L, by[s - 1] * W
    rx, ry = bx[e] * L if e < N else np.nan, by[e] * W if e < N else np.nan
    jump = np.hypot(rx - lx, ry - ly) if not np.isnan(rx) else np.nan
    rows.append((s, n, zone(lx, ly), np.nanmean(mean_speed[s:e]), np.nanmean(sep[s:e]), sp_mask[s:e].mean(), jump,
                 (~np.isnan(px[s:e])).sum(1).mean()))
rows = np.array(rows, dtype=object)

print("\nball location just before vanishing (long gaps):")
for z in ("penalty-area-x", "corner/goal-line-flank", "middle", "unknown"):
    m = rows[:, 2] == z
    if m.sum():
        print("  %-24s n=%3d  frames=%6d  mean player speed %.2f  team sep %.1f m  in set-piece window %.0f%%" % (
            z, m.sum(), sum(rows[m, 1]), np.mean(rows[m, 3].astype(float)), np.mean(rows[m, 4].astype(float)),
            100 * np.mean(rows[m, 5].astype(float))))
print("\nreference on ball-visible graded frames: mean player speed %.2f, team sep %.1f m" % (
    np.nanmean(mean_speed[graded & vis]), np.nanmean(sep[graded & vis])))
print("ball displacement across the gap (where it reappears vs where it vanished): median %.1f m, p75 %.1f m" % (
    np.nanmedian(rows[:, 6].astype(float)), np.nanpercentile(rows[:, 6].astype(float), 75)))

# Gap-frame share by "activity": low speed (<1.6) suggests dead ball
gm = np.concatenate([np.arange(s, e) for s, e in zip(starts[long], stops[long])])
print("\nwithin long gaps: frames with mean player speed < 1.6 m/s: %.0f%%   (ball-visible graded reference: %.0f%%)" % (
    100 * np.nanmean(mean_speed[gm] < 1.6), 100 * np.nanmean(mean_speed[graded & vis] < 1.6)))
print("within long gaps: frames in a set-piece tag window: %.0f%%   (ball-visible reference: %.0f%%)" % (
    100 * sp_mask[gm].mean(), 100 * sp_mask[graded & vis].mean()))

# The 20 longest gaps
order = np.argsort(-rows[:, 1].astype(int))[:12]
print("\nlongest gaps: start, seconds, zone, mean speed, set-piece window share")
for i in order:
    print("  %6d  %5.1fs  %-22s  %.2f  %.0f%%" % (rows[i, 0], rows[i, 1] / 25, rows[i, 2], rows[i, 3], 100 * rows[i, 5]))

"""GK-swap follow-up, part 1: GK-like slot robustness and the frozen artefact.
Usage: python tools/analysis/r2_gk_frozen.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import r2_gk_common as G  # noqa: E402
from engine import data, tracks  # noqa: E402

L, W = data.pitch_m()
T = data.load_tactical()
blocks = data.graded_blocks()
px, py, ps = G.metres(T, L, W)
vis = np.isfinite(px)
team = np.asarray(T["team_of_slot"])
gks = G.gk_slots(T, blocks, L, W)
frozen = G.frozen_mask(T, L, W)

print("=== 1a. GK-like slot per graded block ===")
for (h, t), g in sorted(gks.items()):
    lo, hi = [(a, b) for hh, a, b in blocks if hh == h][0]
    order = np.argsort(-g["counts"])
    s1, s2 = g["cols"][order[0]], g["cols"][order[1]]
    starts, stops = tracks._runs(vis[lo:hi + 1, s1])
    big = np.sort(g["shares"][g["visible"] > 1500])[::-1]
    print("half %d team %s: own goal x=%.0f | GK-like slot %d: %d own-box frames, share of its visible frames %.2f, visible %.1f%% of block, %d tracklets"
          % (h, "HA"[t], g["own_x"], s1, g["counts"][order[0]], g["shares"][order[0]],
             100.0 * vis[lo:hi + 1, s1].mean(), len(starts)))
    print("     second-best slot %d: %d own-box frames (share %.2f) | best own-box share among other slots with > 60 s visible: %.2f"
          % (s2, g["counts"][order[1]], g["shares"][order[1]], big[1] if len(big) > 1 else np.nan))
    # GK-like set by share rule (substitution-robust): share >= 0.5 and >= 1500 own-box frames
    sel = g["cols"][(g["shares"] >= 0.5) & (g["counts"] >= 1500)]
    print("     share rule (own-box share >= 0.5 and >= 1500 own-box frames) selects slots %s" % list(sel))
    fr = frozen[lo:hi + 1, s1]
    print("     GK-like slot frozen on %d frames (%.1f%% of its visible); frozen frames inside own box %.0f%%"
          % (fr.sum(), 100.0 * fr.sum() / max(vis[lo:hi + 1, s1].sum(), 1),
             100.0 * np.mean(G.box_mask(px[lo:hi + 1, s1][fr], py[lo:hi + 1, s1][fr], g["own_side"], L, W)) if fr.any() else np.nan))

print("\n=== 1b. frozen artefact (visible, NaN speed, < %.2f m in %d frames), graded frames ===" % (G.FROZEN_MOVE_M, G.FROZEN_WIN))
graded = np.zeros(len(px), bool)
for _, lo, hi in blocks:
    graded[lo:hi + 1] = True
print("visible player-frames %d | NaN speed %d (%.1f%%) | frozen %d (%.1f%%)"
      % (vis[graded].sum(), (vis & np.isnan(ps))[graded].sum(), 100 * (vis & np.isnan(ps))[graded].sum() / vis[graded].sum(),
         frozen[graded].sum(), 100 * frozen[graded].sum() / vis[graded].sum()))
lens, slots_ep = [], []
for s in range(px.shape[1]):
    a, b = tracks._runs(frozen[:, s] & graded)
    lens += list(b - a)
    slots_ep += [s] * len(a)
lens = np.array(lens)
slots_ep = np.array(slots_ep)
gk_set = set(g["gk"] for g in gks.values())
isgk = np.isin(slots_ep, list(gk_set))
print("episodes %d; length p50 %d p90 %d p99 %d max %d frames; >= 25 frames: %d; >= 250 frames: %d"
      % (len(lens), np.median(lens), np.percentile(lens, 90), np.percentile(lens, 99), lens.max(), (lens >= 25).sum(), (lens >= 250).sum()))
print("GK-like slots: %d episodes, %d frames (%.0f%% of frozen frames); length p50 %d max %d"
      % (isgk.sum(), lens[isgk].sum(), 100.0 * lens[isgk].sum() / lens.sum(), np.median(lens[isgk]), lens[isgk].max()))

print("\n=== 1c. what is near a frozen slot (graded, per frozen frame, 5 m) ===")
live = vis & ~frozen
for (h, t), g in sorted(gks.items()):
    lo, hi = [(a, b) for hh, a, b in blocks if hh == h][0]
    o = gks[(h, 1 - t)]["gk"]
    s = g["gk"]
    f = np.arange(lo, hi + 1)
    fz = frozen[f, s]
    d_ogk = np.hypot(px[f, s] - px[f, o], py[f, s] - py[f, o])
    near_ogk = fz & live[f, o] & (d_ogk <= 5)
    dd = np.hypot(px[f] - px[f, s][:, None], py[f] - py[f, s][:, None])
    dd[:, s] = np.inf
    dd[~live[f]] = np.inf
    near_any = fz & (dd.min(1) <= 5)
    near_own = fz & (np.where(team[None, :] == t, dd, np.inf).min(1) <= 5)
    print("half %d team %s GK slot %d frozen %d frames: live other-team GK slot within 5 m %d (%.0f%%) | any live slot within 5 m %d (%.0f%%) | live own-team slot within 5 m %d"
          % (h, "HA"[t], s, fz.sum(), near_ogk.sum(), 100.0 * near_ogk.sum() / max(fz.sum(), 1), near_any.sum(),
             100.0 * near_any.sum() / max(fz.sum(), 1), near_own.sum()))
    # of the near-other-GK frames: is the live other-GK beyond its own half-way?
    beyond = np.abs(px[f, o] - gks[(h, 1 - t)]["own_x"]) > L / 2
    print("     of those, live other-team GK beyond its half-way %d; inside its OWN half %d"
          % ((near_ogk & beyond).sum(), (near_ogk & ~beyond).sum()))
    dist_b = d_ogk[fz & live[f, o] & beyond]
    if dist_b.size:
        print("     distance frozen GK -> live other GK when that one is beyond half-way: p10 %.1f p50 %.1f p90 %.1f, <=5 m %.0f%%, <=8 m %.0f%%, <=15 m %.0f%% (n=%d)"
              % (np.percentile(dist_b, 10), np.median(dist_b), np.percentile(dist_b, 90), 100 * np.mean(dist_b <= 5),
                 100 * np.mean(dist_b <= 8), 100 * np.mean(dist_b <= 15), dist_b.size))
# non-GK frozen slots: how often is a live other-team GK within 5 m?
nongk = frozen & graded[:, None]
for s in gk_set:
    nongk[:, s] = False
fr_idx, sl_idx = np.nonzero(nongk)
print("non-GK frozen player-frames %d" % len(fr_idx))

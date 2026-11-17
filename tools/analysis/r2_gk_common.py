"""GK-swap guard prototype helpers (research only, not engine code). Match-independent: no slot or
frame numbers. Everything is derived per graded block (a half) from the tactical ATD.

  box_mask(px, py, side, L, W)      inside the left (side=0) or right (side=1) penalty area
  frozen_mask(tactical, L, W)       visible, NaN speed, and < FROZEN_MOVE_M movement over the
                                    centred window of FROZEN_WIN frames (both ends visible)
  gk_slots(tactical, blocks, L, W)  per block and team: own goal side, the GK-like slot (most
                                    own-box frames), and the per-slot own-box counts / shares
  guard_frames(...)                 per-frame swap list for G1 / G2
  patched_tactical(tactical, swaps) live GK position moved into the frozen other-GK slot
"""
import numpy as np

PEN_DEPTH = 16.5        # m, Laws of the Game: penalty area depth
PEN_HALF_WIDTH = 20.16  # m, Laws of the Game: 40.32 m wide penalty area
FROZEN_WIN = 25         # frames (1 s): window of the frozen test
FROZEN_MOVE_M = 0.25    # m: a live player moves more than this in 1 s (walking 1 m/s = 1 m)


def metres(tactical, L, W):
    return (tactical["px"].astype(float) * L, tactical["py"].astype(float) * W,
            tactical["ps"].astype(float))


def box_mask(px, py, side, L, W):
    inx = px < PEN_DEPTH if side == 0 else px > L - PEN_DEPTH
    return inx & (np.abs(py - W / 2.0) < PEN_HALF_WIDTH)


def frozen_mask(tactical, L, W):
    px, py, ps = metres(tactical, L, W)
    vis = np.isfinite(px)
    h = FROZEN_WIN // 2
    n = px.shape[0]
    a = np.clip(np.arange(n) - h, 0, n - 1)
    b = np.clip(np.arange(n) + h, 0, n - 1)
    move = np.hypot(px[b] - px[a], py[b] - py[a])
    return vis & np.isnan(ps) & np.isfinite(move) & (move < FROZEN_MOVE_M)


def gk_slots(tactical, blocks, L, W):
    """{(half, team): dict(own_side, own_x, gk, counts, shares, visible)} per graded block."""
    px, py, _ = metres(tactical, L, W)
    vis = np.isfinite(px)
    team = np.asarray(tactical["team_of_slot"])
    out = {}
    for h, lo, hi in blocks:
        s = slice(lo, hi + 1)
        left = (vis[s] & box_mask(px[s], py[s], 0, L, W)).sum(0)
        right = (vis[s] & box_mask(px[s], py[s], 1, L, W)).sum(0)
        nvis = vis[s].sum(0)
        for t in (0, 1):
            cols = np.flatnonzero(team == t)
            side = 0 if left[cols].max() > right[cols].max() else 1
            cnt = (left if side == 0 else right)[cols]
            gk = int(cols[np.argmax(cnt)])
            with np.errstate(all="ignore"):
                share = np.where(nvis[cols] > 0, cnt / nvis[cols], 0.0)
            out[(h, t)] = {"own_side": side, "own_x": 0.0 if side == 0 else L, "gk": gk,
                           "cols": cols, "counts": cnt, "shares": share, "visible": nvis[cols]}
    return out


def guard_frames(tactical, blocks, L, W, R, need_beyond, gks=None, frozen=None):
    """[(frame, live_slot, frozen_slot)] where the guard fires.

    Fires on frame f of block b for team T when: T's GK-like slot is visible and not frozen; the
    other team's GK-like slot is frozen at f and within R m of it; and (need_beyond) T's slot is
    more than half a pitch length from T's own goal line.
    """
    px, py, _ = metres(tactical, L, W)
    vis = np.isfinite(px)
    gks = gks or gk_slots(tactical, blocks, L, W)
    frozen = frozen if frozen is not None else frozen_mask(tactical, L, W)
    out = []
    for h, lo, hi in blocks:
        f = np.arange(lo, hi + 1)
        for t in (0, 1):
            g, o = gks[(h, t)]["gk"], gks[(h, 1 - t)]["gk"]
            d = np.hypot(px[f, g] - px[f, o], py[f, g] - py[f, o])
            ok = vis[f, g] & ~frozen[f, g] & frozen[f, o] & (d <= R)
            if need_beyond:
                ok &= np.abs(px[f, g] - gks[(h, t)]["own_x"]) > L / 2.0
            out += [(int(x), g, o) for x in f[ok]]
    out.sort()
    return out


def patched_tactical(tactical, swaps):
    """Copy of the tactical dict: on each swap frame the live slot's x, y, speed go into the frozen
    slot (the other team's GK slot) and the live slot becomes missing. Team counts are kept."""
    out = dict(tactical)
    px, py, ps = tactical["px"].copy(), tactical["py"].copy(), tactical["ps"].copy()
    for f, g, o in swaps:
        px[f, o], py[f, o], ps[f, o] = px[f, g], py[f, g], ps[f, g]
        px[f, g] = py[f, g] = ps[f, g] = np.nan
    out.update(px=px, py=py, ps=ps)
    return out

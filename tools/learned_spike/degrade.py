"""Synthetic degradation of clean external tracking, for training only. Not part of the submission.

The open DFL tracking is clean. The competition tracking is automatic and patchy: the ball is
missing on about 41 % of graded frames, mostly in long gaps during stoppages (PLAN.md 1.1, a
label-free statistic). This module makes one degraded copy of a training match:

1. Dead-ball blanking. In each reference D segment, with probability P_DEAD_BLANK, the ball is
   hidden from a random point in the first DEAD_LEAD_MAX_S seconds to the end of the segment.
2. Live gaps. Gaps of log-uniform length LIVE_GAP_S are placed uniformly on alive frames until
   LIVE_GAP_SHARE of the alive frames are hidden.
3. Track fragments. Each player slot loses chunks of log-uniform length PLAYER_GAP_S until
   PLAYER_DROP_SHARE of its visible frames are hidden.

The reference labels place the dead-ball blanking. That is allowed: the copy is training data
of an external match, and the model never sees a label at inference. All draws use one
numpy RandomState per match with a fixed seed. Python 3.9, numpy only.
"""

import zlib

import numpy as np

P_DEAD_BLANK = 0.7
DEAD_LEAD_MAX_S = 1.5
LIVE_GAP_S = (1.0, 12.0)
LIVE_GAP_SHARE = 0.10
PLAYER_GAP_S = (1.0, 8.0)
PLAYER_DROP_SHARE = 0.08
FPS = 25
D_CODE = 2
ALIVE = (0, 1)


def seed_of(game):
    """A fixed seed per match name."""
    return zlib.crc32(game.encode("utf-8")) & 0x7FFFFFFF


def _runs(mask):
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2]


def _loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def degrade(tac, codes, game):
    """A degraded copy of a tactical dict (new px, py, bx, by arrays). codes: reference codes."""
    rng = np.random.RandomState(seed_of(game))
    out = dict(tac)
    bx = np.array(tac["bx"], dtype=np.float32)
    by = np.array(tac["by"], dtype=np.float32)
    n = len(bx)
    codes = np.asarray(codes)
    starts, stops = _runs(codes == D_CODE)
    for a, b in zip(starts, stops):
        if rng.uniform() < P_DEAD_BLANK:
            s = a + int(rng.uniform(0, DEAD_LEAD_MAX_S * FPS))
            bx[s:b] = np.nan
            by[s:b] = np.nan
    alive = np.isin(codes, ALIVE)
    target = int(LIVE_GAP_SHARE * alive.sum())
    hidden = 0
    tries = 0
    alive_idx = np.flatnonzero(alive)
    while hidden < target and tries < 100000 and alive_idx.size:
        tries += 1
        length = int(_loguniform(rng, *LIVE_GAP_S) * FPS)
        s = int(alive_idx[rng.randint(0, alive_idx.size)])
        e = min(n, s + length)
        newly = np.isfinite(bx[s:e]).sum()
        bx[s:e] = np.nan
        by[s:e] = np.nan
        hidden += int(newly)
    px = np.array(tac["px"], dtype=np.float32)
    py = np.array(tac["py"], dtype=np.float32)
    for slot in range(px.shape[1]):
        vis = np.isfinite(px[:, slot])
        target = int(PLAYER_DROP_SHARE * vis.sum())
        idx = np.flatnonzero(vis)
        dropped = 0
        tries = 0
        while dropped < target and tries < 10000 and idx.size:
            tries += 1
            length = int(_loguniform(rng, *PLAYER_GAP_S) * FPS)
            s = int(idx[rng.randint(0, idx.size)])
            e = min(n, s + length)
            dropped += int(np.isfinite(px[s:e, slot]).sum())
            px[s:e, slot] = np.nan
            py[s:e, slot] = np.nan
    out.update(px=px, py=py, bx=bx, by=by)
    return out

"""Defect augmentation of clean external tracking, for training only (round 3, switch AUG).

Not part of the submission. The module makes K defected copies ("aug views") of a training
match. Each copy draws its own defect mix and rates from WIDE GENERIC RANGES (constants below),
then draws every event afresh. The sources of every range are in
docs/plan/calibration/R3_AUG.md. No range is read from, or fitted to, the provided match.

Defects (each copy switches each component on with probability P_ON, then draws its rate):

1. dead   Dead-ball blanking (degrade.py step 1, widened). In each reference D segment, with
          probability p, the ball is hidden from a point in the first DEAD_LEAD_MAX_S seconds to
          the segment end, and the return is moved by a uniform RETURN_JITTER_S: the ball comes
          back up to 0.5 s before the restart or up to 0.5 s after it (the review's point).
2. live   Live ball gaps (degrade.py step 2, widened). Log-uniform lengths LIVE_GAP_S until a
          share of the alive frames is hidden. A gap starts, with probability p_crowd, on a frame
          where at least CROWD_N players stand within CROWD_R_M of the ball (occlusion), else
          on a uniform alive frame.
3. drop   Single-player drop-outs (degrade.py step 3, widened): per slot, log-uniform chunks
          PLAYER_GAP_S until a share of its visible frames is hidden.
4. group  Collective drop-outs: an event hides the M players of one team farthest from the ball
          (the broadcast camera follows the ball) for a log-uniform GROUP_S.
5. wrong  Wrong-team stretches: an event moves one player (with probability NEAR_BALL_P the one
          nearest the ball) into a free slot of the other team for a log-uniform WRONG_S. The
          other team then shows an extra player (over-count) and his own team one fewer.
6. ghost  Duplicated players: an event copies one player (with probability NEAR_BALL_P the one
          nearest the ball), moved by GHOST_OFFSET_M, into a free slot of his team for a
          log-uniform GHOST_S (over-count).
7. reslot Fragmentation into new slots: every visible run of a slot after a gap moves, with
          probability p, to the least recently used free slot of its team (each team gets
          EXTRA_SLOTS extra columns, which components 5 and 6 also use).

8. ball   Ball smoothing and offset (round 4, switch BALL, off by default; ranges and sources in
          docs/plan/calibration/R4_AUG_BALL.md, measured on external matches only). Metrica and
          DFL copies only: SkillCorner's ball is already a broadcast ball and stays as it is.
          The visible ball is Gaussian-smoothed with BALL_SIGMA frames inside each visible run
          (a run ends at a missing frame and at a graded block edge; atd_view.smooth_runs), then
          moved by a slowly varying offset: per axis Gaussian-filtered white noise of time
          scale BALL_TAU_S, scaled so that the offset vector has RMS length BALL_OFFSET_M and
          clipped at BALL_OFFSET_CAP times that length. It runs after component 2, so a missing
          frame stays missing and a visible one stays visible. Its draws use their own seeds
          ("ballplan", "ball"), so with the switch off every copy is byte-identical to round 3,
          and the players (hence the ball-free twins) are the same with the switch on.

9. flicker Short breaks in tracks (round 5, switch X = plan version "x", off by default; ranges
          and sources in docs/plan/calibration/R5_AUG.md). Each slot's visible frames get gaps of
          log-uniform FLICKER_GAP_S at a per-copy rate FLICKER_PER_SLOT_MIN (gaps per slot per
          minute of visible time); every piece that starts right after such a gap moves to a new
          slot by component 7's rule (least recently used free column of its team), always.
          With version "x", component 6 also draws its offset from GHOST_OFFSET_M_X (rate and
          duration keep round 3's ranges; the plan's random stream is unchanged).
          Its draws use their own seeds (FLICKER_SEEDS), so with version "r3" (the default)
          every copy is byte-identical to round 4's.

Left out on purpose: identity swaps of two nearby players (corrupt.py component (b), same-team
and cross-team). The stress view keeps them, so its swap part stays a defect the model never
trained on (R3_AUG.md Section 4).

Seeds. Copy k of match g draws from numpy RandomState(seed_of(g, k, component)), a crc32 of
the string "r3aug|<g>|<k>|<component>". check_seeds() proves that no such seed equals a seed
of corrupt.py's stress view (crc32 of the name, or 0..N, plus its component offsets).

The reference labels place component 1 only. That is allowed for training data of an external
match, as in degrade.py: the model never sees a label at inference. Python 3.9, numpy only.
"""

import zlib

import numpy as np

FPS = 25
D_CODE = 2
ALIVE = (0, 1)

P_ON = 0.75                      # each component is on in a copy with this probability
# 1. dead-ball blanking
P_DEAD = (0.5, 0.9)              # per D segment; degrade.P_DEAD_BLANK = 0.7 at the centre
DEAD_LEAD_MAX_S = 1.5            # degrade.DEAD_LEAD_MAX_S
RETURN_JITTER_S = 0.5            # review W5: return moved by U(-0.5, +0.5) s
# 2. live ball gaps
LIVE_SHARE = (0.0, 0.25)         # degrade.LIVE_GAP_SHARE = 0.10, widened
LIVE_GAP_S = (1.0, 30.0)         # degrade 1 s low end; 30 s = corrupt.UNIFORM30 / stress.py top
P_CROWD = (0.0, 1.0)             # share of live gaps placed on crowded frames
CROWD_N = 3                      # players (any team) within CROWD_R_M of the ball
CROWD_R_M = 3.0                  # features.RADII middle radius
# 3. single-player drop-outs
DROP_SHARE = (0.0, 0.20)         # degrade.PLAYER_DROP_SHARE = 0.08, widened
PLAYER_GAP_S = (1.0, 30.0)       # degrade 1 s low end; 30 s top as above
# 4. collective drop-outs
GROUP_PER_10MIN = (0.0, 10.0)    # judgement: up to one per minute
GROUP_M = (3, 11)                # several players to the whole team (inclusive)
GROUP_S = (1.0, 15.0)            # judgement: a camera cut or pan of 1-15 s
# 5. wrong-team stretches
WRONG_PER_10MIN = (0.0, 20.0)    # judgement: up to one per 30 s
WRONG_S = (0.5, 20.0)            # judgement
NEAR_BALL_P = 0.5                # half the events take the player nearest the ball
# 6. ghosts
GHOST_PER_10MIN = (0.0, 20.0)    # judgement, same scale as 5
GHOST_S = (1.0, 10.0)            # corrupt.GHOST_S = (1, 3) s, widened
GHOST_OFFSET_M = (0.1, 1.0)      # corrupt.GHOST_OFFSET_M = 0.3 m, widened
# 7. re-slotting
P_RESLOT = (0.0, 1.0)
EXTRA_SLOTS = 11                 # one team's worth of extra columns per team

# 8. ball smoothing and offset (round 4; R4_AUG_BALL.md, measured on external matches only)
BALL_SIGMA = (0.0, 4.0)          # frames at 25 fps; SkillCorner-equivalent sigma 2.99-3.91
BALL_OFFSET_M = (0.0, 1.5)       # RMS length of the offset vector, m; SC-equivalent 0-1.51
BALL_TAU_S = (1.0, 5.0)          # time scale of the offset, seconds, log-uniform (judgement)
BALL_OFFSET_CAP = 2.0            # |offset| <= BALL_OFFSET_CAP * RMS length
BALL_TRUNCATE = 4.0              # kernel truncated at 4 sigma (atd_view.TRUNCATE_SIGMAS)
BALL_PROVIDERS = ("METRICA-", "DFL-")
BALL_SEEDS = ("ballplan", "ball")

# 9. round 5 switch X (plan version "x"; R5_AUG.md): component 6 widened, flicker added.
# Generic judgement ranges; the flicker rate must cover SkillCorner's measured tracklet births.
PLAN_VERSIONS = ("r3", "x")
# Narrowed after the independent review of 13 Nov 2026: rate and duration stay at round 3's
# ranges; only the offset widens (duplicates farther apart than the feature cleaning's 0.5 m).
GHOST_PER_10MIN_X = GHOST_PER_10MIN   # U(0, 20) per 10 min, as round 3
GHOST_S_X = GHOST_S                   # log-uniform 1-10 s, as round 3
GHOST_OFFSET_M_X = (0.1, 1.5)         # judgement: round 3's 0.1-1.0 m widened to 1.5 m
FLICKER_PER_SLOT_MIN = (0.0, 6.0)  # gaps per slot per minute of visible time (judgement)
FLICKER_GAP_S = (0.2, 2.0)       # log-uniform (judgement)
FLICKER_SEEDS = ("flickerplan", "flicker")

COMPONENTS = ("dead", "live", "drop", "group", "wrong", "ghost", "reslot")
LEFT_OUT = ("swap",)
MAX_TRIES = 100000

# corrupt.py seed offsets (SEED_FRAG, SEED_FLIP, SEED_SWAP, SEED_BALL, SEED_GHOST,
# SEED_SWAP_CROSS). Copied, not imported, so that this module stays numpy only; a unit test
# checks they equal corrupt.py's.
STRESS_OFFSETS = (11, 13, 17, 19, 23, 29)


def seed_of(game, k, component):
    """The seed of one component of copy k of a match."""
    key = "r3aug|{}|{}|{}".format(game, int(k), component)
    return zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF


def stress_seeds(game, bases=range(0, 100)):
    """Every seed that corrupt.py's stress view uses on `game`: crc32 of the name or a base."""
    roots = [zlib.crc32(game.encode("utf-8"))] + list(bases)
    return {(r + o) & 0x7FFFFFFF for r in roots for o in STRESS_OFFSETS}


def check_seeds(games, k_views, ball=False, flicker=False):
    """Raise AssertionError if a seed of a copy equals a stress-view seed, or if two components
    of one copy share a seed. Return the count.

    ball=True also checks the seeds of component 8 (BALL_SEEDS); flicker=True those of
    component 9 (FLICKER_SEEDS, switch X)."""
    n = 0
    names = COMPONENTS + ("plan",) + (BALL_SEEDS if ball else ()) \
        + (FLICKER_SEEDS if flicker else ())
    for g in games:
        bad = stress_seeds(g)
        for k in range(k_views):
            seen = set()
            for c in names:
                s = seed_of(g, k, c)
                assert s not in bad, (g, k, c)
                assert s not in seen, (g, k, c, "seed shared within a copy")
                seen.add(s)
                n += 1
    return n


def _loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _runs(mask):
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2]


def plan(game, k, version="r3"):
    """{component: rate dict or None}: the defect mix of copy k (None = component off).

    version "x" (switch X): the ghost rate is drawn from GHOST_PER_10MIN_X (since the review of
    13 Nov 2026 equal to round 3's range), with the same random number as version "r3".
    """
    if version not in PLAN_VERSIONS:
        raise ValueError("plan version {!r} not in {}".format(version, PLAN_VERSIONS))
    ghost_range = GHOST_PER_10MIN_X if version == "x" else GHOST_PER_10MIN
    rng = np.random.RandomState(seed_of(game, k, "plan"))
    out = {}
    for c in COMPONENTS:
        on = rng.uniform() < P_ON
        # Draw the rates whether or not the component is on, so the draws do not shift.
        rates = {
            "dead": {"p": rng.uniform(*P_DEAD)},
            "live": {"share": rng.uniform(*LIVE_SHARE), "p_crowd": rng.uniform(*P_CROWD)},
            "drop": {"share": rng.uniform(*DROP_SHARE)},
            "group": {"per_10min": rng.uniform(*GROUP_PER_10MIN)},
            "wrong": {"per_10min": rng.uniform(*WRONG_PER_10MIN)},
            "ghost": {"per_10min": rng.uniform(*ghost_range)},
            "reslot": {"p": rng.uniform(*P_RESLOT)},
        }[c]
        out[c] = rates if on else None
    return out


def _ball_metres(bx, by, pitch):
    return np.asarray(bx, dtype=np.float64) * pitch[0], np.asarray(by, dtype=np.float64) * pitch[1]


def hide_dead(bx, by, codes, rng, p, fps=FPS):
    """Component 1. Hide the ball in D segments; the return moves by +-RETURN_JITTER_S."""
    n = len(bx)
    hidden = 0
    starts, stops = _runs(np.asarray(codes) == D_CODE)
    jit = int(round(RETURN_JITTER_S * fps))
    for a, b in zip(starts, stops):
        go = rng.uniform() < p
        lead = int(rng.uniform(0, DEAD_LEAD_MAX_S * fps))
        shift = int(rng.randint(-jit, jit + 1))
        if not go:
            continue
        s = min(int(a) + lead, int(b) - 1)
        e = int(np.clip(int(b) + shift, s + 1, n))
        hidden += int(np.isfinite(bx[s:e]).sum())
        bx[s:e] = np.nan
        by[s:e] = np.nan
    return hidden


def crowd_frames(px, py, bx, by, pitch, n_min=CROWD_N, radius=CROWD_R_M):
    """bool (N,): frames with a visible ball and at least n_min players within radius metres."""
    xm, ym = px.astype(np.float64) * pitch[0], py.astype(np.float64) * pitch[1]
    bxm, bym = _ball_metres(bx, by, pitch)
    with np.errstate(invalid="ignore"):
        d = np.hypot(xm - bxm[:, None], ym - bym[:, None])
        close = (d <= radius).sum(axis=1)
    return np.isfinite(bxm) & (close >= n_min)


def hide_live(bx, by, codes, crowd, rng, share, p_crowd, fps=FPS):
    """Component 2. Live gaps until `share` of the alive frames with a ball are hidden."""
    n = len(bx)
    alive = np.isin(np.asarray(codes), ALIVE)
    target = int(share * (alive & np.isfinite(bx)).sum())
    pools = (np.flatnonzero(alive & crowd), np.flatnonzero(alive))
    hidden = 0
    for _ in range(MAX_TRIES):
        if hidden >= target or pools[1].size == 0:
            break
        length = max(1, int(_loguniform(rng, *LIVE_GAP_S) * fps))
        pool = pools[0] if (rng.uniform() < p_crowd and pools[0].size) else pools[1]
        s = int(pool[rng.randint(pool.size)])
        e = min(n, s + length)
        hidden += int(np.isfinite(bx[s:e]).sum())
        bx[s:e] = np.nan
        by[s:e] = np.nan
    return hidden


def drop_players(px, py, rng, share, fps=FPS):
    """Component 3. Per slot, log-uniform chunks until `share` of its visible frames hide."""
    n = px.shape[0]
    total = 0
    for slot in range(px.shape[1]):
        vis = np.isfinite(px[:, slot])
        target = int(share * vis.sum())
        idx = np.flatnonzero(vis)
        dropped = 0
        for _ in range(10000):
            if dropped >= target or idx.size == 0:
                break
            length = max(1, int(_loguniform(rng, *PLAYER_GAP_S) * fps))
            s = int(idx[rng.randint(idx.size)])
            e = min(n, s + length)
            dropped += int(np.isfinite(px[s:e, slot]).sum())
            px[s:e, slot] = np.nan
            py[s:e, slot] = np.nan
        total += dropped
    return total


def _events(rng, graded_idx, per_10min, fps):
    """Sorted start frames of a Poisson number of events over the graded frames."""
    if per_10min <= 0 or graded_idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    count = int(rng.poisson(graded_idx.size / float(fps) / 600.0 * per_10min))
    return np.sort(graded_idx[rng.randint(0, graded_idx.size, size=count)])


def _ball_at(bxm, bym, t):
    """The ball position at frame t, or the last visible one before it, or None."""
    vis = np.flatnonzero(np.isfinite(bxm[:t + 1]))
    if vis.size == 0:
        return None
    j = int(vis[-1])
    return bxm[j], bym[j]


def group_dropouts(px, py, team, bx_ref, by_ref, graded_idx, rng, per_10min, pitch, fps=FPS):
    """Component 4. Hide the M players of one team farthest from the ball. Returns events."""
    n = px.shape[0]
    bxm, bym = _ball_metres(bx_ref, by_ref, pitch)
    done = 0
    for t in _events(rng, graded_idx, per_10min, fps):
        side = int(rng.randint(2))
        m = int(rng.randint(GROUP_M[0], GROUP_M[1] + 1))
        e = min(n, int(t) + max(1, int(_loguniform(rng, *GROUP_S) * fps)))
        cols = np.flatnonzero((team == side) & np.isfinite(px[t]))
        if cols.size == 0:
            continue
        ball = _ball_at(bxm, bym, int(t))
        if ball is None:
            order = cols[rng.permutation(cols.size)]
        else:
            d = np.hypot(px[t, cols] * pitch[0] - ball[0], py[t, cols] * pitch[1] - ball[1])
            order = cols[np.argsort(-d, kind="stable")]
        hide = order[:m]
        px[t:e, hide] = np.nan
        py[t:e, hide] = np.nan
        done += 1
    return done


def _free_slot(px, team, side, a, b, rng=None):
    """The lowest column of team `side` that is invisible on frames a-1 .. b, or -1."""
    lo, hi = max(0, a - 1), min(px.shape[0], b + 1)
    free = np.flatnonzero((team == side) & ~np.isfinite(px[lo:hi]).any(axis=0))
    return int(free[0]) if free.size else -1


def _pick_player(px, py, cols, bxm, bym, t, rng, pitch):
    """A visible column at frame t: nearest the ball with probability NEAR_BALL_P."""
    near = rng.uniform() < NEAR_BALL_P
    pick = int(cols[rng.randint(cols.size)])
    ball = _ball_at(bxm, bym, int(t))
    if near and ball is not None:
        d = np.hypot(px[t, cols] * pitch[0] - ball[0], py[t, cols] * pitch[1] - ball[1])
        pick = int(cols[int(np.argmin(d))])
    return pick


def wrong_team(px, py, ps, team, bx_ref, by_ref, graded_idx, rng, per_10min, pitch, fps=FPS):
    """Component 5. Move one player into a free slot of the other team. Returns events."""
    n = px.shape[0]
    bxm, bym = _ball_metres(bx_ref, by_ref, pitch)
    done = 0
    for t in _events(rng, graded_idx, per_10min, fps):
        t = int(t)
        e = min(n, t + max(1, int(_loguniform(rng, *WRONG_S) * fps)))
        cols = np.flatnonzero(np.isfinite(px[t]))
        if cols.size == 0:
            continue
        src = _pick_player(px, py, cols, bxm, bym, t, rng, pitch)
        dst = _free_slot(px, team, 1 - team[src], t, e)
        if dst < 0:
            continue
        for arr in (px, py, ps):
            arr[t:e, dst] = arr[t:e, src]
            arr[t:e, src] = np.nan
        done += 1
    return done


def add_ghosts(px, py, ps, team, bx_ref, by_ref, graded_idx, rng, per_10min, pitch, fps=FPS,
               ghost_s=GHOST_S, offset_m=GHOST_OFFSET_M):
    """Component 6. Copy one player, moved by `offset_m` (range), into a free slot of his team
    for a log-uniform `ghost_s` (range). Defaults: the round 3 ranges."""
    n = px.shape[0]
    bxm, bym = _ball_metres(bx_ref, by_ref, pitch)
    done = 0
    for t in _events(rng, graded_idx, per_10min, fps):
        t = int(t)
        e = min(n, t + max(1, int(_loguniform(rng, *ghost_s) * fps)))
        off = rng.uniform(*offset_m)
        angle = rng.uniform(0, 2 * np.pi)
        cols = np.flatnonzero(np.isfinite(px[t]))
        if cols.size == 0:
            continue
        src = _pick_player(px, py, cols, bxm, bym, t, rng, pitch)
        dst = _free_slot(px, team, team[src], t, e)
        if dst < 0:
            continue
        px[t:e, dst] = px[t:e, src] + off * np.cos(angle) / pitch[0]
        py[t:e, dst] = py[t:e, src] + off * np.sin(angle) / pitch[1]
        ps[t:e, dst] = ps[t:e, src]
        done += 1
    return done


def reslot(px, py, ps, team, rng, p):
    """Component 7. Each visible run after a slot's first moves to a new slot with prob p.

    The new slot is the least recently used column of the same team that is free on the run
    and on one frame either side. Returns the number of runs moved.
    """
    n = px.shape[0]
    runs = []
    for col in range(px.shape[1]):
        s, e = _runs(np.isfinite(px[:, col]))
        runs += [(int(a), int(b), col) for a, b in list(zip(s, e))[1:]]
    runs.sort()
    last_use = np.full(px.shape[1], -1, dtype=np.int64)
    moved = 0
    for a, b, col in runs:
        if not rng.uniform() < p:
            continue
        moved += int(_move_run(px, py, ps, team, a, b, col, last_use))
    return moved


def _move_run(px, py, ps, team, a, b, col, last_use):
    """Component 7's rule: move frames a..b-1 of column `col` to the least recently used column
    of the same team that is free on a-1 .. b. Return True if moved."""
    n = px.shape[0]
    lo, hi = max(0, a - 1), min(n, b + 1)
    free = np.flatnonzero((team == team[col]) & ~np.isfinite(px[lo:hi]).any(axis=0))
    free = free[free != col]
    if free.size == 0:
        return False
    dst = int(free[np.argmin(last_use[free])])
    for arr in (px, py, ps):
        arr[a:b, dst] = arr[a:b, col]
        arr[a:b, col] = np.nan
    last_use[dst] = b
    return True


def plan_flicker(game, k):
    """The rate dict of component 9 (switch X) in copy k, or None (off). Own seed."""
    rng = np.random.RandomState(seed_of(game, k, "flickerplan"))
    on = rng.uniform() < P_ON
    rates = {"per_slot_min": float(rng.uniform(*FLICKER_PER_SLOT_MIN))}
    return rates if on else None


def flicker(px, py, ps, team, rng, per_slot_min, fps=FPS):
    """Component 9. Break visible runs with short gaps, then re-slot the pieces.

    For each column, a Poisson number of gaps (mean: visible minutes x per_slot_min) starts on
    uniform visible frames, each of log-uniform FLICKER_GAP_S. Then every visible run that
    starts on the frame right after a gap of this component moves by component 7's rule
    (_move_run), in order of its first frame. Returns {"gaps", "pieces", "moved"}."""
    n, n_cols = px.shape
    ends = []
    gaps = 0
    for col in range(n_cols):
        idx = np.flatnonzero(np.isfinite(px[:, col]))
        if idx.size == 0 or per_slot_min <= 0:
            continue
        count = int(rng.poisson(idx.size / float(fps) / 60.0 * per_slot_min))
        for s in np.sort(idx[rng.randint(0, idx.size, size=count)]):
            e = min(n, int(s) + max(1, int(round(_loguniform(rng, *FLICKER_GAP_S) * fps))))
            px[s:e, col] = np.nan
            py[s:e, col] = np.nan
            ps[s:e, col] = np.nan
            ends.append((e, col))
            gaps += 1
    pieces = []
    for e, col in sorted(set(ends)):
        if e < n and np.isfinite(px[e, col]):
            stop = e + int(np.argmax(~np.isfinite(np.append(px[e:, col], np.nan))))
            pieces.append((e, stop, col))
    last_use = np.full(n_cols, -1, dtype=np.int64)
    moved = sum(int(_move_run(px, py, ps, team, a, b, col, last_use))
                for a, b, col in sorted(pieces))
    return {"gaps": gaps, "pieces": len(pieces), "moved": moved}


def ball_applies(game):
    """Component 8 changes Metrica and DFL copies only."""
    return str(game).startswith(BALL_PROVIDERS)


def plan_ball(game, k):
    """The rate dict of component 8 in copy k, or None (off). Own seed: plan() is unchanged."""
    rng = np.random.RandomState(seed_of(game, k, "ballplan"))
    on = rng.uniform() < P_ON
    rates = {"sigma": float(rng.uniform(*BALL_SIGMA)),
             "offset_m": float(rng.uniform(*BALL_OFFSET_M)),
             "tau_s": float(_loguniform(rng, *BALL_TAU_S))}
    return rates if on else None


def gaussian_kernel(sigma, truncate=BALL_TRUNCATE):
    """Unnormalised Gaussian weights over -r..r frames, r = ceil(truncate * sigma)."""
    radius = int(np.ceil(truncate * sigma))
    lags = np.arange(-radius, radius + 1, dtype=np.float64)
    return np.exp(-0.5 * (lags / sigma) ** 2)


def run_breaks(blocks, n):
    """bool (N,): True on the first frame of each graded block and on the frame after it."""
    br = np.zeros(n + 1, dtype=bool)
    for _, first, last in blocks:
        br[int(first)] = True
        br[min(int(last) + 1, n)] = True
    return br[:n]


def smooth_runs(values, visible, breaks, sigma):
    """Gaussian-smoothed copy of a (N,) series inside each run of `visible` cut at `breaks`.

    Same rule as tools/atd_view.smooth_runs (a unit test checks they agree): the truncated
    kernel is renormalised over the frames of the run it covers; other frames keep their value.
    """
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    if sigma <= 0:
        return out
    kernel = gaussian_kernel(sigma)
    radius = (kernel.size - 1) // 2
    visible = np.asarray(visible, dtype=bool)
    breaks = np.asarray(breaks, dtype=bool)
    prev = np.concatenate([[False], visible[:-1]])
    starts = np.flatnonzero(visible & (~prev | breaks))
    goes_on = np.zeros(values.size, dtype=bool)
    goes_on[:-1] = visible[1:] & ~breaks[1:]
    stops = np.flatnonzero(visible & ~goes_on) + 1
    for start, stop in zip(starts, stops):
        seg = values[start:stop]
        num = np.convolve(seg, kernel, mode="full")[radius:radius + seg.size]
        den = np.convolve(np.ones(seg.size), kernel, mode="full")[radius:radius + seg.size]
        out[start:stop] = num / den
    return out


def slow_offset(n, rng, rms_m, tau_frames, cap=BALL_OFFSET_CAP):
    """(ox, oy) metres, (N,) each: Gaussian-filtered white noise per axis, vector RMS `rms_m`.

    Each axis is white noise convolved with a Gaussian of sigma `tau_frames`, divided by its
    theoretical standard deviation, times rms_m / sqrt(2). The vector is clipped at cap * rms_m.
    """
    if rms_m <= 0 or n == 0:
        return np.zeros(n), np.zeros(n)
    kernel = gaussian_kernel(max(float(tau_frames), 1e-6))
    kernel = kernel / np.sqrt((kernel ** 2).sum())
    radius = (kernel.size - 1) // 2
    out = []
    for _ in range(2):
        w = rng.standard_normal(n + 2 * radius)
        out.append(np.convolve(w, kernel, mode="valid")[:n] * rms_m / np.sqrt(2.0))
    ox, oy = out
    norm = np.hypot(ox, oy)
    limit = cap * rms_m
    scale = np.where(norm > limit, limit / np.maximum(norm, 1e-12), 1.0)
    return ox * scale, oy * scale


def smooth_offset_ball(bx, by, blocks, pitch, rates, rng, fps=FPS):
    """Component 8 on normalised ball arrays: (bx, by, report). Missing frames stay missing."""
    bx = np.asarray(bx, dtype=np.float32)
    by = np.asarray(by, dtype=np.float32)
    n = len(bx)
    vis = np.isfinite(bx) & np.isfinite(by)
    xm, ym = _ball_metres(bx, by, pitch)
    br = run_breaks(blocks, n)
    xs = smooth_runs(xm, vis, br, rates["sigma"])
    ys = smooth_runs(ym, vis, br, rates["sigma"])
    ox, oy = slow_offset(n, rng, rates["offset_m"], rates["tau_s"] * fps)
    xs = np.where(vis, xs + ox, np.nan)
    ys = np.where(vis, ys + oy, np.nan)
    with np.errstate(invalid="ignore"):
        shift = np.hypot(xs - xm, ys - ym)[vis]
    rep = {"visible": int(vis.sum()),
           "shift_p50_m": float(np.median(shift)) if shift.size else 0.0,
           "offset_max_m": float(np.hypot(ox, oy)[vis].max()) if vis.any() else 0.0}
    return (xs / pitch[0]).astype(np.float32), (ys / pitch[1]).astype(np.float32), rep


def graded_index(blocks, n):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    return np.flatnonzero(mask)


def augment(tac, codes, game, k, blocks, pitch, fps=FPS, only=None, ball=False, version="r3"):
    """(copy k of a tactical dict, report). The input dict is not changed.

    codes: reference codes at 25 fps (component 1 only). blocks: graded blocks
    [(half, first, last)]. pitch: (length, width) metres. `only` restricts the copy to one
    component at a given rate dict (the unit tests), e.g. {"wrong": {"per_10min": 5}}; with
    ball=True, only["ball"] (if given) is the rate dict of component 8.
    ball: switch BALL (component 8, round 4). Off: the copy is the round 3 copy, byte for byte.
    version: "r3" (default, the round 4 copies byte for byte) or "x" (switch X, round 5:
    component 6 from the GHOST_*_X ranges and component 9 flicker; with `only`, only["flicker"]
    is the rate dict of component 9).
    """
    if version not in PLAN_VERSIONS:
        raise ValueError("plan version {!r} not in {}".format(version, PLAN_VERSIONS))
    pl = plan(game, k, version) if only is None else {c: only.get(c) for c in COMPONENTS}
    team0 = np.asarray(tac["team_of_slot"])
    n, n0 = np.asarray(tac["px"]).shape
    extra = np.concatenate([np.zeros(EXTRA_SLOTS, dtype=team0.dtype),
                            np.ones(EXTRA_SLOTS, dtype=team0.dtype)])
    team = np.concatenate([team0, extra])
    pad = np.full((n, extra.size), np.nan, dtype=np.float32)
    px = np.concatenate([np.asarray(tac["px"], dtype=np.float32), pad], axis=1)
    py = np.concatenate([np.asarray(tac["py"], dtype=np.float32), pad], axis=1)
    ps_in = tac.get("ps")
    ps = (np.concatenate([np.asarray(ps_in, dtype=np.float32), pad], axis=1)
          if ps_in is not None else np.full_like(px, np.nan))
    bx0 = np.asarray(tac["bx"], dtype=np.float32)
    by0 = np.asarray(tac["by"], dtype=np.float32)
    bx, by = bx0.copy(), by0.copy()
    gidx = graded_index(blocks, n)
    rep = {"game": game, "k": int(k), "plan": pl}

    def rng(c):
        return np.random.RandomState(seed_of(game, k, c))

    # Event placement reads the clean ball (bx0) and the clean players; hiding comes after.
    crowd = None
    if pl["live"] is not None:
        crowd = crowd_frames(px[:, :n0], py[:, :n0], bx0, by0, pitch)
    if pl["ghost"] is not None:
        gx = {"ghost_s": GHOST_S_X, "offset_m": GHOST_OFFSET_M_X} if version == "x" else {}
        rep["ghost"] = add_ghosts(px, py, ps, team, bx0, by0, gidx, rng("ghost"),
                                  pl["ghost"]["per_10min"], pitch, fps, **gx)
    if pl["wrong"] is not None:
        rep["wrong"] = wrong_team(px, py, ps, team, bx0, by0, gidx, rng("wrong"),
                                  pl["wrong"]["per_10min"], pitch, fps)
    if pl["drop"] is not None:
        rep["drop"] = drop_players(px, py, rng("drop"), pl["drop"]["share"], fps)
    if pl["group"] is not None:
        rep["group"] = group_dropouts(px, py, team, bx0, by0, gidx, rng("group"),
                                      pl["group"]["per_10min"], pitch, fps)
    ps[~np.isfinite(px)] = np.nan
    if pl["reslot"] is not None:
        rep["reslot"] = reslot(px, py, ps, team, rng("reslot"), pl["reslot"]["p"])
    if version == "x":
        fp = plan_flicker(game, k) if only is None else only.get("flicker")
        rep["plan"] = dict(rep["plan"], flicker=fp, version="x")
        if fp is not None:
            rep["flicker"] = flicker(px, py, ps, team, rng("flicker"), fp["per_slot_min"], fps)
    if pl["dead"] is not None:
        rep["dead"] = hide_dead(bx, by, codes, rng("dead"), pl["dead"]["p"], fps)
    if pl["live"] is not None:
        rep["live"] = hide_live(bx, by, codes, crowd, rng("live"), pl["live"]["share"],
                                pl["live"]["p_crowd"], fps)
    if ball and ball_applies(game):
        bp = plan_ball(game, k) if only is None else only.get("ball")
        rep["plan"] = dict(rep["plan"], ball=bp)
        if bp is not None:
            bx, by, rep["ball"] = smooth_offset_ball(bx, by, blocks, pitch, bp, rng("ball"), fps)
    out = dict(tac)
    out.update(px=px, py=py, ps=ps, bx=bx, by=by, team_of_slot=team, n_slots=int(team.size))
    return out, rep

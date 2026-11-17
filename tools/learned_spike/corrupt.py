"""A label-free corruption view that makes an external match look like BAR-ATH tracking.

    python -m tools.learned_spike.corrupt stats  --game G --out STATS.json [caches]
    python -m tools.learned_spike.corrupt engine --game G --out PRED.csv [--config CFG.json] [caches]
    python -m tools.learned_spike.corrupt model  --game G --out PRED.csv (--fold DIR | --lomo DIR)
                                                 [--probs-out P.npz] [caches]

Caches: --engine-cache DIR (open references) and --sc-engine-cache DIR (SkillCorner), the
engine.data npz caches of the raw view. Component switches: --no-frag, --no-swap, --flip-p P,
--ball matched|uniform30|off, --ghost-per-10min R.

Not part of the submission. The parameters and their sources are in
docs/plan/calibration/STRESS_RATES.md. Each one comes from a label-free summary statistic of
the BAR-ATH tracking (tools/learned_spike/slot_stats.py) or from the advisor's default.

The view changes the tactical dict before anything reads it, so c22 and the model see the same
input. It reads no label. Each component draws from its own numpy RandomState, seeded with
zlib.crc32 of the match name plus a fixed offset. These are the components:

(a) Fragmentation. Each visible run of a player slot is cut into chunks. A chunk lifetime is a
    draw from the BAR-ATH tracklet lifetimes (LIFE_Q). After each chunk the player is hidden
    for an exponential gap with the mean that brings the visible share of player frames to
    BAR-ATH's (R_BAR). Each chunk then takes a new slot of its team (assign_slots()). Each team
    has POOL slots. A chunk with no free slot is dropped.
(a') Team flip. Off by default (P_FLIP = 0). With --flip-p P a chunk goes to the slot pool of
    the other team with probability P.
(b) Slot swaps. Two visible chunks within SWAP_R_M (same team, or one of each team) exchange
    the players they carry from their frame of closest approach to their ends. The same-team
    rate is the advisor's default (SWAP_SAME_S). The cross-team rate (SWAP_CROSS_S) gives the
    BAR-ATH over-count: more than 11 visible slots of a team.
(c) Ball gaps. Gaps with BAR-ATH gap lengths (BALL_GAP_Q) hide the ball until the visible share
    of the graded frames is BAR-ATH's (V_BAR). A gap starts on a collective-stop frame with
    probability SLOW_START. The mode "uniform30" is the view of stress.py instead.
(d) Ghost duplicates. Off by default (rate 0). A ghost copies a visible slot, moved by
    GHOST_OFFSET_M, into a free slot of the same team for GHOST_S seconds.

engine/ does not change. The `engine` mode replaces engine.data.load_tactical in this process
with a function that returns the corrupted view (the pattern of tools/gapmask.py), then calls
engine.run.main with an optional --config, so it runs c22 and every LEARNED_MODE config. The
`model` mode predicts with the torch models of the spike. In the `engine` mode, --hidden-out
writes the mask of the frames whose ball the view hides (hidden_ball()). The mask does not
change the view. tools/learned_spike/evaluate.py needs it for criterion 5 of the protocol.

--seed N replaces the zlib.crc32 of the match name by N in every component seed (the component
offsets stay). Without --seed the draws are those of round 1.

The competition match. LEARNED_STAGE_PROTOCOL_R2.md Section 3, veto V2, applies the ball-only
view `--ball uniform30 --no-frag --no-swap --ghost-per-10min 0` with seed 0 to BAR-ATH. Only
this case may load BAR-ATH: the `engine` mode with --allow-competition, --game BAR-ATH, --seed
and exactly that view (no team flips). The view hides ball positions only. It reads no label,
and the outputs stay where --out and --hidden-out point (evaluate.py keeps them outside the
repository). Python 3.9.
"""

import argparse
import json
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import games  # noqa: E402
from engine import data  # noqa: E402
from tools.learned_spike import slot_stats  # noqa: E402

# BAR-ATH tracklet lifetimes, quantiles at 0, 5, ..., 100 %, seconds (STRESS_RATES.md 2).
LIFE_Q = (0.32, 1.72, 2.60, 3.79, 5.70, 7.92, 9.93, 13.08, 17.82, 26.16, 35.52, 49.11, 64.48,
          88.41, 112.71, 145.32, 183.13, 226.53, 284.63, 401.36, 2937.08)
MEAN_LIFE_S = 106.8      # BAR-ATH mean tracklet lifetime (STRESS_RATES.md 2)
R_BAR = 0.925            # BAR-ATH visible share of player frames (STRESS_RATES.md 2)
POOL = (24, 22)          # BAR-ATH slots per team in the ATD XML
# Team flips are off: the cross-team swaps carry the over-count (STRESS_RATES.md 7).
P_FLIP = 0.0
SWAP_SAME_S = 20.0       # advisor default: one same-team swap per 20 s of graded play
# One cross-team swap per 1,200 s: the rate that gives the BAR-ATH over-count share 0.064 on
# the eight open references (STRESS_RATES.md 7). The advisor default of 60 s gives 0.24.
SWAP_CROSS_S = 1200.0
# A swap needs two players this close: the teleport threshold of slot_stats (12 / 25 + 0.5 m),
# so a swap adds no teleport that BAR-ATH does not show (STRESS_RATES.md 7).
SWAP_R_M = slot_stats.V_MAX / 25.0 + slot_stats.NOISE_M
SWAP_SEARCH_S = 15.0     # look this far ahead for the closest approach of such a pair
SWAP_RETRIES = 20        # attempts per swap before the tool gives up
# BAR-ATH ball gaps on graded frames, quantiles at 0, 5, ..., 100 %, seconds.
BALL_GAP_Q = (0.04, 0.08, 0.47, 5.37, 5.82, 6.09, 6.79, 7.50, 8.59, 9.56, 10.72, 11.56, 12.57,
              16.14, 19.44, 21.06, 24.13, 29.06, 33.81, 50.80, 221.48)
V_BAR = 0.593            # BAR-ATH ball visible share of graded frames
SLOW_START = 0.85        # BAR-ATH share of missing-ball frames that are collective stops
UNIFORM30 = (0.30, 2.0, 30.0)   # stress.py: hide share, log-uniform gap range in seconds
GHOST_PER_10MIN = 0.0    # BAR-ATH ghosts are rarer than on the open externals: off
GHOST_S = (1.0, 3.0)
GHOST_OFFSET_M = 0.3
MAX_DRAWS = 100000
FORBIDDEN = {games.GAME, "DFL-J03WQQ", "DFL-J03WR9"}
# Seed offsets of the components, so that one component never changes the draws of another.
SEED_FRAG, SEED_FLIP, SEED_SWAP, SEED_BALL, SEED_GHOST, SEED_SWAP_CROSS = 11, 13, 17, 19, 23, 29


def seed_of(game, offset, base=None):
    """The seed of one component on one match: crc32 of the name, or `base`, plus `offset`."""
    root = zlib.crc32(game.encode("utf-8")) if base is None else int(base)
    return (root + offset) & 0x7FFFFFFF


# The one view that may run on the competition match (R2 Section 3, V2): ball gaps only.
V2_VIEW = {"frag": False, "flip_p": 0.0, "swap": False, "ball": "uniform30",
           "ghost_per_10min": 0.0}


def is_v2_view(kwargs):
    """True if corrupt() keyword arguments are exactly the V2 view (V2_VIEW)."""
    return all(kwargs.get(k) == v for k, v in V2_VIEW.items())


def sample_table(rng, table):
    """One draw from a quantile table at equal steps, log-linear between the points."""
    u = rng.uniform() * (len(table) - 1)
    i = min(int(u), len(table) - 2)
    f = u - i
    return float(np.exp(np.log(table[i]) * (1.0 - f) + np.log(table[i + 1]) * f))


def graded_mask(blocks, n):
    """bool (n,): the frames of the graded blocks [(half, first, last inclusive)]."""
    mask = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    return mask


class Chunk(object):
    """One tracklet of the corrupted view: a team and contiguous segments (column, a, b).

    Each segment carries the original column `column` on frames a to b - 1. Consecutive
    segments touch (b of one is a of the next). Two segments differ in column after a swap.
    """

    def __init__(self, team, segs):
        self.team = int(team)
        self.segs = list(segs)

    @property
    def start(self):
        return self.segs[0][1]

    @property
    def stop(self):
        return self.segs[-1][2]

    def column_at(self, t):
        for column, a, b in self.segs:
            if a <= t < b:
                return column
        return -1

    def split(self, t):
        """(segments before t, segments from t)."""
        before, after = [], []
        for column, a, b in self.segs:
            if b <= t:
                before.append((column, a, b))
            elif a >= t:
                after.append((column, a, b))
            else:
                before.append((column, a, t))
                after.append((column, t, b))
        return before, after


def visible_ratio(vis, team, graded):
    """Mean visible slots per team on graded frames, divided by 11 and capped at 1."""
    means = [vis[graded][:, team == t].sum(axis=1).mean() for t in (0, 1)]
    return min(1.0, float(np.mean(means)) / 11.0)


def fragment(vis, team, fps, rng, mean_off_s, enabled=True):
    """[Chunk]: each visible run of each column cut with BAR-ATH lifetimes and gaps."""
    chunks = []
    n_frames, n_cols = vis.shape
    for column in range(n_cols):
        starts, stops = slot_stats.runs(vis[:, column])
        first = True
        for s, e in zip(starts, stops):
            if not enabled:
                chunks.append(Chunk(team[column], [(column, int(s), int(e))]))
                continue
            pos = int(s)
            while pos < e:
                life = sample_table(rng, LIFE_Q)
                if first:
                    life *= rng.uniform()   # the player is somewhere inside his first tracklet
                    first = False
                end = min(int(e), pos + max(1, int(round(life * fps))))
                chunks.append(Chunk(team[column], [(column, pos, end)]))
                pos = end
                if mean_off_s > 0:
                    pos += max(1, int(round(rng.exponential(mean_off_s) * fps)))
    return chunks


def flip_teams(chunks, rng, probability):
    """Put each chunk in the other team's pool with `probability`. Returns the flip count."""
    flips = 0
    for chunk in chunks:
        if rng.uniform() < probability:
            chunk.team = 1 - chunk.team
            flips += 1
    return flips


def _find_pair(chunks, px, py, t0, n_frames, cross, window):
    """(t, i, j) of the first frame in [t0, t0 + window) with a chunk pair within SWAP_R_M."""
    stop = min(n_frames, t0 + window)
    alive = [k for k, c in enumerate(chunks) if c.start < stop and c.stop > t0 + 1]
    if len(alive) < 2:
        return None
    frames = np.arange(t0, stop)
    xs = np.full((frames.size, len(alive)), np.nan)
    ys = np.full((frames.size, len(alive)), np.nan)
    teams = np.array([chunks[k].team for k in alive])
    for j, k in enumerate(alive):
        for column, a, b in chunks[k].segs:
            # Strictly inside the chunk: a swap needs a part before and a part after.
            lo, hi = max(a, chunks[k].start + 1, t0), min(b, stop)
            if lo < hi:
                xs[lo - t0:hi - t0, j] = px[lo:hi, column]
                ys[lo - t0:hi - t0, j] = py[lo:hi, column]
    same = teams[:, None] == teams[None, :]
    allowed = ~same if cross else same
    allowed &= np.triu(np.ones_like(allowed), k=1).astype(bool)
    ii, jj = np.nonzero(allowed)
    if ii.size == 0:
        return None
    with np.errstate(invalid="ignore"):
        dist = np.hypot(xs[:, ii] - xs[:, jj], ys[:, ii] - ys[:, jj])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    best = dist.argmin(axis=1)
    closest = dist[np.arange(frames.size), best]
    if not (closest <= SWAP_R_M).any():
        return None
    row = int(np.argmin(closest))   # the frame of closest approach in the window
    pair = int(best[row])
    return int(frames[row]), alive[ii[pair]], alive[jj[pair]]


def swap_slots(chunks, px, py, graded, fps, game, enabled=True, periods=None, seed=None):
    """Apply same-team and cross-team identity swaps. Returns (same done, cross done).

    For each kind the number of swaps is a Poisson draw with mean graded seconds / period.
    An attempt starts at a uniform graded frame and swaps the closest pair of the kind at its
    frame of closest approach within SWAP_SEARCH_S, if that pair is within SWAP_R_M. A failed
    attempt draws a new frame, up to SWAP_RETRIES attempts per swap, so the done count meets
    the drawn count. Same-team swaps come first. Each kind draws from its own RandomState.
    `periods` is (same-team, cross-team) seconds per swap, SWAP_SAME_S and SWAP_CROSS_S when
    None. A period of 0 turns that kind off.
    """
    done = [0, 0]
    if not enabled:
        return tuple(done)
    if periods is None:
        periods = (SWAP_SAME_S, SWAP_CROSS_S)
    idx = np.flatnonzero(graded)
    seconds = idx.size / float(fps)
    window = int(round(SWAP_SEARCH_S * fps))
    for cross, period, offset in ((0, periods[0], SEED_SWAP), (1, periods[1], SEED_SWAP_CROSS)):
        rng = np.random.RandomState(seed_of(game, offset, seed))
        count = int(rng.poisson(seconds / period)) if period > 0 else 0
        for _ in range(SWAP_RETRIES * count):
            if done[cross] >= count:
                break
            t0 = int(idx[rng.randint(idx.size)])
            found = _find_pair(chunks, px, py, t0, px.shape[0], bool(cross), window)
            if found is None:
                continue
            t, i, j = found
            before_i, after_i = chunks[i].split(t)
            before_j, after_j = chunks[j].split(t)
            chunks[i].segs = before_i + after_j
            chunks[j].segs = before_j + after_i
            done[cross] += 1
    return tuple(done)


def assign_slots(chunks, pool=POOL):
    """(slot of each chunk or -1, team_of_slot): least recently freed free slot of the team.

    A slot is free for a chunk when its last chunk stopped at least one frame before the new
    chunk starts. Ties go to the lower slot index.
    """
    team_of_slot = np.concatenate([np.full(pool[0], 0), np.full(pool[1], 1)]).astype(np.int8)
    last_stop = np.full(team_of_slot.size, -1, dtype=np.int64)
    order = sorted(range(len(chunks)),
                   key=lambda k: (chunks[k].start, chunks[k].stop, chunks[k].segs[0][0]))
    slot_of = np.full(len(chunks), -1, dtype=np.int64)
    for k in order:
        chunk = chunks[k]
        cols = np.flatnonzero((team_of_slot == chunk.team) & (last_stop < chunk.start))
        if cols.size == 0:
            continue
        pick = int(cols[np.argmin(last_stop[cols])])  # argmin keeps the lowest index on ties
        slot_of[k] = pick
        last_stop[pick] = chunk.stop
    return slot_of, team_of_slot


def add_ghosts(px, py, ps, team_of_slot, graded, fps, rng, per_10min, pitch):
    """Copy visible slots into free slots of the same team, moved by GHOST_OFFSET_M."""
    if per_10min <= 0:
        return 0
    idx = np.flatnonzero(graded)
    count = rng.poisson(idx.size / float(fps) / 600.0 * per_10min)
    done = 0
    for t in np.sort(idx[rng.randint(0, idx.size, size=count)]):
        seen = np.flatnonzero(np.isfinite(px[t]))
        if seen.size == 0:
            continue
        src = int(seen[rng.randint(seen.size)])
        length = int(round(rng.uniform(*GHOST_S) * fps))
        end = min(px.shape[0], t + length)
        rows = slice(t, end)
        free = np.flatnonzero((team_of_slot == team_of_slot[src])
                              & ~np.isfinite(px[max(0, t - 1):min(px.shape[0], end + 1)]).any(axis=0))
        if free.size == 0:
            continue
        dst = int(free[0])
        angle = rng.uniform(0, 2 * np.pi)
        px[rows, dst] = px[rows, src] + GHOST_OFFSET_M * np.cos(angle) / pitch[0]
        py[rows, dst] = py[rows, src] + GHOST_OFFSET_M * np.sin(angle) / pitch[1]
        ps[rows, dst] = ps[rows, src]
        done += 1
    return done


def ball_hide_mask(bx, slow, graded, fps, rng, mode):
    """bool (N,): the ball frames that the view hides."""
    n = len(bx)
    hide = np.zeros(n, dtype=bool)
    visible = graded & np.isfinite(bx)
    if mode == "off" or not visible.any():
        return hide
    if mode == "uniform30":
        share, low, high = UNIFORM30
    else:
        share = max(0.0, 1.0 - V_BAR / (visible.sum() / float(graded.sum())))
    target = int(share * visible.sum())
    pools = (np.flatnonzero(graded & slow), np.flatnonzero(graded & ~slow))
    everywhere = np.flatnonzero(graded)
    hidden = 0
    for _ in range(MAX_DRAWS):
        if hidden >= target:
            break
        if mode == "uniform30":
            length = int(np.exp(rng.uniform(np.log(low), np.log(high))) * fps)
            s = int(everywhere[rng.randint(everywhere.size)])
        else:
            length = max(1, int(round(sample_table(rng, BALL_GAP_Q) * fps)))
            pool = pools[0] if rng.uniform() < SLOW_START else pools[1]
            if pool.size == 0:
                pool = everywhere
            s = int(pool[rng.randint(pool.size)])
        e = min(n, s + length)
        seg = graded[s:e]
        hidden += int((visible[s:e] & ~hide[s:e] & seg).sum())
        hide[s:e] |= seg
    return hide


def corrupt(tac, game, frag=True, flip_p=P_FLIP, swap=True, ball="matched",
            ghost_per_10min=GHOST_PER_10MIN, blocks=None, fps=None, pitch=None, seed=None,
            allow_competition=False):
    """(corrupted tactical dict, report). The input dict is not changed. Reads no label.

    `game` seeds the draws, or `seed` replaces its crc32 (seed_of()). `blocks`, `fps` and
    `pitch` default to the games.py entry of `game`. The unit tests give them for a synthetic
    match. The competition match needs allow_competition, a seed and the V2 view.
    """
    check_game(game, allow_competition, seed, {"frag": frag, "flip_p": flip_p, "swap": swap,
                                                "ball": ball, "ghost_per_10min": ghost_per_10min})
    fps = data.fps(game) if fps is None else int(fps)
    pitch = data.pitch_m(game) if pitch is None else (float(pitch[0]), float(pitch[1]))
    blocks = data.graded_blocks(game) if blocks is None else blocks
    px0 = np.asarray(tac["px"], dtype=np.float32)
    py0 = np.asarray(tac["py"], dtype=np.float32)
    ps0 = np.asarray(tac["ps"], dtype=np.float32)
    team = np.asarray(tac["team_of_slot"])
    n = px0.shape[0]
    vis = np.isfinite(px0) & np.isfinite(py0)
    graded = graded_mask(blocks, n)
    r_ext = visible_ratio(vis, team, graded)
    r_eff = min(1.0, R_BAR / r_ext) if r_ext > 0 else 1.0
    mean_off_s = MEAN_LIFE_S * (1.0 - r_eff) / r_eff if frag else 0.0

    chunks = fragment(vis, team, fps, np.random.RandomState(seed_of(game, SEED_FRAG, seed)),
                      mean_off_s, enabled=frag)
    flips = flip_teams(chunks, np.random.RandomState(seed_of(game, SEED_FLIP, seed)),
                       flip_p)
    xm = px0.astype(np.float64) * pitch[0]
    ym = py0.astype(np.float64) * pitch[1]
    same, cross = swap_slots(chunks, xm, ym, graded, fps, game, enabled=swap, seed=seed)
    slot_of, team_of_slot = assign_slots(chunks)

    n_slots = team_of_slot.size
    px = np.full((n, n_slots), np.nan, dtype=np.float32)
    py = np.full((n, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n, n_slots), np.nan, dtype=np.float32)
    dropped_frames = 0
    for chunk, slot in zip(chunks, slot_of):
        if slot < 0:
            dropped_frames += chunk.stop - chunk.start
            continue
        for column, a, b in chunk.segs:
            px[a:b, slot] = px0[a:b, column]
            py[a:b, slot] = py0[a:b, column]
            ps[a:b, slot] = ps0[a:b, column]
    ghosts = add_ghosts(px, py, ps, team_of_slot, graded, fps,
                        np.random.RandomState(seed_of(game, SEED_GHOST, seed)), ghost_per_10min,
                        pitch)

    slow = slot_stats.slow_frames(xm, ym, vis, fps)
    hide = ball_hide_mask(tac["bx"], slow, graded, fps,
                          np.random.RandomState(seed_of(game, SEED_BALL, seed)), ball)
    out = dict(tac)
    out.update(px=px, py=py, ps=ps, team_of_slot=team_of_slot, n_slots=int(n_slots),
               bx=np.where(hide, np.nan, tac["bx"]).astype(np.float32),
               by=np.where(hide, np.nan, tac["by"]).astype(np.float32))
    report = {"game": game, "r_ext": round(r_ext, 4), "mean_off_s": round(mean_off_s, 2),
              "chunks": len(chunks), "dropped_chunks": int((slot_of < 0).sum()),
              "dropped_frames": int(dropped_frames), "flips": flips,
              "swaps_same": same, "swaps_cross": cross, "ghosts": ghosts,
              "ball_mode": ball, "ball_hidden_graded_share": round(
                  float((hide & graded & np.isfinite(tac["bx"])).sum()) / graded.sum(), 4),
              "components": {"frag": frag, "flip_p": flip_p, "swap": swap, "ball": ball,
                             "ghost_per_10min": ghost_per_10min},
              "seed": seed, "ball_seed": seed_of(game, SEED_BALL, seed)}
    return out, report


def check_game(game, allow_competition=False, seed=None, view=None):
    """Raise SystemExit for a forbidden match. BAR-ATH passes only as the V2 case (docstring)."""
    if game == games.GAME and allow_competition:
        if seed is None or view is None or not is_v2_view(view):
            raise SystemExit("{}: only the V2 view with a seed may run on the competition "
                             "match".format(game))
        return game
    if game in FORBIDDEN:
        raise SystemExit("{} is forbidden here".format(game))
    return game


# -- loading and the CLI modes ----------------------------------------------------------------

def raw_loader(original, engine_cache, sc_engine_cache):
    """A function game -> raw tactical dict of an external match, through `original`."""
    from analysis import external_sweep as xs

    def load(game):
        if game == games.GAME:
            return original(game)       # the default data folder and engine cache (V2 only)
        root = xs.data_root(game, "raw", slot_stats.REFS_ROOT, slot_stats.SAMPLE_ROOT,
                            slot_stats.MTAH_ROOT)
        cache = sc_engine_cache if game.startswith("SC-") else engine_cache
        return original(game, data_dir=root, cache_dir=cache, use_cache=cache is not None)
    return load


def view_kwargs(args):
    return {"frag": not args.no_frag, "flip_p": args.flip_p, "swap": not args.no_swap,
            "ball": args.ball, "ghost_per_10min": args.ghost_per_10min}


def corrupt_kwargs(args):
    out = view_kwargs(args)
    out.update(seed=args.seed, allow_competition=bool(getattr(args, "allow_competition", False)))
    return out


def run_stats(args):
    """Slot statistics of the raw and the corrupted view (the smoke check)."""
    load = raw_loader(data.load_tactical, args.engine_cache, args.sc_engine_cache)
    tac = load(args.game)
    bad, report = corrupt(tac, args.game, **corrupt_kwargs(args))
    result = {"report": report, "raw": slot_stats.match_stats(tac, args.game),
              "corrupted": slot_stats.match_stats(bad, args.game)}
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    print(json.dumps(report, sort_keys=True))
    return 0


def run_engine(args):
    """engine.run on the corrupted view, in this process. engine/ does not change."""
    from analysis import external_sweep as xs
    from engine import run as engine_run
    original = data.load_tactical
    load = raw_loader(original, args.engine_cache, args.sc_engine_cache)
    if args.game != games.GAME:
        games.DATA_DIR = xs.data_root(args.game, "raw", slot_stats.REFS_ROOT,
                                      slot_stats.SAMPLE_ROOT, slot_stats.MTAH_ROOT)
    memo = {}
    hidden = {}

    def corrupted(game=None, *unused, **kwunused):
        name = data.resolve_game(game)
        if name not in memo:
            raw = load(name)
            memo[name] = corrupt(raw, name, **corrupt_kwargs(args))
            hidden[name] = hidden_ball(raw, memo[name][0])
            print(json.dumps(memo[name][1], sort_keys=True), file=sys.stderr)
        return dict(memo[name][0])

    data.load_tactical = corrupted
    argv = ["--game", args.game, "--out", args.out]
    if args.config:
        argv += ["--config", args.config]
    code = engine_run.main(argv)
    if args.hidden_out:
        if args.game not in hidden:
            raise SystemExit("{}: the engine did not load the match".format(args.game))
        np.savez_compressed(args.hidden_out, hidden=hidden[args.game])
    return code


def hidden_ball(raw, view):
    """bool (N,): the frames where the raw view has a ball and the corrupted view has none.

    The view hides the ball on graded frames only (ball_hide_mask()), so the mask is graded.
    """
    return np.isfinite(np.asarray(raw["bx"], dtype=np.float64)) & ~np.isfinite(
        np.asarray(view["bx"], dtype=np.float64))


def run_model(args):
    """The spike's torch models on the corrupted view (fallback until the engine has them)."""
    import torch
    from tools.learned_spike import features, model
    from tools.learned_spike.lomo import write_csv
    torch.set_num_threads(args.threads)
    if args.fold:
        folds = [args.fold]
    else:
        own = os.path.join(args.lomo, args.game)
        folds = [own] if os.path.isfile(os.path.join(own, "net.pt")) else sorted(
            os.path.join(args.lomo, g) for g in os.listdir(args.lomo)
            if os.path.isfile(os.path.join(args.lomo, g, "net.pt")))
    nets, costs = [], []
    for fold in folds:
        with open(os.path.join(fold, "fold.json")) as handle:
            rec = json.load(handle)
        costs.append((rec["cost_ha"], rec["cost_ad"]))
        net = model.Net()
        net.load_state_dict(torch.load(os.path.join(fold, "net.pt")))
        nets.append(net)
    cha = float(np.median([c[0] for c in costs]))
    cad = float(np.median([c[1] for c in costs]))
    load = raw_loader(data.load_tactical, args.engine_cache, args.sc_engine_cache)
    tac, report = corrupt(load(args.game), args.game, **corrupt_kwargs(args))
    f5 = features.downsample(features.frame_features(tac, args.game))
    p5 = np.mean([model.predict_symmetric(net, f5) for net in nets], axis=0)
    probs = model.upsample(p5, tac["n_frames"])
    write_csv(args.out, model.decode(probs, data.graded_blocks(args.game), cha, cad))
    if args.probs_out:
        np.savez(args.probs_out, p5=p5.astype(np.float32))
    report.update(folds=len(nets), cost_ha=cha, cost_ad=cad)
    print(json.dumps(report, sort_keys=True), file=sys.stderr)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("stats", "engine", "model"))
    parser.add_argument("--game", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--engine-cache")
    parser.add_argument("--sc-engine-cache")
    parser.add_argument("--config", help="engine config JSON (engine mode)")
    parser.add_argument("--hidden-out", help="npz of the hidden-ball mask `hidden` (engine mode)")
    parser.add_argument("--fold", help="one fold directory with net.pt and fold.json (model mode)")
    parser.add_argument("--lomo", help="a LOMO directory: the match's own fold, else all folds")
    parser.add_argument("--probs-out", help="npz of the 5 fps probabilities (model mode)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-frag", action="store_true")
    parser.add_argument("--flip-p", type=float, default=P_FLIP)
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument("--ball", choices=("matched", "uniform30", "off"), default="matched")
    parser.add_argument("--ghost-per-10min", type=float, default=GHOST_PER_10MIN)
    parser.add_argument("--seed", type=int, help="base seed in place of the crc32 of the name")
    parser.add_argument("--allow-competition", action="store_true",
                        help="engine mode, BAR-ATH, the V2 view and --seed only (R2 Section 3)")
    args = parser.parse_args(argv)
    if args.allow_competition and args.mode != "engine":
        raise SystemExit("--allow-competition works in the engine mode only")
    check_game(args.game, args.allow_competition, args.seed, view_kwargs(args))
    if args.mode == "stats":
        return run_stats(args)
    if args.mode == "engine":
        return run_engine(args)
    if not (args.fold or args.lomo):
        raise SystemExit("model mode needs --fold or --lomo")
    return run_model(args)


if __name__ == "__main__":
    sys.exit(main())

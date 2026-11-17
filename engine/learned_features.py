"""Identity-free features of the learned stage at 5 fps (LEARNED_STAGE_PROTOCOL.md Section 2).

The module turns the tactical ATD of one match (any games.py entry) into a feature matrix at
5 fps. It reads the tracking only. It never reads a label. engine.learned feeds the matrix to
the network. The code is a copy of tools/learned_spike/features.py, the module that built the
training features. tests/test_learned.py checks that the two give the same bytes, and
tools/learned_spike/feature_md5.py compares them on the external matches.

Conventions:

- The pitch turns per half so that the home team attacks +x. The home team attacks +x in a
  half when the median x of its visible players over the half is below the centre.
- Coordinates are centred and scaled to [-1, 1] on both axes.
- Each feature has a kind. The kind says how the feature changes under the two symmetries.
  A swap exchanges home and away and turns the pitch 180 degrees, so the new home team attacks
  +x. A mirror sends y to -y. So a symmetric copy is a column permutation and a sign change.
  See transform().
- The module computes the features at 25 fps and averages them over bins of BIN frames.

Feature versions (VERSIONS): "stage50-v2" (the default, 50 columns, the round 2 to 4 weights),
"stage-v3" (55 columns: track cleaning, players without a defined speed left out of the
speed columns, median, inter-quartile, second-deepest, walk and none columns) and
"stage50-v2s" (stage50-v2 with its team-swap faults removed, 50 columns) and "stage50-v2t"
(stage50-v2s with an attack direction that compares the two teams, 50 columns). The weights file
records its version and engine.learned computes that version.

The constants below define the features that the committed weights saw in training. They are
part of the trained artefact, not thresholds to tune. A change to one of them needs new weights
and a new feature version. engine.learned refuses weights of an unknown version.

Python 3.9, numpy only.
"""

import numpy as np

from engine import data, tracks
from engine.config import Config

# The version string of this feature definition. tools/learned_spike/export.py writes it into
# the weights file, and engine.learned.load_weights() checks it.
FEATURE_VERSION = "stage50-v2"

BIN = 5                # 25 fps -> 5 fps
FPS = 25.0
STILL_MPS = 1.0        # a player under this speed counts as near-stationary
RADII = (1.5, 3.0, 5.0)
CLIP_D = 30.0          # distances are clipped at 30 m
CLIP_V = 12.0          # player speeds, m/s
CLIP_BV = 35.0         # ball speed, m/s
OUT_MARGIN = 0.5       # metres beyond a line before the ball counts as out

# Round 5 feature versions. "stage50-v2" stays the default: frame_features() without a version
# computes exactly the bytes of the committed weights. "stage-v3" cleans the player tracks first
# (clean_tracks()), leaves players without a defined speed out of the speed columns, and
# replaces or adds columns (GLOBAL_V3, TEAM_V3). The width of the matrix differs per version, so
# transform() and engine.learned can tell the version from the width (version_of_width()).
VERSION_V3 = "stage-v3"
# Round 6: "stage50-v2s" is "stage50-v2" (same 50 columns, same order, same code path, no track
# cleaning) with the team-swap faults removed (_symmetric_fallbacks): a team with no visible
# player has its mean position cx, cy at the pitch centre (0) instead of -1, and the attack
# direction of a block whose home team has no visible player comes from the away team. It has
# the width of stage50-v2, so the width no longer gives the version: the weights file and the
# cache marker are authoritative (a file without a version is stage50-v2; version_of_width(50)
# answers stage50-v2).
VERSION_V2S = "stage50-v2s"
# Round 6: "stage50-v2t" is "stage50-v2s" with one change, the attack direction of a graded
# block (home_attacks_positive(compare_teams=True)): home attacks +x when the median x of the
# visible home players over the block is below that of the away players. The v2s rule (home
# median against the centre) is not team-swap equivariant when both medians lie on the same
# side of the centre, and it was wrong on 9 of the 56 external blocks; the new rule was right
# on all of them. Same width as stage50-v2: only a recorded version names it.
VERSION_V2T = "stage50-v2t"
VERSIONS = (FEATURE_VERSION, VERSION_V3, VERSION_V2S, VERSION_V2T)
BRIDGE_FRAMES = 25     # v3: a gap between two tracklets of up to this many frames is bridged
BRIDGE_M = 1.5         # v3: ... when the start is this close to the predicted position (m)
BRIDGE_VEL_FRAMES = 5  # v3: the end velocity is the mean over this many last frames
DUP_M = 0.5            # v3: two visible slots of one team closer than this ...
DUP_FRAMES = 25        # v3: ... for at least this many consecutive frames are one player
WALK_MPS = 2.0         # v3: a player under this speed counts as walking
SPREAD_SCALE = 20.0    # v3: the inter-quartile ranges are divided by this, as sx and sy in v2
# (name, kind). kind: "inv" no change; "x" negated by swap; "y" negated by swap and mirror;
# "neg" negated by swap; "H:<name>" / "A:<name>" a team pair, exchanged by swap; a pair can also
# carry an axis, "Hx"/"Ax" (exchanged and negated by swap) or "Hy"/"Ay" (exchanged, negated by
# swap and by mirror).
GLOBAL = [
    ("ball_vis", "inv"), ("ball_raw_vis", "inv"),
    ("bx", "x"), ("by", "y"), ("bspeed", "inv"), ("bvx", "x"), ("bvy", "y"),
    ("t_since", "inv"), ("t_until", "inv"), ("ball_out", "inv"),
    ("goal_dist", "inv"), ("touch_dist", "inv"),
    ("dmin", "inv"), ("d_second", "inv"), ("near_diff", "neg"),
    ("n_tracks", "inv"), ("meanv_all", "inv"), ("still_all", "inv"),
]
TEAM = [
    ("d", ""), ("d2", ""), ("n15", ""), ("n30", ""), ("n50", ""),
    ("vnear", ""), ("rel", ""), ("approach", ""),
    ("cx", "x"), ("cy", "y"), ("sx", ""), ("sy", ""),
    ("nvis", ""), ("meanv", ""), ("still", ""), ("lastx", "x"),
]
# v3: median position (medx, medy) for the mean (cx, cy), inter-quartile range (iqrx, iqry) for
# the standard deviation (sx, sy), the second-deepest player (deep2x) for the deepest (lastx),
# and two new columns: the share of walking players (walk) and no visible player (none).
GLOBAL_V3 = GLOBAL + [("walk_all", "inv")]
TEAM_V3 = [
    ("d", ""), ("d2", ""), ("n15", ""), ("n30", ""), ("n50", ""),
    ("vnear", ""), ("rel", ""), ("approach", ""),
    ("medx", "x"), ("medy", "y"), ("iqrx", ""), ("iqry", ""),
    ("nvis", ""), ("meanv", ""), ("still", ""), ("deep2x", "x"),
    ("walk", ""), ("none", ""),
]
def check_version(version):
    """`version` itself. Raise ValueError for a version outside VERSIONS."""
    if version not in VERSIONS:
        raise ValueError("feature version must be one of {}, not {!r}".format(VERSIONS, version))
    return version


def feature_spec(version=FEATURE_VERSION):
    """[(name, kind)] in column order for a feature version."""
    check_version(version)
    v3 = version == VERSION_V3
    spec = list(GLOBAL_V3 if v3 else GLOBAL)
    for team in ("H", "A"):
        for name, axis in (TEAM_V3 if v3 else TEAM):
            spec.append((team + "_" + name, team + axis))
    return spec


def feature_names(version=FEATURE_VERSION):
    """The column names of a feature version, in order."""
    return [name for name, _ in feature_spec(version)]


def n_features(version=FEATURE_VERSION):
    """The number of columns of a feature version."""
    return len(feature_spec(version))


def version_of_width(width):
    """The feature version whose matrix has `width` columns. Raise ValueError if none has.

    50 columns answer "stage50-v2": "stage50-v2s" and "stage50-v2t" have the same width and
    the same symmetry tables, so only a recorded version (weights file, cache marker) can name it.
    """
    for version in VERSIONS:
        if n_features(version) == int(width):
            return version
    raise ValueError("no feature version has {} columns ({})".format(
        width, {v: n_features(v) for v in VERSIONS}))


def column_indices(names, version):
    """int array: the column of each name of `names` in the matrix of `version`."""
    have = feature_names(version)
    missing = [n for n in names if n not in have]
    if missing:
        raise ValueError("{}: no columns {}".format(version, missing[:5]))
    return np.array([have.index(n) for n in names], dtype=np.int64)


def select_version(feats, have, want):
    """The columns of version `want` taken by name from a (..., NF) matrix of version `have`."""
    return np.asarray(feats)[..., column_indices(feature_names(want), have)]


def _symmetry_tables(spec=None):
    spec = feature_spec() if spec is None else spec
    names = [n for n, _ in spec]
    nf = len(spec)
    perm = np.arange(nf)
    swap_sign = np.ones(nf, dtype=np.float32)
    mirror_sign = np.ones(nf, dtype=np.float32)
    for i, (name, kind) in enumerate(spec):
        axis = kind
        if kind[:1] in ("H", "A"):
            other = ("A" if kind[0] == "H" else "H") + name[1:]
            perm[i] = names.index(other)
            axis = kind[1:]
        if axis in ("x", "neg"):
            swap_sign[i] = -1.0
        if axis == "y":
            swap_sign[i] = -1.0
            mirror_sign[i] = -1.0
    return perm, swap_sign, mirror_sign


_TABLES = {}


def symmetry_tables(version=FEATURE_VERSION):
    """(PERM, SWAP_SIGN, MIRROR_SIGN) of a feature version. See transform()."""
    if version not in _TABLES:
        _TABLES[version] = _symmetry_tables(feature_spec(version))
    return _TABLES[version]


SPEC = feature_spec()
NF = len(SPEC)
PERM, SWAP_SIGN, MIRROR_SIGN = symmetry_tables(FEATURE_VERSION)


def transform(feats, swap=False, mirror=False, version=None):
    """A symmetric copy of a (..., NF) feature array. Labels: swap exchanges H and A.

    `version` None takes the version from the width of the last axis (version_of_width()).
    """
    if version is None:
        version = version_of_width(np.shape(feats)[-1])
    perm, swap_sign, mirror_sign = symmetry_tables(version)
    out = feats
    if swap:
        out = out[..., perm] * swap_sign
    if mirror:
        out = out * mirror_sign
    return out


# -- geometry --------------------------------------------------------------------------------

def home_attacks_positive(tac, blocks, length, away_fallback=False, compare_teams=False):
    """{half: True if home attacks +x in that half}, from the median x of home players.

    away_fallback (stage50-v2s): a block without any visible home player takes the direction
    from the away team (home attacks +x when the away median x is at or above the centre), so
    that a team swap turns the pitch there too. Without it such a block reads median 0.5.

    compare_teams (stage50-v2t): when both teams have a visible player in the block, home
    attacks +x if the home median x is below the away median x; equal medians (within 1e-9)
    keep +x. A block with one team visible falls back to the v2s rule for that team; a block
    with neither keeps +x. A team swap exchanges the medians, so it turns the answer (ties
    aside); y plays no part.
    """
    team = np.asarray(tac["team_of_slot"])
    out = {}
    for half, first, last in blocks:
        px = np.asarray(tac["px"][first:last + 1], dtype=np.float64)[:, team == 0]
        if compare_teams:
            pa = np.asarray(tac["px"][first:last + 1], dtype=np.float64)[:, team == 1]
            hv, av = np.isfinite(px).any(), np.isfinite(pa).any()
            if hv and av:
                mh, ma = np.nanmedian(px), np.nanmedian(pa)
                out[int(half)] = True if abs(mh - ma) <= 1e-9 else bool(mh < ma)
            elif hv:
                out[int(half)] = bool(np.nanmedian(px) < 0.5)
            elif av:
                out[int(half)] = bool(np.nanmedian(pa) >= 0.5)
            else:
                out[int(half)] = True
            continue
        if away_fallback and not np.isfinite(px).any():
            pa = np.asarray(tac["px"][first:last + 1], dtype=np.float64)[:, team == 1]
            if np.isfinite(pa).any():
                out[int(half)] = bool(np.nanmedian(pa) >= 0.5)
                continue
        med = np.nanmedian(px) if np.isfinite(px).any() else 0.5
        out[int(half)] = bool(med < 0.5)
    return out


def oriented_arrays(tac, game, away_fallback=False, compare_teams=False):
    """(px, py, bx, by) normalised, turned so the home team attacks +x in every half."""
    blocks = data.graded_blocks(game)
    direction = home_attacks_positive(tac, blocks, data.pitch_m(game)[0], away_fallback,
                                      compare_teams)
    n = tac["px"].shape[0]
    flip = np.zeros(n, dtype=bool)
    # Each frame takes the direction of the block it is in, or of the nearest block before it.
    edges = [(first, last, direction[half]) for half, first, last in blocks]
    for k, (first, last, positive) in enumerate(edges):
        start = 0 if k == 0 else first
        stop = n if k == len(edges) - 1 else edges[k + 1][0]
        flip[start:stop] = not positive
    px = np.where(flip[:, None], 1.0 - tac["px"], tac["px"]).astype(np.float64)
    py = np.where(flip[:, None], 1.0 - tac["py"], tac["py"]).astype(np.float64)
    bx = np.where(flip, 1.0 - tac["bx"], tac["bx"]).astype(np.float64)
    by = np.where(flip, 1.0 - tac["by"], tac["by"]).astype(np.float64)
    return px, py, bx, by


def _fill_nearest(values, visible):
    """Forward fill, then back fill before the first visible frame."""
    n = len(values)
    idx = np.where(visible, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    first = np.flatnonzero(visible)
    if first.size == 0:
        return np.zeros(n)
    idx = np.where(idx < 0, first[0], idx)
    return values[idx]


def _time_since(visible):
    """Frames since the last visible frame (0 on a visible frame), large before the first."""
    n = len(visible)
    idx = np.where(visible, np.arange(n), -10 ** 7)
    last = np.maximum.accumulate(idx)
    return np.arange(n) - last


# -- track cleaning (v3 only) ---------------------------------------------------------------

def bridge_gaps(px_m, py_m, team_of_slot, blocks=None):
    """(px_m, py_m, filled): float64 copies with short gaps between tracklets bridged.

    Step 1 of clean_tracks(), in metres. A tracklet end is a slot visible at frame t0 and not
    at t0 + 1; a tracklet start is a slot visible at t1 and not at t1 - 1. An end and a later
    start of the same team are candidates when 0 < t1 - t0 <= BRIDGE_FRAMES, both frames lie in
    one block of `blocks` (graded blocks [(half, first, last)]; None: the whole match is one
    block; a frame outside every block links nothing), the ending slot stays invisible from
    t0 + 1 to t1 - 1, and the start position lies within BRIDGE_M of the constant-velocity
    prediction from the end: end position + end velocity x (t1 - t0), the end velocity being
    the mean velocity over the last BRIDGE_VEL_FRAMES frames of the tracklet (zero if it is
    shorter). A candidate pair is linked when each is the other's nearest candidate (smallest
    prediction error; ties to the lower slot index, then the earlier frame). For a linked pair
    the frames t0 + 1 .. t1 - 1 of the ending slot get the linear interpolation between the
    end and the start position. Nothing moves between slots. `filled` (bool (N, S)) marks the
    filled entries.
    """
    px = np.array(px_m, dtype=np.float64)
    py = np.array(py_m, dtype=np.float64)
    team = np.asarray(team_of_slot)
    vis = np.isfinite(px) & np.isfinite(py)
    n, n_slots = vis.shape
    filled = np.zeros(vis.shape, dtype=bool)
    if n < 3:
        return px, py, filled
    block_id = np.full(n, -1, dtype=np.int64)
    if blocks is None:
        block_id[:] = 0
    else:
        for k, (_, first, last) in enumerate(blocks):
            block_id[max(int(first), 0):int(last) + 1] = k
    e_t, e_s = np.nonzero(vis[:-1] & ~vis[1:])            # row-major: by frame, then slot
    s_t, s_s = np.nonzero(~vis[:-1] & vis[1:])
    s_t = s_t + 1
    if e_t.size == 0 or s_t.size == 0:
        return px, py, filled
    # End velocity (m per frame) over the last BRIDGE_VEL_FRAMES frames of the tracklet.
    k = BRIDGE_VEL_FRAMES
    csum = np.concatenate([np.zeros((1, n_slots), dtype=np.int64),
                           np.cumsum(vis, axis=0, dtype=np.int64)])
    lo = np.maximum(e_t - k + 1, 0)
    full = (csum[e_t + 1, e_s] - csum[lo, e_s] == k) & (e_t >= k - 1)
    back = np.where(full, e_t - (k - 1), e_t)
    vx = np.where(full, (px[e_t, e_s] - px[back, e_s]) / float(k - 1), 0.0)
    vy = np.where(full, (py[e_t, e_s] - py[back, e_s]) / float(k - 1), 0.0)
    # The next start of the ending slot itself (it must stay invisible until the start).
    order_s = np.lexsort((s_t, s_s))                        # by slot, then frame
    ss_slot, ss_t = s_s[order_s], s_t[order_s]
    own_next = np.full(e_t.size, np.iinfo(np.int64).max)
    for slot in np.unique(e_s):
        sel = e_s == slot
        frames = ss_t[ss_slot == slot]
        if frames.size == 0:
            continue
        pos = np.searchsorted(frames, e_t[sel], side="right")
        own_next[sel] = np.where(pos < frames.size, frames[np.minimum(pos, frames.size - 1)],
                                 np.iinfo(np.int64).max)
    pe, ps, perr = [], [], []
    for t in np.unique(team):
        ends = np.flatnonzero(team[e_s] == t)
        starts = np.flatnonzero(team[s_s] == t)
        if ends.size == 0 or starts.size == 0:
            continue
        starts = starts[np.argsort(s_t[starts], kind="stable")]
        st = s_t[starts]
        for i in ends:
            t0 = e_t[i]
            a = np.searchsorted(st, t0 + 1, side="left")
            b = np.searchsorted(st, t0 + BRIDGE_FRAMES, side="right")
            if a >= b or block_id[t0] < 0:
                continue
            j = starts[a:b]
            t1 = s_t[j]
            gap = (t1 - t0).astype(np.float64)
            err = np.hypot(px[t1, s_s[j]] - (px[t0, e_s[i]] + vx[i] * gap),
                           py[t1, s_s[j]] - (py[t0, e_s[i]] + vy[i] * gap))
            ok = (block_id[t1] == block_id[t0]) & (own_next[i] >= t1) & (err <= BRIDGE_M)
            for jj, ee in zip(j[ok], err[ok]):
                pe.append(i)
                ps.append(jj)
                perr.append(ee)
    if not pe:
        return px, py, filled
    pe, ps, perr = np.array(pe), np.array(ps), np.array(perr)
    # Best start of each end: smallest error, then lower start slot, then earlier start frame.
    o = np.lexsort((s_t[ps], s_s[ps], perr, pe))
    first = np.concatenate(([True], pe[o][1:] != pe[o][:-1]))
    best_start = dict(zip(pe[o][first].tolist(), ps[o][first].tolist()))
    # Best end of each start: smallest error, then lower end slot, then earlier end frame.
    o = np.lexsort((e_t[pe], e_s[pe], perr, ps))
    first = np.concatenate(([True], ps[o][1:] != ps[o][:-1]))
    best_end = dict(zip(ps[o][first].tolist(), pe[o][first].tolist()))
    for i, j in sorted(best_start.items()):
        if best_end.get(j) != i:
            continue
        t0, t1, slot = int(e_t[i]), int(s_t[j]), int(e_s[i])
        if t1 - t0 < 2:
            continue
        w = (np.arange(t0 + 1, t1) - t0) / float(t1 - t0)
        x0, y0 = px[t0, slot], py[t0, slot]
        x1, y1 = px[t1, s_s[j]], py[t1, s_s[j]]
        px[t0 + 1:t1, slot] = x0 + w * (x1 - x0)
        py[t0 + 1:t1, slot] = y0 + w * (y1 - y0)
        filled[t0 + 1:t1, slot] = True
    return px, py, filled


def clean_mask(px_m, py_m, team_of_slot):
    """bool (N, S): the slots that stay visible after step 2 of clean_tracks().

    Persistent duplicates, in metres, deterministic: for each pair of slots i < j of the same
    team, the frames where both are visible and closer than DUP_M form runs. Over every run of
    at least DUP_FRAMES consecutive frames the higher slot j is treated as not visible. A
    shorter run is left alone (two real players in a wall or a duel). The pairs are judged
    independently on the input visibility, so the result does not depend on an order.
    """
    px_m = np.asarray(px_m, dtype=np.float64)
    py_m = np.asarray(py_m, dtype=np.float64)
    team = np.asarray(team_of_slot)
    vis = np.isfinite(px_m) & np.isfinite(py_m)
    keep = vis.copy()
    for t in np.unique(team):
        cols = np.flatnonzero(team == t)
        for a in range(cols.size):
            i = cols[a]
            for j in cols[a + 1:]:
                both = vis[:, i] & vis[:, j]
                if not both.any():
                    continue
                with np.errstate(invalid="ignore"):
                    close = both & (np.hypot(px_m[:, j] - px_m[:, i],
                                             py_m[:, j] - py_m[:, i]) < DUP_M)
                starts, stops = tracks._runs(close)
                for start, stop in zip(starts, stops):
                    if stop - start >= DUP_FRAMES:
                        keep[start:stop, j] = False
    return keep


def clean_tracks(px_m, py_m, team_of_slot, blocks=None):
    """(px_m, py_m) float64 copies after the track cleaning of "stage-v3".

    1. bridge_gaps(): short gaps between tracklets of one team filled in the ending slot
       (never across a block of `blocks`; None: one block);
    2. clean_mask(): persistent duplicates; the dropped entries become NaN.
    There is no cap on the number of visible slots of a team: the columns that count players
    keep their own clipping.
    The feature version "stage-v3" cleans the tracks before anything else
    (frame_features()). The rule engine and "stage50-v2" never see this cleaning.
    """
    px_m, py_m, _ = bridge_gaps(px_m, py_m, team_of_slot, blocks)
    keep = clean_mask(px_m, py_m, team_of_slot)
    return np.where(keep, px_m, np.nan), np.where(keep, py_m, np.nan)


def _clean_tactical(tac, game):
    """A copy of a tactical dict whose player slots are cleaned (clean_tracks(), in metres,
    with the graded blocks of the match). Positions that the cleaning keeps keep their bytes."""
    length, width = data.pitch_m(game)
    px = np.asarray(tac["px"])
    py = np.asarray(tac["py"])
    team = tac["team_of_slot"]
    xm, ym, filled = bridge_gaps(px.astype(np.float64) * length, py.astype(np.float64) * width,
                                 team, data.graded_blocks(game))
    keep = clean_mask(xm, ym, team)
    nan = np.array(np.nan, dtype=px.dtype)
    px = np.where(filled, (xm / length).astype(px.dtype), px)
    py = np.where(filled, (ym / width).astype(py.dtype), py)
    return dict(tac, px=np.where(keep, px, nan), py=np.where(keep, py, nan))


def _sorted_quantile(values, count, q):
    """float64 (N,): quantile q of each row, linear interpolation (numpy's default method).

    `values` is sorted along axis 1 with the NaN at the end and `count` finite values per row.
    A row without a finite value gives NaN.
    """
    n = values.shape[0]
    rows = np.arange(n)
    count = np.asarray(count, dtype=np.int64)
    pos = q * np.maximum(count - 1, 0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(count - 1, 0))
    frac = pos - lo
    if values.shape[1] == 0:
        return np.full(n, np.nan)
    a = values[rows, np.minimum(lo, values.shape[1] - 1)]
    b = values[rows, np.minimum(hi, values.shape[1] - 1)]
    out = a + frac * (b - a)
    return np.where(count > 0, out, np.nan)


# Round 3 switch (DRAFT): a stable argsort, so that a tie between two equal finite distances
# goes to the lowest slot index on every platform. ON for round 3 (R3 §5b: byte-identical to the
# round 2 cache on all 115 cached match-views and BAR-ATH on macOS arm64).
STABLE_SORT = True


def frame_features(tac, game, ball_override=None, cfg=None, version=FEATURE_VERSION):
    """float32 (N, n_features(version)) features at 25 fps. ball_override=(bx, by) replaces the
    ATD ball (oriented, normalised coordinates).

    `cfg` gives the ball cleaning fields of tracks.clean_ball(). None means Config.default(),
    the configuration of the training features. `version` is a feature version of VERSIONS:
    "stage50-v2" (the default) computes the bytes of the round 2 to 4 weights. "stage-v3"
    first cleans the player tracks (clean_tracks(): gap bridging, persistent duplicates);
    every player velocity and feature comes after.
    """
    check_version(version)
    v3 = version == VERSION_V3
    v2t = version == VERSION_V2T
    v2s = version in (VERSION_V2S, VERSION_V2T)  # v2t keeps every v2s change
    if v3:
        tac = _clean_tactical(tac, game)
    length, width = data.pitch_m(game)
    cfg = cfg or Config.default()
    px, py, bx, by = oriented_arrays(tac, game, away_fallback=v2s, compare_teams=v2t)
    if ball_override is not None:
        bx, by = ball_override
    team = np.asarray(tac["team_of_slot"])
    n = px.shape[0]
    raw_vis = np.isfinite(bx) & np.isfinite(by)
    ball = tracks.clean_ball(bx, by, (length, width), cfg, fps=FPS)
    vis = ball["visible"]
    bxm = _fill_nearest(np.nan_to_num(ball["x"]), vis)
    bym = _fill_nearest(np.nan_to_num(ball["y"]), vis)
    bspeed = np.where(vis, np.nan_to_num(ball["speed"]), 0.0)
    bvx = np.where(vis, np.nan_to_num(ball["vx"]), 0.0)
    bvy = np.where(vis, np.nan_to_num(ball["vy"]), 0.0)
    since = _time_since(vis)
    until = _time_since(vis[::-1])[::-1]

    pxm, pym = px * length, py * width
    pvis = np.isfinite(pxm) & np.isfinite(pym)
    pvx, pvy = tracks._difference_velocity(pxm, pym, pvis, FPS)
    pspeed = np.clip(np.nan_to_num(np.hypot(pvx, pvy)), 0, CLIP_V)
    dist = np.hypot(pxm - bxm[:, None], pym - bym[:, None])
    dist = np.where(pvis, dist, np.inf)
    rows = np.arange(n)

    cols = {}
    cols["ball_vis"] = vis.astype(np.float64)
    cols["ball_raw_vis"] = raw_vis.astype(np.float64)
    cols["bx"] = bxm / (length / 2.0) - 1.0
    cols["by"] = bym / (width / 2.0) - 1.0
    cols["bspeed"] = np.clip(bspeed, 0, CLIP_BV) / 10.0
    cols["bvx"] = np.clip(bvx, -CLIP_BV, CLIP_BV) / 10.0
    cols["bvy"] = np.clip(bvy, -CLIP_BV, CLIP_BV) / 10.0
    cols["t_since"] = np.log1p(np.minimum(since, 60 * FPS) / FPS)
    cols["t_until"] = np.log1p(np.minimum(until, 60 * FPS) / FPS)
    out_x = (bxm < -OUT_MARGIN) | (bxm > length + OUT_MARGIN)
    out_y = (bym < -OUT_MARGIN) | (bym > width + OUT_MARGIN)
    cols["ball_out"] = (out_x | out_y).astype(np.float64)
    cols["goal_dist"] = np.minimum(np.hypot(bxm, bym - width / 2),
                                   np.hypot(length - bxm, bym - width / 2)) / CLIP_D
    cols["touch_dist"] = np.clip(np.minimum(bym, width - bym), -5, 34) / 10.0

    every = np.sort(dist, axis=1)
    d1 = np.minimum(every[:, 0], CLIP_D)
    d2 = np.minimum(every[:, 1], CLIP_D) if every.shape[1] > 1 else np.full(n, CLIP_D)
    cols["dmin"] = d1 / 10.0
    cols["d_second"] = d2 / 10.0
    # Capped at the physical maximum (22 players, 11 per team): more visible tracks are
    # ghosts or fragments (LEARNED_STAGE_PROTOCOL §2). Class F.
    cols["n_tracks"] = np.minimum(pvis.sum(axis=1), 22) / 22.0
    nvis_all = np.maximum(pvis.sum(axis=1), 1)
    cols["meanv_all"] = np.where(pvis, pspeed, 0).sum(axis=1) / nvis_all / 5.0
    cols["still_all"] = (pvis & (pspeed < STILL_MPS)).sum(axis=1) / nvis_all
    if v3:
        # v3: a visible player whose difference velocity is undefined (an isolated visible
        # frame, tracks._difference_velocity) is left out of the numerator and the
        # denominator of every speed column, instead of counting as speed 0.
        sok = pvis & np.isfinite(pvx) & np.isfinite(pvy)
        nsok = np.maximum(sok.sum(axis=1), 1)
        cols["meanv_all"] = np.where(sok, pspeed, 0).sum(axis=1) / nsok / 5.0
        cols["still_all"] = (sok & (pspeed < STILL_MPS)).sum(axis=1) / nsok
        cols["walk_all"] = (sok & (pspeed < WALK_MPS)).sum(axis=1) / nsok

    per = {}
    for key, t in (("H", 0), ("A", 1)):
        c = np.flatnonzero(team == t)
        sub = dist[:, c]
        order = np.argsort(sub, axis=1, kind="stable" if STABLE_SORT else None)
        arg1 = order[:, 0]
        near = sub[rows, arg1]
        sec = sub[rows, order[:, 1]] if c.size > 1 else np.full(n, np.inf)
        found = np.isfinite(near)
        slot = c[arg1]
        dT = np.minimum(np.where(found, near, CLIP_D), CLIP_D)
        per[key] = dT
        cols[key + "_d"] = dT / 10.0
        cols[key + "_d2"] = np.minimum(np.where(np.isfinite(sec), sec, CLIP_D), CLIP_D) / 10.0
        for r, name in zip(RADII, ("n15", "n30", "n50")):
            cols[key + "_" + name] = (sub <= r).sum(axis=1) / 3.0
        cols[key + "_vnear"] = np.where(found, pspeed[rows, slot], 0.0) / 5.0
        rel = np.hypot(bvx - np.nan_to_num(pvx[rows, slot]), bvy - np.nan_to_num(pvy[rows, slot]))
        cols[key + "_rel"] = np.where(found & vis, np.clip(rel, 0, CLIP_BV), 0.0) / 10.0
        ahead = np.concatenate([dT[2:], np.repeat(dT[-1:], 2)])
        behind = np.concatenate([np.repeat(dT[:1], 2), dT[:-2]])
        cols[key + "_approach"] = np.clip((behind - ahead) * FPS / 4.0, -10, 10) / 5.0
        tv = pvis[:, c]
        cnt = np.maximum(tv.sum(axis=1), 1)
        if v3:
            _team_columns_v3(cols, key, tv, pxm[:, c], pym[:, c], pspeed[:, c], length, width,
                             sok[:, c])
            continue
        xs = np.where(tv, pxm[:, c], 0.0)
        ys = np.where(tv, pym[:, c], 0.0)
        cx = xs.sum(axis=1) / cnt
        cy = ys.sum(axis=1) / cnt
        if v2s:
            # stage50-v2s: the pitch centre for a team with no visible player (v2: 0 m, which
            # normalises to -1 and breaks the team swap). The other per-team fallbacks of v2
            # are swap-symmetric already: d, d2 at CLIP_D, counts, vnear, rel, sx, sy, nvis,
            # meanv, still at 0, approach from d, lastx at the centre.
            none = tv.sum(axis=1) == 0
            cx = np.where(none, length / 2.0, cx)
            cy = np.where(none, width / 2.0, cy)
        sx = np.sqrt(np.maximum((np.where(tv, (pxm[:, c] - cx[:, None]) ** 2, 0)).sum(axis=1) / cnt, 0))
        sy = np.sqrt(np.maximum((np.where(tv, (pym[:, c] - cy[:, None]) ** 2, 0)).sum(axis=1) / cnt, 0))
        cols[key + "_cx"] = cx / (length / 2.0) - 1.0
        cols[key + "_cy"] = cy / (width / 2.0) - 1.0
        cols[key + "_sx"] = sx / 20.0
        cols[key + "_sy"] = sy / 20.0
        cols[key + "_nvis"] = np.minimum(tv.sum(axis=1), 11) / 11.0
        cols[key + "_meanv"] = np.where(tv, pspeed[:, c], 0).sum(axis=1) / cnt / 5.0
        cols[key + "_still"] = (tv & (pspeed[:, c] < STILL_MPS)).sum(axis=1) / cnt
        # The x of the team's deepest visible player (towards its own goal, which is x = 0 for H
        # and x = length for A). For A the value is taken as the most advanced x so that a swap
        # of the copy maps it onto the H value with a sign change.
        if key == "H":
            deep = np.where(tv, pxm[:, c], np.inf).min(axis=1)
        else:
            deep = np.where(tv, pxm[:, c], -np.inf).max(axis=1)
        deep = np.where(np.isfinite(deep), deep, length / 2.0)
        cols[key + "_lastx"] = deep / (length / 2.0) - 1.0
    cols["near_diff"] = np.clip(per["A"] - per["H"], -10, 10) / 5.0

    feats = np.stack([cols[name] for name, _ in feature_spec(version)], axis=1)
    return np.nan_to_num(feats).astype(np.float32)


def _team_columns_v3(cols, key, tv, x, y, speed, length, width, sok):
    """The v3 team columns of team `key` ("H" or "A") that differ from v2, into `cols`.

    tv: bool (N, S) visible slots of the team (after cleaning); x, y: their positions (m, the
    oriented pitch, home attacks +x); speed: their speeds (m/s, clipped as in v2); sok: bool
    (N, S) visible slots with a defined speed, the only ones the speed columns count.
    """
    count = tv.sum(axis=1)
    nsok = np.maximum(sok.sum(axis=1), 1)
    xs = np.sort(np.where(tv, x, np.nan), axis=1)      # NaN at the end of each row
    ys = np.sort(np.where(tv, y, np.nan), axis=1)
    # Median position, the pitch centre when no player is visible, so that a swap (which turns
    # the pitch) maps the value of H onto the value of A with a sign change in every case.
    medx = _sorted_quantile(xs, count, 0.5)
    medy = _sorted_quantile(ys, count, 0.5)
    cols[key + "_medx"] = np.where(count > 0, medx, length / 2.0) / (length / 2.0) - 1.0
    cols[key + "_medy"] = np.where(count > 0, medy, width / 2.0) / (width / 2.0) - 1.0
    iqrx = _sorted_quantile(xs, count, 0.75) - _sorted_quantile(xs, count, 0.25)
    iqry = _sorted_quantile(ys, count, 0.75) - _sorted_quantile(ys, count, 0.25)
    cols[key + "_iqrx"] = np.where(count > 0, iqrx, 0.0) / SPREAD_SCALE
    cols[key + "_iqry"] = np.where(count > 0, iqry, 0.0) / SPREAD_SCALE
    cols[key + "_nvis"] = np.minimum(count, 11) / 11.0
    cols[key + "_meanv"] = np.where(sok, speed, 0).sum(axis=1) / nsok / 5.0
    cols[key + "_still"] = (sok & (speed < STILL_MPS)).sum(axis=1) / nsok
    cols[key + "_walk"] = (sok & (speed < WALK_MPS)).sum(axis=1) / nsok
    cols[key + "_none"] = (count == 0).astype(np.float64)
    # The x of the second-deepest visible player (towards the team's own goal: x = 0 for H,
    # x = length for A), the deepest if only one is visible, the centre if none. For A the
    # value is the second-largest x, so that a swap maps it onto the H value (as lastx in v2).
    rows = np.arange(xs.shape[0])
    last = xs.shape[1] - 1
    if key == "H":
        second = np.where(count >= 2, 1, 0)
    else:
        second = np.where(count >= 2, count - 2, count - 1)
    deep = xs[rows, np.clip(second, 0, last)] if xs.shape[1] else np.full(xs.shape[0], np.nan)
    deep = np.where(count > 0, deep, length / 2.0)
    cols[key + "_deep2x"] = deep / (length / 2.0) - 1.0


def downsample(feats, bin_size=BIN):
    """Mean over bins of bin_size frames. The tail is padded with the last frame."""
    n = feats.shape[0]
    nb = -(-n // bin_size)
    pad = nb * bin_size - n
    if pad:
        feats = np.concatenate([feats, np.repeat(feats[-1:], pad, axis=0)])
    return feats.reshape(nb, bin_size, -1).mean(axis=1).astype(np.float32)


def bin_labels(codes, bin_size=BIN):
    """The label code at the centre frame of each bin (codes as possession_metrics codes)."""
    n = len(codes)
    centres = np.minimum(np.arange(-(-n // bin_size)) * bin_size + bin_size // 2, n - 1)
    return np.asarray(codes)[centres]


def matrix_md5(feats):
    """The hex md5 of a feature matrix: its shape, then its float32 bytes in C order.

    tools/learned_spike/feature_md5.py writes it per match, so two feature implementations
    can be compared byte for byte without a copy of the tracking.
    """
    import hashlib

    arr = np.ascontiguousarray(feats, dtype=np.float32)
    digest = hashlib.md5("{}x{}".format(*arr.shape).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()

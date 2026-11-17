"""Extra column groups of the per-frame member (member spec F6, round 8), numpy only.

engine.learned_pf computes these columns only when a weights file lists one of them; the member
of the shipped spec (F5T, learned_pf_v7.npz) lists none, so its features and its output do not
change. Every column is computed from the tracking only, never from a label.

Coordinates: the oriented pitch of the member's base columns (engine.learned_features.
oriented_arrays with the rule of the base version: home attacks +x), in metres. Each team's own
attacking frame is a = x for the home team and a = length - x for the away team, so a team's
columns read the same way for both teams and a team swap (H <-> A, pitch turned) exchanges the
H and A columns exactly up to float rounding; nothing depends on y except through distances and
areas, so a mirror (y -> width - y) changes none of them.

The ball: the cleaned ball of tracks.clean_ball (as the base columns), its position filled from
the nearest visible frame where it is not in view (the convention of the base columns bx, H_d,
dmin); its kinematic columns (acceleration, turn) are 0 where it is not in view (the convention
of bspeed, bvx, H_rel).

Groups (kind as engine.learned_features: "inv", or a team pair H/A exchanged by the swap):

G-a, ball against each team's block (per team T in H, A; a_ball = the ball in T's frame):
  T_gbehind, T_glevel, T_gahead   visible players of T with a < a_ball - LEVEL_M, within
                                  LEVEL_M, and a > a_ball + LEVEL_M; min(count, 11) / 11
  T_gdahead_p1, T_gdahead_n1     T_gahead(t) - T_gahead(t - 25) and T_gahead(t + 25) - T_gahead(t),
                                  the other frame clamped into the graded block
  T_gbc, T_gbdeep, T_gbfront     (a_ball - centroid), (a_ball - deepest outfield line),
                                  (a_ball - front line), metres / 10. Centroid: mean a of the
                                  visible players; deepest outfield line: the second-smallest a
                                  (the smallest is taken as the goalkeeper; the only one if one
                                  is visible); front line: the largest a. No visible player:
                                  the pitch centre for all three.
G-b, block motion (per team): T_gvc, T_gvdeep, T_gvfront: velocity along T's attack of its
  centroid, deepest outfield line and front line, (s(t2) - s(t1)) / (t2 - t1) frames x 25 / 5,
  t1 = t - 12, t2 = t + 12 clamped into the graded block (0 where t1 = t2).
G-c, block area (per team): T_garea, the area of the 1-sigma covariance ellipse of T's visible
  outfield players (all but the smallest a; at least 3 left, else 0), pi sqrt(det cov) / 1000
  (m^2). Between the blocks ("inv"): g_ovx, g_ovy, the overlap of the two teams' outfield x and
  y ranges (intersection over union, 0 if the union is empty); g_gapx = (away centroid x - home
  centroid x) / 10 on the oriented pitch; g_cdist = centroid distance / 10.
G-d, sharpness: g_bacc = |v(t + 2) - v(t - 2)| / (4 / 25 s) of the cleaned ball velocity where
  the ball is in view at t - 2, t and t + 2 (else 0), clipped at 200 m/s^2, / 50; g_bturn = the
  angle between v(t - 3) and v(t + 3) in units of pi, where the ball is in view at the three
  frames and both speeds are at least TURN_MIN_MPS (else 0); g_bacc_m04, g_bacc_m1,
  g_bturn_m04, g_bturn_m1: their maxima over +-10 and +-25 frames inside the graded block;
  T_gnacc: the acceleration (|dv| over +-2 frames of the players' difference velocity, 0 where
  undefined, clipped at 20 m/s^2, / 5) of T's player nearest to the ball (the nearest of
  learned_features: stable order, the filled ball), 0 with no visible player.
Windowed columns are 0 outside the graded blocks.

Scale "robust" (member spec F6c, robust_units()): every continuous column above (not the five
counts per team) minus its median and divided by its inter-quartile range (at least IQR_FLOOR)
over this match, on the frames where it is defined (in a graded block; and the ball in view for
the ball-position and sharpness columns; at least 3 outfield players for the area; the nearest
player's velocity defined for its acceleration; both teams visible for the between-block
columns), the H and A versions of a team column pooled. The statistics come from the match being
labelled itself (label-free), never from a stored constant. Then the missing-value convention:
the ball-position columns keep their rescaled value on every frame (the ball from the nearest
visible frame, as in the raw scale); the other continuous columns are 0 where undefined.
Python 3.9, numpy only.
"""

import numpy as np

from engine import data, tracks
from engine import learned_features as lf
from engine.config import Config

FPS = 25
LEVEL_M = 2.0
AHEAD_LAG = 25
MOTION_HALF = 12
ACC_HALF = 2
TURN_HALF = 3
TURN_MIN_MPS = 2.0
CLIP_BACC = 200.0
CLIP_PACC = 20.0
MAX_WINDOWS = (("m04", 10), ("m1", 25))
# Member spec F6c (round 8, protocol R8 note 6.3): robust units per match (robust_units()).
SCALE_RAW = "raw"
SCALE_ROBUST = "robust"
SCALES = (SCALE_RAW, SCALE_ROBUST)
IQR_FLOOR = 1e-3        # in the raw units of each column; fixed before any F6c fit
COUNT_COLUMNS = frozenset(["gbehind", "glevel", "gahead", "gdahead_p1", "gdahead_n1"])
POSITION_COLUMNS = frozenset(["gbc", "gbdeep", "gbfront"])

TEAM_COLUMNS = ["gbehind", "glevel", "gahead", "gdahead_p1", "gdahead_n1", "gbc", "gbdeep",
                "gbfront", "gvc", "gvdeep", "gvfront", "garea", "gnacc"]
GLOBAL_COLUMNS = ["g_ovx", "g_ovy", "g_gapx", "g_cdist", "g_bacc", "g_bturn"] + [
    "g_{}_{}".format(b, s) for b in ("bacc", "bturn") for s, _ in MAX_WINDOWS]


def group_spec():
    """[(name, kind)] of the group columns, in their order."""
    spec = []
    for team in ("H", "A"):
        for name in TEAM_COLUMNS:
            spec.append(("{}_{}".format(team, name), team))
    return spec + [(name, "inv") for name in GLOBAL_COLUMNS]


GROUP_SPEC = group_spec()
GROUP_NAMES = [n for n, _ in GROUP_SPEC]
GROUP_SET = frozenset(GROUP_NAMES)


def _block_index(n, blocks):
    """(first, last) int arrays: the graded block of each frame (-1, -1 outside the blocks)."""
    first = np.full(n, -1, np.int64)
    last = np.full(n, -1, np.int64)
    for _, a, b in blocks:
        first[a:b + 1] = a
        last[a:b + 1] = b
    return first, last


def _lagged(first, last, offset):
    """int array: t + offset clamped into the frame's block (t itself outside the blocks)."""
    t = np.arange(len(first))
    inside = first >= 0
    return np.where(inside, np.clip(t + offset, first, last), t)


def _sliding_max(x, half, first, last):
    """max of x over [t - half, t + half] clamped into the frame's block; 0 outside."""
    n = len(x)
    out = np.zeros(n)
    t = np.arange(n)
    for k in range(-half, half + 1):
        j = np.clip(t + k, first, last)
        out = np.maximum(out, np.where(first >= 0, x[np.clip(j, 0, n - 1)], 0.0))
    return np.where(first >= 0, out, 0.0)


def _team_lines(a, tv, centre):
    """(centroid, deepest outfield line, front line, outfield mask) of one team in its frame."""
    cnt = tv.sum(axis=1)
    cen = np.where(tv, a, 0.0).sum(axis=1) / np.maximum(cnt, 1)
    srt = np.sort(np.where(tv, a, np.inf), axis=1)
    rows = np.arange(a.shape[0])
    if a.shape[1] == 0:
        c = np.full(a.shape[0], centre)
        return c, c, c, tv
    second = np.where(cnt >= 2, 1, 0)
    deep = srt[rows, np.clip(second, 0, a.shape[1] - 1)]
    front = np.where(tv, a, -np.inf).max(axis=1)
    none = cnt == 0
    cen = np.where(none, centre, cen)
    deep = np.where(none, centre, deep)
    front = np.where(none, centre, front)
    # outfield: every visible player but the one with the smallest a (stable: lowest slot)
    gk = np.argmin(np.where(tv, a, np.inf), axis=1)
    outfield = tv.copy()
    outfield[rows, gk] = False
    outfield &= (cnt >= 1)[:, None]
    return cen, deep, front, outfield


def _ellipse_area(x, y, mask):
    """pi sqrt(det cov) of the masked points per row (population covariance), 0 below 3."""
    cnt = mask.sum(axis=1)
    c = np.maximum(cnt, 1)
    mx = np.where(mask, x, 0.0).sum(axis=1) / c
    my = np.where(mask, y, 0.0).sum(axis=1) / c
    dx = np.where(mask, x - mx[:, None], 0.0)
    dy = np.where(mask, y - my[:, None], 0.0)
    sxx = (dx * dx).sum(axis=1) / c
    syy = (dy * dy).sum(axis=1) / c
    sxy = (dx * dy).sum(axis=1) / c
    det = np.maximum(sxx * syy - sxy * sxy, 0.0)
    return np.where(cnt >= 3, np.pi * np.sqrt(det), 0.0)


def _range(v, mask):
    lo = np.where(mask, v, np.inf).min(axis=1)
    hi = np.where(mask, v, -np.inf).max(axis=1)
    ok = mask.any(axis=1)
    return np.where(ok, lo, 0.0), np.where(ok, hi, 0.0), ok


def _overlap(lo1, hi1, ok1, lo2, hi2, ok2):
    inter = np.maximum(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), 0.0)
    union = np.maximum(hi1, hi2) - np.minimum(lo1, lo2)
    ok = ok1 & ok2 & (union > 0)
    return np.where(ok, inter / np.where(ok, union, 1.0), 0.0)


def _accel(vx, vy, ok, half, first, last):
    """|v(t + half) - v(t - half)| / (2 half / FPS) where ok at the three frames (indices clamped
    into the block; a clamped frame must still be `half` away, else 0)."""
    n = len(vx)
    t = np.arange(n)
    a = np.clip(t - half, 0, n - 1)
    b = np.clip(t + half, 0, n - 1)
    inside = (first >= 0) & (t - half >= first) & (t + half <= last)
    good = inside & ok & ok[a] & ok[b]
    with np.errstate(invalid="ignore"):
        acc = np.hypot(vx[b] - vx[a], vy[b] - vy[a]) * FPS / (2.0 * half)
    return np.where(good & np.isfinite(acc), acc, 0.0), a, b, good


def _robust_stats(values):
    """(median, inter-quartile range floored at IQR_FLOOR) of a 1-d float64 array; (0, 1) for
    fewer than two values."""
    if values.size < 2:
        return 0.0, 1.0
    q25, q50, q75 = np.percentile(values, [25.0, 50.0, 75.0])
    return float(q50), float(max(q75 - q25, IQR_FLOOR))


def robust_units(cols, defd):
    """The continuous columns of `cols` in robust units of the match (member spec F6c):
    (value - median) / max(inter-quartile range, IQR_FLOOR), median and range over the frames
    where the column is defined (`defd`), the two teams' versions of a team column pooled (so a
    team swap stays an exact relabelling and a difference between the teams is kept). Count
    columns (COUNT_COLUMNS) stay as they are. Then the missing-value convention: the ball-position
    columns (POSITION_COLUMNS: the ball from the nearest visible frame where it is hidden) keep
    their rescaled value on every frame; every other continuous column is 0 where undefined."""
    out = dict(cols)
    done = set()
    for name in GROUP_NAMES:
        stem = name[2:] if name[:2] in ("H_", "A_") else None
        if (stem or name) in COUNT_COLUMNS or name in done:
            continue
        names = ["H_" + stem, "A_" + stem] if stem else [name]
        pooled = np.concatenate([np.asarray(cols[k], np.float64)[defd[k]] for k in names])
        med, iqr = _robust_stats(pooled)
        for k in names:
            z = (np.asarray(cols[k], np.float64) - med) / iqr
            out[k] = z if (stem or name) in POSITION_COLUMNS else np.where(defd[k], z, 0.0)
            done.add(k)
    return out


def group_features(tac, game, base_version=lf.VERSION_V2T, cfg=None, scale=SCALE_RAW):
    """float32 (N, len(GROUP_SPEC)) group columns of one match (columns of GROUP_SPEC).

    scale (SCALES): SCALE_RAW, the units above (member spec F6a / F6b); SCALE_ROBUST, the
    continuous columns in robust units of this match (robust_units(); member spec F6c)."""
    lf.check_version(base_version)
    if scale not in SCALES:
        raise ValueError("group scale must be one of {}, not {!r}".format(SCALES, scale))
    v2t = base_version == lf.VERSION_V2T
    v2s = base_version in (lf.VERSION_V2S, lf.VERSION_V2T)
    cfg = cfg or Config.default()
    length, width = data.pitch_m(game)
    px, py, bx, by = lf.oriented_arrays(tac, game, away_fallback=v2s, compare_teams=v2t)
    team = np.asarray(tac["team_of_slot"])
    n = px.shape[0]
    blocks = data.graded_blocks(game)
    first, last = _block_index(n, blocks)
    inside = first >= 0

    ball = tracks.clean_ball(bx, by, (length, width), cfg, fps=float(FPS))
    vis = ball["visible"]
    bxm = lf._fill_nearest(np.nan_to_num(ball["x"]), vis)
    bym = lf._fill_nearest(np.nan_to_num(ball["y"]), vis)
    bvx = np.where(vis, np.nan_to_num(ball["vx"]), 0.0)
    bvy = np.where(vis, np.nan_to_num(ball["vy"]), 0.0)

    pxm, pym = px * length, py * width
    pvis = np.isfinite(pxm) & np.isfinite(pym)
    pvx, pvy = tracks._difference_velocity(pxm, pym, pvis, float(FPS))
    dist = np.where(pvis, np.hypot(pxm - bxm[:, None], pym - bym[:, None]), np.inf)
    vok = pvis & np.isfinite(pvx) & np.isfinite(pvy)
    rows = np.arange(n)
    lag_p = _lagged(first, last, -AHEAD_LAG)
    lag_n = _lagged(first, last, AHEAD_LAG)
    m_lo = _lagged(first, last, -MOTION_HALF)
    m_hi = _lagged(first, last, MOTION_HALF)
    m_dt = (m_hi - m_lo) / float(FPS)

    def motion(s):
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(m_dt > 0, (s[m_hi] - s[m_lo]) / np.where(m_dt > 0, m_dt, 1.0), 0.0)
        return np.where(inside, v, 0.0) / 5.0

    cols = {}
    defd = {}
    outfield_xy = {}
    centroid_xy = {}
    for key, t_id in (("H", 0), ("A", 1)):
        c = np.flatnonzero(team == t_id)
        tv = pvis[:, c]
        x = pxm[:, c]
        a = x if key == "H" else length - x
        a_ball = bxm if key == "H" else length - bxm
        a = np.where(tv, a, 0.0)

        def cnt11(m):
            return np.minimum(m.sum(axis=1), 11) / 11.0
        behind = tv & (a < a_ball[:, None] - LEVEL_M)
        ahead = tv & (a > a_ball[:, None] + LEVEL_M)
        level = tv & ~behind & ~ahead
        n_ahead = cnt11(ahead)
        cols[key + "_gbehind"] = cnt11(behind)
        cols[key + "_glevel"] = cnt11(level)
        cols[key + "_gahead"] = n_ahead
        cols[key + "_gdahead_p1"] = np.where(inside, n_ahead - n_ahead[lag_p], 0.0)
        cols[key + "_gdahead_n1"] = np.where(inside, n_ahead[lag_n] - n_ahead, 0.0)
        cen, deep, front, outfield = _team_lines(a, tv, length / 2.0)
        cols[key + "_gbc"] = (a_ball - cen) / 10.0
        cols[key + "_gbdeep"] = (a_ball - deep) / 10.0
        cols[key + "_gbfront"] = (a_ball - front) / 10.0
        cols[key + "_gvc"] = motion(cen)
        cols[key + "_gvdeep"] = motion(deep)
        cols[key + "_gvfront"] = motion(front)
        y = pym[:, c]
        cols[key + "_garea"] = _ellipse_area(np.where(tv, x, 0.0), np.where(tv, y, 0.0),
                                             outfield) / 1000.0
        for name in ("_gbc", "_gbdeep", "_gbfront"):
            defd[key + name] = inside & vis
        for name in ("_gvc", "_gvdeep", "_gvfront"):
            defd[key + name] = inside.copy()
        defd[key + "_garea"] = inside & (outfield.sum(axis=1) >= 3)
        defd[key + "_gnacc"] = np.zeros(n, bool)
        outfield_xy[key] = (np.where(tv, x, 0.0), np.where(tv, y, 0.0), outfield)
        cnt = np.maximum(tv.sum(axis=1), 1)
        none = tv.sum(axis=1) == 0
        cx = np.where(none, length / 2.0, np.where(tv, x, 0.0).sum(axis=1) / cnt)
        cy = np.where(none, width / 2.0, np.where(tv, y, 0.0).sum(axis=1) / cnt)
        centroid_xy[key] = (cx, cy)
        # acceleration of the team's player nearest to the ball
        sub = dist[:, c]
        if c.size:
            order = np.argsort(sub, axis=1, kind="stable")
            slot = c[order[:, 0]]
            found = np.isfinite(sub[rows, order[:, 0]])
            t = np.arange(n)
            lo = np.clip(t - ACC_HALF, 0, n - 1)
            hi = np.clip(t + ACC_HALF, 0, n - 1)
            ok = (vok[lo, slot] & vok[hi, slot] & found & inside
                  & (t - ACC_HALF >= first) & (t + ACC_HALF <= last))
            with np.errstate(invalid="ignore"):
                acc_all = np.hypot(pvx[hi, slot] - pvx[lo, slot], pvy[hi, slot] - pvy[lo, slot])
            acc_all = acc_all * FPS / (2.0 * ACC_HALF)
            acc = np.where(ok & np.isfinite(acc_all), np.minimum(acc_all, CLIP_PACC), 0.0)
            defd[key + "_gnacc"] = ok & np.isfinite(acc_all) & vis
        else:
            acc = np.zeros(n)
        cols[key + "_gnacc"] = acc / 5.0

    hx, hy, hm = outfield_xy["H"]
    ax, ay, am = outfield_xy["A"]
    hlo, hhi, hok = _range(hx, hm)
    alo, ahi, aok = _range(ax, am)
    cols["g_ovx"] = _overlap(hlo, hhi, hok, alo, ahi, aok)
    hlo, hhi, hok = _range(hy, hm)
    alo, ahi, aok = _range(ay, am)
    cols["g_ovy"] = _overlap(hlo, hhi, hok, alo, ahi, aok)
    defd["g_ovx"] = defd["g_ovy"] = inside & hok & aok
    both = inside & pvis[:, team == 0].any(axis=1) & pvis[:, team == 1].any(axis=1)
    defd["g_gapx"] = defd["g_cdist"] = both
    (hcx, hcy), (acx, acy) = centroid_xy["H"], centroid_xy["A"]
    cols["g_gapx"] = (acx - hcx) / 10.0
    cols["g_cdist"] = np.hypot(acx - hcx, acy - hcy) / 10.0

    bacc, _, _, bgood = _accel(bvx, bvy, vis, ACC_HALF, first, last)
    defd["g_bacc"] = bgood
    bacc = np.minimum(bacc, CLIP_BACC) / 50.0
    t = np.arange(n)
    a = np.clip(t - TURN_HALF, 0, n - 1)
    b = np.clip(t + TURN_HALF, 0, n - 1)
    sa = np.hypot(bvx[a], bvy[a])
    sb = np.hypot(bvx[b], bvy[b])
    good = (inside & (t - TURN_HALF >= first) & (t + TURN_HALF <= last) & vis & vis[a] & vis[b]
            & (sa >= TURN_MIN_MPS) & (sb >= TURN_MIN_MPS))
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (bvx[a] * bvx[b] + bvy[a] * bvy[b]) / np.where(good, sa * sb, 1.0)
    turn = np.where(good, np.arccos(np.clip(cosang, -1.0, 1.0)) / np.pi, 0.0)
    cols["g_bacc"] = bacc
    cols["g_bturn"] = turn
    defd["g_bturn"] = good
    for suf, half in MAX_WINDOWS:
        cols["g_bacc_" + suf] = _sliding_max(bacc, half, first, last)
        cols["g_bturn_" + suf] = _sliding_max(turn, half, first, last)
        defd["g_bacc_" + suf] = defd["g_bturn_" + suf] = inside & vis
    if scale == SCALE_ROBUST:
        cols = robust_units(cols, defd)
    out = np.stack([cols[name] for name in GROUP_NAMES], axis=1)
    return np.nan_to_num(out).astype(np.float32)

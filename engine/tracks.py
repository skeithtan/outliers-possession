"""Tracking cleaning, ball kinematics and per-frame features (PLAN.md 3.2; card P0-B).

    clean_ball(bx, by, pitch, cfg)            cleaned ball in metres + velocity, dv_k, heading, airborne
    frame_features(tactical, ball, pitch, cfg)  per-frame per-team geometry, dict of (N,) arrays
    slot_confidence(tactical, cfg)            float32 (N, n_slots) team-assignment confidence
    mark_nearest(tactical, pitch)             nearest player to the centre mark per frame
    guard_goalkeeper_swaps(tactical, blocks)  the C9 goalkeeper slot-swap guard, run before prepare()
    team_forward_velocity(tactical, blocks, pitch)  forward velocity of each team (C15 shape check)
    speed_from_positions(tactical, pitch)     player speed from positions for an ATD without speed
    prepare(tactical, cfg)                    all three, with the full airborne rule in `ball`

Conventions (PLAN.md 0, 3.2, 6):
  * Metres, seconds, m/s inside; the normalised 0-1 ATD coordinates are converted here with the
    pitch size from games.py (engine.data.pitch_m()).
  * Row i is video frame i. NaN = undefined / not observed.
  * A velocity step is in m/s over k frames, indexed at the START of the window:
    dv_k[t] = |speed[t + k] - speed[t]|. A kick that happens between frames t and t + 1 shows
    at t (and on the plateau before it). Headings are in degrees, in [0, 180].
  * No median filter: the ATD ball is already smooth (PLAN 1.1, findings R-F3).
  * Every threshold comes from engine.config.Config.

numpy + stdlib only. Never reads the ground truth.
"""

import numpy as np

from engine import data
from engine.config import Config

HOME, AWAY = data.HOME, data.AWAY
TEAM_KEYS = {HOME: "H", AWAY: "A"}


def _fps_or_default(fps):
    """fps as float; defaults to the match fps from games.py (via engine.data)."""
    if fps is None:
        fps = data.fps()
    return float(fps)


# -- small array helpers -----------------------------------------------------------------------

def _runs(mask):
    """(starts, stops) of the maximal True runs of a 1-D bool array, half-open."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2]


def _run_ids(mask):
    """int64 (N,): index of the True run each frame belongs to, -1 outside runs."""
    mask = np.asarray(mask, dtype=bool)
    starts = np.zeros(mask.shape, dtype=bool)
    starts[0] = mask[0]
    starts[1:] = mask[1:] & ~mask[:-1]
    ids = np.cumsum(starts) - 1
    return np.where(mask, ids, -1).astype(np.int64)


def _shift(values, k, fill=np.nan):
    """values[t + k] at index t (k may be negative); `fill` where t + k is outside the array."""
    out = np.full(values.shape, fill, dtype=np.result_type(values, type(fill)))
    n = values.shape[0]
    if k >= 0:
        if k < n:
            out[:n - k] = values[k:]
    elif -k < n:
        out[-k:] = values[:n + k]
    return out


def _same_run(run_id, k):
    """bool (N,): frames t and t + k are visible and in the same visible run."""
    ahead = _shift(run_id, k, fill=-1)
    return (run_id >= 0) & (ahead == run_id)


def _difference_velocity(x, y, visible, fps):
    """Velocity (m/s) from positions, per column, on visible frames; NaN elsewhere.

    Central difference where both neighbours are visible; one-sided (forward at a run start,
    backward at a run end) where only one is; NaN for an isolated visible frame. Works on (N,)
    or (N, S) arrays (axis 0 is time).
    """
    prev_ok = np.zeros(visible.shape, dtype=bool)
    next_ok = np.zeros(visible.shape, dtype=bool)
    prev_ok[1:] = visible[1:] & visible[:-1]
    next_ok[:-1] = visible[:-1] & visible[1:]

    def one(pos):
        prev = np.full(pos.shape, np.nan)
        nxt = np.full(pos.shape, np.nan)
        prev[1:] = pos[:-1]
        nxt[:-1] = pos[1:]
        central = (nxt - prev) * (fps / 2.0)
        forward = (nxt - pos) * fps
        backward = (pos - prev) * fps
        out = np.where(prev_ok & next_ok, central,
                       np.where(next_ok, forward, np.where(prev_ok, backward, np.nan)))
        return np.where(visible, out, np.nan)

    return one(x), one(y)


# -- ball -------------------------------------------------------------------------------------

def _teleports(x, y, raw_visible, max_jump):
    """bool (N,): frames rejected as teleports (PLAN 1.1: > 3 m in one frame).

    Frame t is rejected when BOTH hold:
      * the step from the immediately preceding frame t - 1 exceeds `max_jump` (t - 1 must be
        raw-visible: a visible run that starts after a gap is never a jump), and
      * t is not back within reach of the last accepted frame a, i.e. |p[t] - p[a]| exceeds
        max_jump x (t - a).
    The second clause keeps the frame after a one-frame spike (it jumps back next to the
    accepted frame), so a spike rejects only itself. A streak of implausibly fast motion rejects
    every frame whose own step is > max_jump and ends at the first plausible step. On the match
    data the jumps are such streaks (smooth 75-110 m/s motion), not single-frame spikes.
    """
    n = x.shape[0]
    step = np.full(n, np.inf)
    step[1:] = np.hypot(np.diff(x), np.diff(y))
    candidate = np.zeros(n, dtype=bool)
    candidate[1:] = raw_visible[1:] & raw_visible[:-1] & (step[1:] > max_jump)

    rejected = np.zeros(n, dtype=bool)
    resume = 0  # first frame not yet decided by a previous span
    for t0 in np.flatnonzero(candidate):
        if t0 < resume:
            continue
        last_ok = t0 - 1
        t = t0
        while t < n and raw_visible[t]:
            back = np.hypot(x[t] - x[last_ok], y[t] - y[last_ok]) <= max_jump * (t - last_ok)
            if step[t] > max_jump and not back:
                rejected[t] = True
                t += 1
            else:
                break
        resume = t + 1  # frame t is accepted (or missing): later candidates start afresh
    return rejected


def velocity_step(speed, k, run_id=None):
    """Signed velocity step speed[t + k] - speed[t], m/s over k frames, indexed at t.

    NaN where either end is NaN, beyond the end of the array, or (with `run_id`) when t and t + k
    are not in the same visible run: a reappearance after a gap is not a kick.
    P1-B's restart rule uses it with k = 9 (dv_9); the touch detector uses abs(...) with k = DV_K.
    """
    speed = np.asarray(speed, dtype=np.float64)
    out = _shift(speed, k) - speed
    if run_id is not None:
        out = np.where(_same_run(run_id, k), out, np.nan)
    return out


def heading_change(vx, vy, k, run_id=None, min_speed=0.0):
    """Angle between the velocity at t and at t + k, degrees in [0, 180], indexed at t.

    NaN where either velocity is NaN or slower than `min_speed` (heading undefined), beyond the
    end, or (with `run_id`) across a gap.
    """
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    vx2, vy2 = _shift(vx, k), _shift(vy, k)
    dot = vx * vx2 + vy * vy2
    cross = vx * vy2 - vy * vx2
    out = np.degrees(np.arctan2(np.abs(cross), dot))
    speed, speed2 = np.hypot(vx, vy), np.hypot(vx2, vy2)
    with np.errstate(invalid="ignore"):
        defined = (speed >= min_speed) & (speed2 >= min_speed) & (speed > 0) & (speed2 > 0)
    if run_id is not None:
        defined &= _same_run(run_id, k)
    return np.where(defined, out, np.nan)


def airborne_suspect(speed, run_id, dmin, cfg):
    """bool (N,): ball suspected in the air (PLAN 3.2; P0-B card).

    True where speed > AIRBORNE_SPEED, or on the frames t..t+AIRBORNE_DECEL_FRAMES of a window in
    which the speed starts at >= AIRBORNE_DECEL_MIN_SPEED and falls by more than
    AIRBORNE_DECEL_FRAC with no player within AIRBORNE_CLEAR_M on any frame of the window
    (a rolling ball cannot lose 40% in 0.12 s without a contact). `dmin` is the nearest-player
    distance of either team (NaN = no player visible, counted as clear). With dmin=None only the
    speed clause is evaluated.
    """
    speed = np.asarray(speed, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        fast = speed > cfg.AIRBORNE_SPEED
    if dmin is None:
        return fast
    k = cfg.AIRBORNE_DECEL_FRAMES
    later = _shift(speed, k)
    clear = np.where(np.isfinite(dmin), dmin, np.inf) > cfg.AIRBORNE_CLEAR_M
    clear_window = clear.copy()
    for j in range(1, k + 1):
        clear_window &= _shift(clear, j, fill=False)
    with np.errstate(invalid="ignore"):
        drop = ((speed >= cfg.AIRBORNE_DECEL_MIN_SPEED)
                & (later < (1.0 - cfg.AIRBORNE_DECEL_FRAC) * speed))
    starts = drop & _same_run(run_id, k) & clear_window
    window = starts.copy()
    for j in range(1, k + 1):
        window |= _shift(starts, -j, fill=False)
    return fast | window


def clean_ball(bx, by, pitch, cfg=None, dmin=None, fps=None):
    """Cleaned ball track and kinematics in metres / m/s (PLAN 1.1, 3.2).

    Steps: normalised -> metres; reject teleports (> BALL_MAX_JUMP_M per frame, see _teleports);
    then drop visible runs shorter than BALL_MIN_RUN frames. Velocity by central difference
    inside visible runs and one-sided at run ends (so a ball that reappears moving has a speed
    on its first frame; runs are >= BALL_MIN_RUN >= 2 frames long, so it is always defined).

    Returns a dict of (N,) arrays:
      x_raw, y_raw        float64 m, the ATD ball as given (NaN = missing), teleports included
      x, y                float64 m, cleaned (NaN where not visible)
      visible             bool, cleaned visibility
      teleport            bool, frames rejected by the jump rule
      run_id              int64, index of the cleaned visible run, -1 when not visible
      vx, vy, speed       float64 m/s
      dv_signed_k, dv_k   speed[t+k] - speed[t] and its absolute value, m/s over k = DV_K frames,
                          NaN across gaps (also stored as dv_signed_<k>, dv_<k>)
      heading_change_k    degrees over k frames, NaN below HEADING_MIN_SPEED (also heading_change_<k>)
      airborne_fast       bool, speed clause of the airborne rule only
      airborne_suspect    bool, full rule if `dmin` is given, else equal to airborne_fast
      k, fps              the window and frame rate used
    `prepare()` / `frame_features()` supply `dmin` and the full airborne rule.
    """
    cfg = cfg or Config.default()
    fps = _fps_or_default(fps)
    length, width = pitch
    x_raw = np.asarray(bx, dtype=np.float64) * float(length)
    y_raw = np.asarray(by, dtype=np.float64) * float(width)
    raw_visible = np.isfinite(x_raw) & np.isfinite(y_raw)

    teleport = _teleports(x_raw, y_raw, raw_visible, cfg.BALL_MAX_JUMP_M)
    visible = raw_visible & ~teleport
    starts, stops = _runs(visible)
    for start, stop in zip(starts, stops):
        if stop - start < cfg.BALL_MIN_RUN:
            visible[start:stop] = False
    run_id = _run_ids(visible)

    x = np.where(visible, x_raw, np.nan)
    y = np.where(visible, y_raw, np.nan)
    vx, vy = _difference_velocity(x, y, visible, fps)
    speed = np.hypot(vx, vy)

    k = int(cfg.DV_K)
    dv_signed = velocity_step(speed, k, run_id)
    heading = heading_change(vx, vy, k, run_id, cfg.HEADING_MIN_SPEED)
    fast = airborne_suspect(speed, run_id, None, cfg)
    out = {
        "x_raw": x_raw, "y_raw": y_raw, "x": x, "y": y,
        "visible": visible, "teleport": teleport, "run_id": run_id,
        "vx": vx, "vy": vy, "speed": speed,
        "dv_signed_k": dv_signed, "dv_k": np.abs(dv_signed), "heading_change_k": heading,
        "airborne_fast": fast,
        "airborne_suspect": (airborne_suspect(speed, run_id, dmin, cfg)
                             if dmin is not None else fast.copy()),
        "k": k, "fps": fps,
    }
    out["dv_signed_{}".format(k)] = out["dv_signed_k"]
    out["dv_{}".format(k)] = out["dv_k"]
    out["heading_change_{}".format(k)] = out["heading_change_k"]
    return out


# -- convex hull ------------------------------------------------------------------------------

def hull_area(xs, ys):
    """Convex-hull area per row of point sets (m^2), vectorised gift wrapping.

    xs, ys: (F, P) with NaN for absent points. Rows with < 3 points, or collinear points, give 0.
    Counter-clockwise Jarvis march from the leftmost (then lowest) point; at each step the next
    vertex is the candidate with every other point on its left (farthest on ties), and the
    shoelace term of the edge is accumulated. Loops are over hull steps and candidate columns
    (<= P each); every operation is over all rows at once.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n_rows, n_pts = xs.shape
    area = np.zeros(n_rows)
    valid = np.isfinite(xs) & np.isfinite(ys)
    rows = np.flatnonzero(valid.sum(axis=1) >= 3)
    if rows.size == 0:
        return area
    V = valid[rows]
    cnt = V.sum(axis=1)
    # Centre each row on its mean point to keep cross products well conditioned.
    X = np.where(V, xs[rows], 0.0)
    Y = np.where(V, ys[rows], 0.0)
    X = np.where(V, X - (X.sum(axis=1) / cnt)[:, None], 0.0)
    Y = np.where(V, Y - (Y.sum(axis=1) / cnt)[:, None], 0.0)

    x_key = np.where(V, X, np.inf)
    leftmost = V & (X == x_key.min(axis=1)[:, None])
    start = np.argmin(np.where(leftmost, Y, np.inf), axis=1)

    idx = np.arange(rows.size)       # rows still wrapping (positions into X, Y, V)
    cur = start.copy()
    acc = np.zeros(rows.size)
    for _ in range(n_pts + 1):
        if idx.size == 0:
            break
        Xa, Ya, Va = X[idx], Y[idx], V[idx]
        ca = cur[idx]
        ox, oy = Xa[np.arange(idx.size), ca], Ya[np.arange(idx.size), ca]
        best = np.full(idx.size, -1, dtype=np.int64)
        bdx = np.zeros(idx.size)
        bdy = np.zeros(idx.size)
        for j in range(n_pts):
            ok = Va[:, j] & (ca != j)
            dx, dy = Xa[:, j] - ox, Ya[:, j] - oy
            cross = bdx * dy - bdy * dx
            better = (best < 0) | (cross < 0) | (
                (cross == 0) & (dx * dx + dy * dy > bdx * bdx + bdy * bdy))
            take = ok & better
            best = np.where(take, j, best)
            bdx = np.where(take, dx, bdx)
            bdy = np.where(take, dy, bdy)
        nx, ny = ox + bdx, oy + bdy
        acc[idx] += 0.5 * (ox * ny - oy * nx)
        cur[idx] = best
        idx = idx[best != start[idx]]
    if idx.size:  # not closed within the step cap (degenerate input): close the polygon
        ca = cur[idx]
        ox, oy = X[idx, ca], Y[idx, ca]
        sx, sy = X[idx, start[idx]], Y[idx, start[idx]]
        acc[idx] += 0.5 * (ox * sy - oy * sx)
    area[rows] = np.abs(acc)
    return area


# -- per-frame features -----------------------------------------------------------------------

def _masked_mean_std(values, mask):
    """Mean and population std over the True entries of each row; NaN where a row has none."""
    count = mask.sum(axis=1)
    filled = np.where(mask, values, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = filled.sum(axis=1) / count
        var = np.where(mask, (values - mean[:, None]) ** 2, 0.0).sum(axis=1) / count
    mean = np.where(count > 0, mean, np.nan)
    std = np.where(count > 0, np.sqrt(var), np.nan)
    return mean, std, count


def _player_metres(tactical, pitch):
    """Player x, y (m), visibility and velocity (m/s) from position differences."""
    length, width = pitch
    px = np.asarray(tactical["px"], dtype=np.float64) * float(length)
    py = np.asarray(tactical["py"], dtype=np.float64) * float(width)
    visible = np.isfinite(px) & np.isfinite(py)
    return px, py, visible


def frame_features(tactical, ball, pitch, cfg=None, fps=None):
    """Per-frame features of the players around the ball and of each team's shape (PLAN 3.2).

    Returns a dict of (N,) arrays (float64 m, m/s, m^2 unless stated). Per team T in {H, A}:
      dT              nearest T player to the ball, m (NaN if ball not visible or no T player)
      nearest_slot_T  int64 slot of that player, -1 if none
      nT_15, nT_30    int16 T players within N_RADIUS_TIGHT_M / N_RADIUS_WIDE_M (0 if no ball)
      vT_near         speed of the nearest T player: the ATD `ps` column (m/s) where present,
                      else from position differences
      relT            |ball velocity - nearest T player velocity|, player velocity from position
                      differences (central, one-sided at tracklet ends)
      cxT, cyT        centroid of visible T players; sxT, syT their population std
      meanvT          mean `ps` of visible T players with a speed value
      nvisT           int16 visible T tracks
      hullT           convex-hull area of visible T players (0 with < 3 players)
    Across teams:
      sep             |cxH - cxA|
      meanv_all       mean `ps` over all visible players with a speed value
      n_tracks        int16 visible tracks
      dmin            nearest player of either team, m; nearest_team int8 (0 H, 1 A, -1 none)
      d_second        second-nearest player of either team, m (for the contested rule d2 - d1)
      meanv_near      mean speed (`ps`, else position differences) of the PURSUIT_N_NEAREST
                      players nearest the ball, either team, NaN without ball or players
                      (P1-B pursuit stop: "the 6 players nearest the ball decelerate", PLAN 3.4)
      airborne_suspect  bool, the full airborne rule of clean_ball with dmin (P0-B card)
    Ball-dependent features use the CLEANED ball (`ball["x"]`, `ball["vx"]`).
    """
    cfg = cfg or Config.default()
    fps = _fps_or_default(fps if fps is not None else tactical.get("fps"))
    team_of_slot = np.asarray(tactical["team_of_slot"])
    px, py, visible = _player_metres(tactical, pitch)
    pvx, pvy = _difference_velocity(px, py, visible, fps)
    ps = np.asarray(tactical["ps"], dtype=np.float64)
    ps = np.where(visible, ps, np.nan)
    p_speed = np.where(np.isfinite(ps), ps, np.hypot(pvx, pvy))
    has_ps = visible & np.isfinite(ps)

    bxm, bym = ball["x"], ball["y"]
    dist = np.hypot(px - bxm[:, None], py - bym[:, None])
    dist_inf = np.where(np.isfinite(dist), dist, np.inf)
    n_frames = px.shape[0]
    frame_idx = np.arange(n_frames)

    out = {}
    for team, key in sorted(TEAM_KEYS.items()):
        cols = np.flatnonzero(team_of_slot == team)
        sub = dist_inf[:, cols]
        if cols.size:
            arg = np.argmin(sub, axis=1)
            nearest = sub[frame_idx, arg]
        else:
            arg = np.zeros(n_frames, dtype=np.int64)
            nearest = np.full(n_frames, np.inf)
        found = np.isfinite(nearest)
        slot = np.where(found, cols[arg] if cols.size else -1, -1).astype(np.int64)
        safe = np.where(found, slot, 0)
        out["d" + key] = np.where(found, nearest, np.nan)
        out["nearest_slot_" + key] = slot
        out["n{}_15".format(key)] = (sub <= cfg.N_RADIUS_TIGHT_M).sum(axis=1).astype(np.int16)
        out["n{}_30".format(key)] = (sub <= cfg.N_RADIUS_WIDE_M).sum(axis=1).astype(np.int16)
        out["v{}_near".format(key)] = np.where(found, p_speed[frame_idx, safe], np.nan)
        rel = np.hypot(ball["vx"] - pvx[frame_idx, safe], ball["vy"] - pvy[frame_idx, safe])
        out["rel" + key] = np.where(found, rel, np.nan)

        tvis = visible[:, cols]
        cx, sx, count = _masked_mean_std(px[:, cols], tvis)
        cy, sy, _ = _masked_mean_std(py[:, cols], tvis)
        meanv, _, _ = _masked_mean_std(ps[:, cols], has_ps[:, cols])
        out["cx" + key], out["cy" + key] = cx, cy
        out["sx" + key], out["sy" + key] = sx, sy
        out["meanv" + key] = meanv
        out["nvis" + key] = count.astype(np.int16)
        out["hull" + key] = hull_area(np.where(tvis, px[:, cols], np.nan),
                                      np.where(tvis, py[:, cols], np.nan))

    out["sep"] = np.abs(out["cxH"] - out["cxA"])
    out["meanv_all"], _, _ = _masked_mean_std(ps, has_ps)
    out["n_tracks"] = visible.sum(axis=1).astype(np.int16)

    dmin = np.fmin(out["dH"], out["dA"])
    out["dmin"] = dmin
    nearest_team = np.where(out["dH"] <= out["dA"], HOME, AWAY)
    nearest_team = np.where(np.isfinite(out["dH"]) & ~np.isfinite(out["dA"]), HOME, nearest_team)
    nearest_team = np.where(~np.isfinite(out["dH"]) & np.isfinite(out["dA"]), AWAY, nearest_team)
    out["nearest_team"] = np.where(np.isfinite(dmin), nearest_team, -1).astype(np.int8)
    if dist_inf.shape[1] >= 2:
        second = np.partition(dist_inf, 1, axis=1)[:, 1]
    else:
        second = np.full(n_frames, np.inf)
    out["d_second"] = np.where(np.isfinite(second), second, np.nan)
    out["meanv_near"] = _nearest_mean_speed(dist_inf, p_speed, int(cfg.PURSUIT_N_NEAREST))
    out["airborne_suspect"] = airborne_suspect(ball["speed"], ball["run_id"], dmin, cfg)
    return out


def _nearest_mean_speed(dist_inf, p_speed, n_nearest):
    """Mean of p_speed over the n_nearest finite-distance players per row; NaN if none."""
    n_frames, n_slots = dist_inf.shape
    if n_slots == 0 or n_nearest <= 0:
        return np.full(n_frames, np.nan)
    take = min(n_nearest, n_slots)
    if take < n_slots:
        cols = np.argpartition(dist_inf, take - 1, axis=1)[:, :take]
    else:
        cols = np.tile(np.arange(n_slots), (n_frames, 1))
    rows = np.arange(n_frames)[:, None]
    ok = np.isfinite(dist_inf[rows, cols]) & np.isfinite(p_speed[rows, cols])
    count = ok.sum(axis=1)
    total = np.where(ok, p_speed[rows, cols], 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


# -- slot confidence --------------------------------------------------------------------------

def slot_confidence(tactical, cfg=None):
    """float32 (N, n_slots) confidence in each slot's team assignment (PLAN 1.2, 3.2).

    1.0, x OVERCOUNT_CONF_FACTOR in frames where the slot's team has more than MAX_TEAM_TRACKS
    visible tracks, x SHORT_TRACKLET_CONF_FACTOR on tracklets (maximal visible runs of the slot)
    shorter than MIN_TRACKLET_FRAMES. Frames where the slot is not visible keep their value; only
    visible slots are ever read downstream.
    """
    cfg = cfg or Config.default()
    team_of_slot = np.asarray(tactical["team_of_slot"])
    visible = np.isfinite(tactical["px"]) & np.isfinite(tactical["py"])
    n_frames, n_slots = visible.shape
    conf = np.ones((n_frames, n_slots), dtype=np.float32)

    for team in sorted(TEAM_KEYS):
        cols = np.flatnonzero(team_of_slot == team)
        over = visible[:, cols].sum(axis=1) > cfg.MAX_TEAM_TRACKS
        conf[np.ix_(over, cols)] *= np.float32(cfg.OVERCOUNT_CONF_FACTOR)

    for slot in range(n_slots):
        starts, stops = _runs(visible[:, slot])
        short = (stops - starts) < cfg.MIN_TRACKLET_FRAMES
        for start, stop in zip(starts[short], stops[short]):
            conf[start:stop, slot] *= np.float32(cfg.SHORT_TRACKLET_CONF_FACTOR)
    return conf


def mark_nearest(tactical, pitch):
    """The tracked player nearest the centre mark on each frame (Law 8, RESEARCH rank 3).

    Returns {"dist": float64 (N,) m, inf where no player is visible; "team": int8 (N,), 0 for
    H, 1 for A, -1 where no player is visible}. The centre mark is the middle of the pitch
    rectangle. A tie goes to the lower slot index.
    """
    length, width = pitch
    px, py, visible = _player_metres(tactical, pitch)
    team_of_slot = np.asarray(tactical["team_of_slot"])
    dist = np.where(visible, np.hypot(px - float(length) / 2.0, py - float(width) / 2.0), np.inf)
    n_frames = dist.shape[0]
    if dist.shape[1] == 0:
        return {"dist": np.full(n_frames, np.inf), "team": np.full(n_frames, -1, dtype=np.int8)}
    arg = np.argmin(dist, axis=1)
    near = dist[np.arange(n_frames), arg]
    team = np.where(np.isfinite(near), team_of_slot[arg], -1).astype(np.int8)
    return {"dist": near.astype(np.float64), "team": team}


# -- goalkeeper slot-swap guard (C9; RESEARCH-2 1d, M14) -------------------------------------

def penalty_area_mask(px, py, own_goal_x, pitch, cfg=None):
    """bool: the points (m) inside the penalty area in front of the goal line x = own_goal_x.

    own_goal_x is 0 for the goal at x = 0, and any other value for the goal at x = length.
    A NaN point is outside.
    """
    cfg = cfg or Config.default()
    length, width = pitch
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        if own_goal_x == 0:
            in_x = px < cfg.PENALTY_AREA_DEPTH_M
        else:
            in_x = px > float(length) - cfg.PENALTY_AREA_DEPTH_M
        in_y = np.abs(py - float(width) / 2.0) < cfg.PENALTY_AREA_WIDTH_M / 2.0
    return in_x & in_y


def goalkeeper_slots(tactical, blocks, pitch, cfg=None):
    """The goalkeeper-like slot of each team in each graded block, from the ATD only.

    The own goal of a team is at x = 0 if the median x of its visible players over the block is
    below the centre, else at x = length. The goalkeeper-like slot is the team slot with the most
    visible frames inside its own penalty area. The slot must also have an own-box share of at
    least GK_OWN_BOX_SHARE of its visible frames in the block, and at least
    GK_MIN_OWN_BOX_FRAMES own-box frames. Else the team has no goalkeeper-like slot (-1).

    `blocks` is [(half, first_frame, last_frame_inclusive)], as engine.data.graded_blocks().
    Returns {(half, team): {"slot", "own_goal_x", "own_box", "share", "visible"}} with team
    0 for H and 1 for A. The last three describe the best slot even when "slot" is -1.
    """
    cfg = cfg or Config.default()
    length = float(pitch[0])
    team_of_slot = np.asarray(tactical["team_of_slot"])
    px, py, visible = _player_metres(tactical, pitch)
    out = {}
    for half, first, last in blocks:
        rows = slice(int(first), int(last) + 1)
        bx, by, bvis = px[rows], py[rows], visible[rows]
        for team in sorted(TEAM_KEYS):
            cols = np.flatnonzero(team_of_slot == team)
            info = {"slot": -1, "own_goal_x": 0.0, "own_box": 0, "share": 0.0, "visible": 0}
            team_x = bx[:, cols][bvis[:, cols]]
            if team_x.size:
                own_goal_x = 0.0 if np.median(team_x) < length / 2.0 else length
                inside = bvis[:, cols] & penalty_area_mask(bx[:, cols], by[:, cols],
                                                           own_goal_x, pitch, cfg)
                counts = inside.sum(axis=0)
                n_visible = bvis[:, cols].sum(axis=0)
                best = int(np.argmax(counts))
                own_box, seen = int(counts[best]), int(n_visible[best])
                share = own_box / float(seen) if seen else 0.0
                info.update(own_goal_x=own_goal_x, own_box=own_box, share=share, visible=seen)
                if share >= cfg.GK_OWN_BOX_SHARE and own_box >= cfg.GK_MIN_OWN_BOX_FRAMES:
                    info["slot"] = int(cols[best])
            out[(int(half), int(team))] = info
    return out


def goalkeeper_positions(tactical, keepers, blocks, pitch):
    """The position of each team's goalkeeper-like slot on every frame (C13).

    `keepers` is the result of goalkeeper_slots() for the same `blocks`. Returns a dict of
    float64 (N, 2) arrays "x", "y" and "own_goal_x", with column 0 for H and column 1 for A.
    A value is NaN outside the graded blocks, where the team has no goalkeeper-like slot, and
    where that slot is not visible. The int64 (N, 2) array "slot" gives the slot number, -1
    where the other arrays are NaN.
    """
    px, py, visible = _player_metres(tactical, pitch)
    n_frames = px.shape[0]
    out = {key: np.full((n_frames, 2), np.nan) for key in ("x", "y", "own_goal_x")}
    out["slot"] = np.full((n_frames, 2), -1, dtype=np.int64)
    for half, first, last in blocks:
        rows = np.arange(int(first), min(int(last) + 1, n_frames))
        for team in sorted(TEAM_KEYS):
            info = keepers.get((int(half), int(team)))
            if info is None or info["slot"] < 0:
                continue
            slot = int(info["slot"])
            seen = rows[visible[rows, slot]]
            out["x"][seen, team] = px[seen, slot]
            out["y"][seen, team] = py[seen, slot]
            out["own_goal_x"][seen, team] = float(info["own_goal_x"])
            out["slot"][seen, team] = slot
    return out


def centred_nanmean(values, half):
    """float64 (N,): mean of the finite values over [t - half, t + half], NaN if there is none."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    csum = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    ccount = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    n = values.size
    lo = np.clip(np.arange(n) - int(half), 0, n)
    hi = np.clip(np.arange(n) + int(half) + 1, 0, n)
    count = ccount[hi] - ccount[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (csum[hi] - csum[lo]) / count
    return np.where(count > 0, out, np.nan)


def team_forward_velocity(tactical, blocks, pitch, cfg=None, fps=None, keepers=None):
    """Team forward velocities H/A in m/s, plus graded block IDs (C15, P10-A).

    For team T and frame t, the function takes the mean x velocity of the visible T players
    (position differences). It multiplies the mean by +1 when T defends x = 0 in the graded
    block of t and by -1 when T defends x = length (`own_goal_x` of goalkeeper_slots()). Then it
    takes a NaN-aware centred mean over +-SHAPE_HALF_FRAMES frames. The value is NaN outside the
    graded blocks and where fewer than SHAPE_MIN_PLAYERS players of T are visible, so the
    centroid is undefined there. tools/analysis/handover_confirm.py measured the shape response
    with the same definition before P10-A (P9-B), with a minimum of one player.

    `block_id` identifies each graded block and is -1 outside the blocks. The shape check uses
    it to reject a lag that crosses a block boundary (PLAN Phase 10, P10-A).
    `keepers` is an earlier goalkeeper_slots() result for the same `blocks`, or None.
    """
    cfg = cfg or Config.default()
    fps = _fps_or_default(fps if fps is not None else tactical.get("fps"))
    if keepers is None:
        keepers = goalkeeper_slots(tactical, blocks, pitch, cfg)
    px, py, visible = _player_metres(tactical, pitch)
    pvx, _ = _difference_velocity(px, py, visible, fps)
    team_of_slot = np.asarray(tactical["team_of_slot"])
    n = pvx.shape[0]
    block_id = np.full(n, -1, dtype=np.int64)
    for index, (_, first, last) in enumerate(blocks):
        block_id[max(0, int(first)):min(int(last) + 1, n)] = index
    out = {"block_id": block_id}
    for team in sorted(TEAM_KEYS):
        cols = np.flatnonzero(team_of_slot == team)
        sub = pvx[:, cols]
        finite = np.isfinite(sub)
        count = finite.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(finite, sub, 0.0).sum(axis=1) / count
        mean = np.where((count > 0) & (count >= int(cfg.SHAPE_MIN_PLAYERS)), mean, np.nan)
        sign = np.full(n, np.nan)
        for half, first, last in blocks:
            info = keepers.get((int(half), int(team)))
            if info is None:
                continue
            sign[int(first):min(int(last) + 1, n)] = (
                1.0 if float(info["own_goal_x"]) == 0.0 else -1.0)
        signed = mean * sign
        smoothed = centred_nanmean(signed, int(cfg.SHAPE_HALF_FRAMES))
        # P10-A: averaging must not restore a frame with an undefined centroid or block.
        out[TEAM_KEYS[team]] = np.where(np.isfinite(signed), smoothed, np.nan)
    return out


def frozen_mask(tactical, pitch, cfg=None):
    """bool (N, n_slots): the slot is frozen on the frame.

    A frozen slot is visible, has a NaN ATD speed, and moves less than FROZEN_MOVE_M between the
    two ends of a centred window of FROZEN_WIN frames. Both ends must be visible. The window is
    clipped at the ends of the array.
    """
    cfg = cfg or Config.default()
    px, py, visible = _player_metres(tactical, pitch)
    ps = np.asarray(tactical["ps"], dtype=np.float64)
    n_frames = px.shape[0]
    half = int(cfg.FROZEN_WIN) // 2
    index = np.arange(n_frames)
    before = np.clip(index - half, 0, max(n_frames - 1, 0))
    after = np.clip(index + half, 0, max(n_frames - 1, 0))
    move = np.hypot(px[after] - px[before], py[after] - py[before])
    with np.errstate(invalid="ignore"):
        still = np.isfinite(move) & (move < cfg.FROZEN_MOVE_M)
    return visible & np.isnan(ps) & still


def goalkeeper_swaps(tactical, blocks, pitch, cfg=None, keepers=None, frozen=None):
    """int64 (M, 3) rows (frame, live_slot, frozen_slot) where the guard fires, sorted.

    On frame f of a block, for team T with goalkeeper-like slot g and the other team with
    goalkeeper-like slot o, the guard fires when all of these hold:
      * g is visible and not frozen,
      * o is frozen on f,
      * g is within GK_SWAP_R of o,
      * g is more than half a pitch length from the own goal line of T.
    Then the body at g is the goalkeeper of the other team (RESEARCH-2 1d). The guard is off for
    a block where either team has no goalkeeper-like slot. `keepers` and `frozen` are the
    results of goalkeeper_slots() and frozen_mask(), computed here when None.
    """
    cfg = cfg or Config.default()
    length = float(pitch[0])
    px, py, visible = _player_metres(tactical, pitch)
    if keepers is None:
        keepers = goalkeeper_slots(tactical, blocks, pitch, cfg)
    if frozen is None:
        frozen = frozen_mask(tactical, pitch, cfg)
    found = []
    for half, first, last in blocks:
        frames = np.arange(int(first), int(last) + 1)
        for team in sorted(TEAM_KEYS):
            own = keepers.get((int(half), team))
            other = keepers.get((int(half), 1 - team))
            if own is None or other is None or own["slot"] < 0 or other["slot"] < 0:
                continue
            g, o = own["slot"], other["slot"]
            dist = np.hypot(px[frames, g] - px[frames, o], py[frames, g] - py[frames, o])
            with np.errstate(invalid="ignore"):
                near = dist <= cfg.GK_SWAP_R
                beyond = np.abs(px[frames, g] - own["own_goal_x"]) > length / 2.0
            fire = visible[frames, g] & ~frozen[frames, g] & frozen[frames, o] & near & beyond
            for frame in frames[fire]:
                found.append((int(frame), int(g), int(o)))
    found.sort()
    return np.asarray(found, dtype=np.int64).reshape(-1, 3)


def patch_goalkeeper_swaps(tactical, swaps):
    """Copy of `tactical` with each swap applied. The input dict is not changed.

    On each row (frame, live_slot, frozen_slot) the x, y and speed of the live slot move into
    the frozen slot, and the live slot becomes missing (NaN). The number of visible tracks of
    each team stays the same.
    """
    out = dict(tactical)
    px = np.array(tactical["px"], copy=True)
    py = np.array(tactical["py"], copy=True)
    ps = np.array(tactical["ps"], copy=True)
    swaps = np.asarray(swaps, dtype=np.int64).reshape(-1, 3)
    if swaps.size:
        frames, live, still = swaps[:, 0], swaps[:, 1], swaps[:, 2]
        px[frames, still] = px[frames, live]
        py[frames, still] = py[frames, live]
        ps[frames, still] = ps[frames, live]
        px[frames, live] = np.nan
        py[frames, live] = np.nan
        ps[frames, live] = np.nan
    out.update(px=px, py=py, ps=ps)
    return out


def speed_from_positions(tactical, pitch=None):
    """`tactical` with `ps` from position differences if the ATD has no player speed at all.

    An ATD XML without a speed channel gives a `ps` that is NaN everywhere (engine.data). The
    mean speeds of frame_features() and the frozen-slot test read `ps` only, so they would be
    NaN or True on every frame (G1 finding 7). This function then fills `ps` on visible frames
    with the speed of _difference_velocity(). If any `ps` value is finite, it returns `tactical`
    unchanged, so an ATD with a speed channel keeps its own values. Pitch defaults to games.py.
    """
    ps = np.asarray(tactical["ps"])
    if np.isfinite(ps).any():
        return tactical
    if pitch is None:
        pitch = data.pitch_m(tactical.get("game"))
    px, py, visible = _player_metres(tactical, pitch)
    vx, vy = _difference_velocity(px, py, visible, _fps_or_default(tactical.get("fps")))
    out = dict(tactical)
    out["ps"] = np.where(visible, np.hypot(vx, vy), np.nan).astype(np.float32)
    return out


def guard_goalkeeper_swaps(tactical, blocks, cfg=None, pitch=None):
    """(patched tactical, swaps, keepers): the goalkeeper slot-swap guard of C9.

    Run it on the ATD before prepare(), so the features, the slot confidence, the events and the
    nearest-player evidence all see the corrected team. Pitch defaults to games.py.
    """
    cfg = cfg or Config.default()
    if pitch is None:
        pitch = data.pitch_m(tactical.get("game"))
    keepers = goalkeeper_slots(tactical, blocks, pitch, cfg)
    frozen = frozen_mask(tactical, pitch, cfg)
    swaps = goalkeeper_swaps(tactical, blocks, pitch, cfg, keepers, frozen)
    return patch_goalkeeper_swaps(tactical, swaps), swaps, keepers


# -- convenience ------------------------------------------------------------------------------

def prepare(tactical, cfg=None, pitch=None):
    """(ball, feats, slot_conf) for a tactical dict; ball["airborne_suspect"] uses the full rule.

    Pitch defaults to games.py via engine.data.pitch_m(tactical["game"]).
    """
    cfg = cfg or Config.default()
    if pitch is None:
        pitch = data.pitch_m(tactical.get("game"))
    fps = tactical.get("fps")
    ball = clean_ball(tactical["bx"], tactical["by"], pitch, cfg, fps=fps)
    feats = frame_features(tactical, ball, pitch, cfg, fps=fps)
    ball["airborne_suspect"] = feats["airborne_suspect"].copy()
    return ball, feats, slot_confidence(tactical, cfg)

"""Touch, release, control, contested and dribble events of the ball (PLAN.md 1.1, 1.3, 3.3).

    detect_events(ball, feats, tactical, slot_conf, cfg, e_dead=None, shape=None)
        DataFrame[frame, kind, team, confidence, dist_m, notes], sorted by frame
    event_table(tactical, cfg=None)
        DataFrame[frame, kind, team, confidence], the hook of tools/forensics.py

The GT switches possession when a team plays the ball, not when the ball comes near a player
(PLAN 1.3). The events in this module mark the moments a player plays the ball. They gate team
switches in the orchestrator. They do not set the owner on every frame.

Two contact detectors give the touch, release and contested rows. CONTACT_DETECTOR picks one.

"bump" (the default, C8, RESEARCH-2 1g1). The ATD smooths the ball with a symmetric kernel of
about 20 frames (M13). A contact is a bump of the ball acceleration, and its centre is the
contact. The impulse is J(t) = v(t + h) - v(t - h), a vector, with h = CONTACT_HALF_WINDOW. J is
NaN unless both ends lie in one visible run.
  contact    a local maximum c of |J| over [c - h, c + h] with |J| >= DV_CONTACT, on an alive
             frame. The row is at c. For each team the function takes the smallest ball
             distance over [c - h, c + h]. The nearer team must be within R_TOUCH, else there
             is no row, except for a reach contact. The kind comes from
             ds = speed(c + h) - speed(c - h):
             ds >= REDIRECT_FRACTION x |J| is a kick, row kind `release`;
             ds <= -REDIRECT_FRACTION x |J| is a receive, row kind `control`, note `receive`;
             any other contact is a redirect.
  redirect   a redirect by the team that owns the ball is a `touch` with the note `redirect`.
             A redirect by the other team is a `control` with the note `redirect` only if the
             next contact is also by that team and no frame between them is dead (Metrica: a
             touch with no significant impact does not change possession), or, under
             REDIRECT_SHAPE_CONFIRM (C15), if the team shape confirms it: the forward velocity
             of the toucher's team rises against the other team's by at least
             SHAPE_CONFIRM_MS from c to c + SHAPE_LAG_FRAMES, with both defined and no dead
             frame in [c, c + SHAPE_LAG_FRAMES]. Otherwise the redirect has no row. The owner
             here is the team of the last kick, receive, confirmed redirect or control run. A
             dead frame resets it (_redirect_kinds). The shape check never changes a redirect
             by the owner, which stays a `touch`.
  contested  a contact where the other team's best distance is also within R_TOUCH and less
             than CONTESTED_MARGIN_M further. The team is None.
  reach      under REACH_SHAPE_CONFIRM (C18), a maximum with no player of either team within
             R_TOUCH is a contact of the nearer team when its best distance is within
             R_REACH_M and the shape response of that team at c confirms it (shape_confirms,
             the C15 SHAPE_LAG_FRAMES and SHAPE_CONFIRM_MS). Otherwise there is no row. The
             kind comes from ds as for any contact (kick, receive or redirect), and a redirect
             then follows the redirect rule above. Its dist_m is above R_TOUCH, and no other
             row of the table has that, so dist_m marks a reach row. Its confidence is at the
             distance floor. A maximum with a player within R_TOUCH never becomes a reach
             contact. Measured on BAR-ATH before the rule (calibration/P9B_wider_radius.md 4b):
             no reach candidate of the team without the ball lies in the public window.
  kink point under CONTACT_POINT "kink_min" (K1, calibration/K1_contact_point.md). The
             smoothed ball cuts the corner where the ball changed velocity, so it can pass
             metres from the player who played it and close to a player who did not. A kink
             model fits the true path around c: straight, with one velocity change at tau,
             seen through a Gaussian kernel (kink_points). For each team the distance of the
             rules above is the smaller of its window distance and its distance to the fitted
             contact point at tau (kink_team_distances). The frame of the row stays c.

"dv3" (P1-A, the rows of `--mode m2`).
  touch      a local maximum t of `dv_k` above DV_TOUCH with a player within R_TOUCH on a frame
             of [c, t + k]. The contact c is the start of the monotone speed change that holds
             t (see contact_frame). The team is the team of the nearest player.
  release    a touch after which the speed rises by more than DV_RELEASE within RELEASE_K
             frames of the contact (a kick).
  contested  a touch where the nearest H and the nearest A are both within R_TOUCH and their
             distances differ by less than CONTESTED_MARGIN_M. The team is None.

Both detectors:
  control    the ball within R_CONTROL of one team's nearest player, slower than
             CONTROL_MAX_SPEED, uncontested, for at least CONTROL_FRAMES frames. The row is at
             the first frame of the run. The note `end=<frame>` gives the last frame.
  dribble    the same player slot is the nearest player and within R_DRIBBLE of the visible
             ball on at least DRIBBLE_FRAMES consecutive alive frames. Over the run the median of
             |v_ball - v_player| is below DRIBBLE_REL_SPEED and the median ball speed is above
             CONTROL_MAX_SPEED, so a dribble is never a control (RESEARCH.md 3 M10). The row is
             at the first frame of the run. The note `end=<frame>` gives the last frame. If the
             nearest opponent at the first frame is within R_TOUCH and less than
             CONTESTED_MARGIN_M further away, the team is None and the note starts with
             `contested`.

Rules:
  * The frame of a touch is its contact c. This module adds no timing offset. Phase 2 adds
    the offsets.
  * "dv3" only: a touch while `airborne_suspect` is True at t has no row. The ground
    projection of a ball in the air is metres away from the players who play it (PLAN 1.1,
    11351). The airborne rule does not apply to a dribble or to a bump contact. A ball within
    1 m of one player for 1 s is on the ground.
  * "dv3" only: a touch with a heading change below DEFLECTION_MAX_HEADING_DEG and a relative
    speed change below DEFLECTION_MAX_SPEED_FRAC has the note `deflection` and half its
    confidence. A NaN heading (ball below HEADING_MIN_SPEED) is not a deflection. Under
    "bump" the redirect rule takes the place of the deflection rule.
  * No event starts on a frame where the dead-ball evidence is above EVENT_SUPPRESS_DEAD
    (17380: a player plays the ball again inside `D`).
  * Confidence is EVENT_CONF_MAX x slot confidence x a distance factor x a margin factor. Each
    factor falls linearly from 1 to EVENT_CONF_FACTOR_FLOOR. A single frame never gives 1.0.

numpy and pandas only. Deterministic. The module never reads the ground truth.
"""

import math

import numpy as np
import pandas as pd

from engine import data, deadball
from engine.config import Config
from engine.tracks import _runs

COLUMNS = ["frame", "kind", "team", "confidence", "dist_m", "notes"]
TABLE_COLUMNS = ["frame", "kind", "team", "confidence"]
KINDS = ("touch", "release", "contested", "control", "dribble")
TEAM_LETTER = {data.HOME: "H", data.AWAY: "A"}
DEFLECTION_NOTE = "deflection"
CONTESTED_NOTE = "contested"
RECEIVE_NOTE = "receive"
REDIRECT_NOTE = "redirect"
DETECTORS = ("bump", "dv3")
CONTACT_POINTS = ("window", "kink_min")

_TABLE_CACHE = {"tactical": None, "cfg": None, "table": None}


# -- small helpers ------------------------------------------------------------------------------

def _finite_or_inf(values):
    """float64 copy of `values` with NaN replaced by +inf (a missing player is far away)."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values), values, np.inf)


def _local_maxima(values):
    """bool (N,): finite local maxima of `values`.

    A frame is a maximum when its value is at least the previous value and above the next one.
    On a plateau the last frame of the plateau is the maximum. NaN neighbours count as lower.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    filled = np.where(finite, values, -np.inf)
    prev = np.concatenate(([-np.inf], filled[:-1]))
    nxt = np.concatenate((filled[1:], [-np.inf]))
    return finite & (filled >= prev) & (filled > nxt)


def _factor(value, full_at, floor_at, floor):
    """Linear factor: 1 at `full_at`, `floor` at `floor_at`, clipped to [floor, 1]."""
    if not np.isfinite(value):
        return 1.0 if full_at > floor_at else floor
    span = float(floor_at - full_at)
    position = (value - full_at) / span if span != 0 else 1.0
    return float(np.clip(1.0 - (1.0 - floor) * position, floor, 1.0))


def confidence(d1, margin, slot_conf, cfg):
    """Event confidence from the nearest distance, the margin to the other team and the slot.

    The distance factor is 1 at 0 m and EVENT_CONF_FACTOR_FLOOR at R_TOUCH. The margin factor is
    EVENT_CONF_FACTOR_FLOOR at CONTESTED_MARGIN_M and 1 at R_TOUCH or more. An infinite margin
    (no opponent visible) gives 1.
    """
    floor = cfg.EVENT_CONF_FACTOR_FLOOR
    dist_factor = _factor(d1, 0.0, cfg.R_TOUCH, floor)
    margin_factor = _factor(margin, cfg.R_TOUCH, cfg.CONTESTED_MARGIN_M, floor)
    return float(cfg.EVENT_CONF_MAX * float(slot_conf) * dist_factor * margin_factor)


def _slot_conf_at(feats, slot_conf, team, frame):
    """Slot confidence of the nearest player of `team` at `frame`, 1.0 without a table."""
    if slot_conf is None:
        return 1.0
    slot = int(feats["nearest_slot_" + team][frame]) if ("nearest_slot_" + team) in feats else -1
    if slot < 0:
        return 1.0
    return float(slot_conf[frame, slot])


def _empty():
    """An empty event table with the right columns and types."""
    table = pd.DataFrame({"frame": np.zeros(0, dtype=np.int64),
                          "kind": np.zeros(0, dtype=object), "team": np.zeros(0, dtype=object),
                          "confidence": np.zeros(0, dtype=np.float64),
                          "dist_m": np.zeros(0, dtype=np.float64),
                          "notes": np.zeros(0, dtype=object)})
    return table[COLUMNS]


# -- touches ------------------------------------------------------------------------------------

def contact_frame(speed, run_id, peak, cfg):
    """Contact frame of the touch whose dv_k peak is at `peak`.

    The function walks back from `peak` while the one-frame speed change has the sign of the
    change at the peak and a size of at least DV_TOUCH / DV_K, inside the same visible run, for
    at most TOUCH_ONSET_MAX_FRAMES frames. A smooth ATD track spreads a kick over several frames,
    so the foot is at the ball where the speed starts to change (config provenance).
    """
    k = int(cfg.DV_K)
    sign = np.sign(speed[min(peak + k, speed.shape[0] - 1)] - speed[peak])
    if sign == 0 or not np.isfinite(sign):
        return peak
    least = cfg.DV_TOUCH / float(k)
    frame = peak
    while (peak - frame < cfg.TOUCH_ONSET_MAX_FRAMES and frame - 1 >= 0
           and run_id[frame - 1] == run_id[peak] and run_id[peak] >= 0):
        step = speed[frame] - speed[frame - 1]
        if not (np.isfinite(step) and sign * step >= least):
            break
        frame -= 1
    return frame


def _rise_after(speed, run_id, frame, cfg):
    """Largest speed rise from `frame` within RELEASE_K frames, in the same visible run."""
    n = speed.shape[0]
    hi = min(n, frame + int(cfg.RELEASE_K) + 1)
    same = run_id[frame:hi] == run_id[frame]
    later = speed[frame:hi][same]
    if later.size < 2 or not np.isfinite(later).any():
        return np.nan
    return float(np.nanmax(later[1:]) - speed[frame]) if np.isfinite(later[1:]).any() else np.nan


def _touch_rows(ball, feats, slot_conf, alive, cfg):
    """Rows for touch, release and contested events (see the module docstring)."""
    k = int(ball.get("k", cfg.DV_K))
    dv = np.asarray(ball["dv_k"], dtype=np.float64)
    dv_signed = np.asarray(ball["dv_signed_k"], dtype=np.float64)
    speed = np.asarray(ball["speed"], dtype=np.float64)
    run_id = np.asarray(ball["run_id"], dtype=np.int64)
    heading = np.asarray(ball["heading_change_k"], dtype=np.float64)
    airborne = np.asarray(ball["airborne_suspect"], dtype=bool)
    d_h, d_a = _finite_or_inf(feats["dH"]), _finite_or_inf(feats["dA"])
    n = dv.shape[0]
    with np.errstate(invalid="ignore"):
        peaks = _local_maxima(dv) & (dv > cfg.DV_TOUCH)
    rows = []
    seen = set()
    for t in np.flatnonzero(peaks):
        t = int(t)
        c = contact_frame(speed, run_id, t, cfg)
        if not alive[c] or airborne[c] or c in seen:
            continue
        hi = min(n, t + k + 1)
        at_h = c + int(np.argmin(d_h[c:hi]))
        at_a = c + int(np.argmin(d_a[c:hi]))
        dist = {"H": d_h[at_h], "A": d_a[at_a]}
        team = "H" if dist["H"] <= dist["A"] else "A"
        other = "A" if team == "H" else "H"
        d1, d2 = dist[team], dist[other]
        if not d1 <= cfg.R_TOUCH:
            continue
        seen.add(c)
        margin = d2 - d1
        if d2 <= cfg.R_TOUCH and margin < cfg.CONTESTED_MARGIN_M:
            conf = confidence(d1, np.inf, 1.0, cfg)  # a touch happened, its team is unknown
            rows.append((c, "contested", None, conf, float(d1), ""))
            continue
        frame_of_team = at_h if team == "H" else at_a
        conf = confidence(d1, margin, _slot_conf_at(feats, slot_conf, team, frame_of_team), cfg)
        rise = _rise_after(speed, run_id, c, cfg)
        kind = "release" if np.isfinite(rise) and rise > cfg.DV_RELEASE else "touch"
        notes = []
        with np.errstate(invalid="ignore", divide="ignore"):
            rel_change = abs(dv_signed[t]) / speed[t] if speed[t] > 0 else np.inf
        if (np.isfinite(heading[t]) and heading[t] < cfg.DEFLECTION_MAX_HEADING_DEG
                and rel_change < cfg.DEFLECTION_MAX_SPEED_FRAC):
            notes.append(DEFLECTION_NOTE)
            conf *= cfg.DEFLECTION_CONF_FACTOR
        rows.append((c, kind, team, conf, float(d1), ";".join(notes)))
    return rows


# -- bump-centre contacts (C8) ------------------------------------------------------------------

def impulse(ball, h):
    """float64 (N,): |v(t + h) - v(t - h)|, NaN unless t - h and t + h lie in one visible run.

    A visible run is contiguous. Both ends in one run means that every frame between is in it.
    """
    vx = np.asarray(ball["vx"], dtype=np.float64)
    vy = np.asarray(ball["vy"], dtype=np.float64)
    run_id = np.asarray(ball["run_id"], dtype=np.int64)
    n = vx.shape[0]
    h = int(h)
    out = np.full(n, np.nan)
    if h < 1 or n <= 2 * h:
        return out
    same = (run_id[2 * h:] == run_id[:-2 * h]) & (run_id[:-2 * h] >= 0)
    jx = vx[2 * h:] - vx[:-2 * h]
    jy = vy[2 * h:] - vy[:-2 * h]
    out[h:n - h] = np.where(same, np.hypot(jx, jy), np.nan)
    return out


def window_maxima(values, radius, frames):
    """int64 array: the frames of `frames` that hold the maximum of `values` over +-radius.

    On a plateau the first frame of the window maximum wins. NaN is lower than any value.
    """
    values = np.asarray(values, dtype=np.float64)
    filled = np.where(np.isfinite(values), values, -np.inf)
    n = filled.shape[0]
    keep = []
    for c in np.asarray(frames, dtype=np.int64):
        c = int(c)
        lo, hi = max(0, c - int(radius)), min(n, c + int(radius) + 1)
        if lo + int(np.argmax(filled[lo:hi])) == c:
            keep.append(c)
    return np.asarray(keep, dtype=np.int64)


def gauss_ramp(u, sigma):
    """float64: G(u) = u Phi(u / sigma) + sigma phi(u / sigma), the kink of the kink model.

    G is the mean of max(u - s, 0) over s ~ N(0, sigma^2): a velocity step of 1 at u = 0 seen
    through a Gaussian kernel of width sigma, integrated to a position. G(u) tends to max(u, 0)
    as sigma tends to 0. `u` and `sigma` are in frames. math.erf keeps it exact and
    deterministic. The grids of kink_points() are small, so the Python loop costs little.
    """
    u = np.asarray(u, dtype=np.float64)
    z = u / float(sigma)
    cdf = 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in z.ravel()],
                                dtype=np.float64).reshape(z.shape))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return u * cdf + float(sigma) * pdf


def kink_points(ball, centres, cfg):
    """(x_c, y_c, tau) float64 arrays: the kink-model contact point of each impulse maximum.

    `centres` are the impulse maxima in frame order (window_maxima). The fit of centre c uses
    the visible frames of c's visible run in [c - KINK_HALF_WINDOW, c + KINK_HALF_WINDOW],
    clipped halfway to the previous and the next centre. It needs KINK_MIN_POINTS frames and
    more than KINK_SIDE_FRAMES frames on each side of c, else the three values are NaN. For
    each sigma of KINK_SIGMAS (outer loop) and each tau offset of the KINK_TAU_HALF and
    KINK_TAU_STEP grid (inner loop), the function solves the least squares of the model
    x = x_c + v1 u + dv G(u), u = t - tau, on each axis, and keeps the first (sigma, tau) with
    the smallest residual over both axes. x_c, y_c is the model ball at tau (m). See
    CONTACT_POINT in engine/config.py for the physics. Deterministic, numpy and math only.
    """
    centres = np.asarray(centres, dtype=np.int64)
    m = centres.shape[0]
    nan = np.full(m, np.nan)
    if m == 0:
        return nan, nan.copy(), nan.copy()
    x = np.asarray(ball["x"], dtype=np.float64)
    y = np.asarray(ball["y"], dtype=np.float64)
    run_id = np.asarray(ball["run_id"], dtype=np.int64)
    n = x.shape[0]
    half = int(cfg.KINK_HALF_WINDOW)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    width = offsets.shape[0]
    mask = np.zeros((m, width), dtype=bool)
    xs = np.zeros((m, width))
    ys = np.zeros((m, width))
    # Each fit works in coordinates relative to the ball at c (origin_x, origin_y), so the sums
    # of the normal equations stay small and the residual does not cancel on pitch coordinates.
    origin_x = np.zeros(m)
    origin_y = np.zeros(m)
    for i in range(m):
        c = int(centres[i])
        left, right = c - half, c + half
        if i > 0:
            left = max(left, (int(centres[i - 1]) + c) // 2)
        if i + 1 < m:
            right = min(right, (int(centres[i + 1]) + c) // 2)
        left, right = max(0, left), min(n - 1, right)
        if run_id[c] < 0 or right < left:
            continue
        idx = np.arange(left, right + 1)
        idx = idx[run_id[idx] == run_id[c]]
        side = int(cfg.KINK_SIDE_FRAMES)
        if idx.size < int(cfg.KINK_MIN_POINTS) or idx.min() >= c - side or idx.max() <= c + side:
            continue
        cols = idx - c + half
        mask[i, cols] = True
        origin_x[i], origin_y[i] = x[c], y[c]
        xs[i, cols] = x[idx] - x[c]
        ys[i, cols] = y[idx] - y[c]
    fitted = mask.any(axis=1)
    weight = mask.astype(np.float64)
    wx, wy = weight * xs, weight * ys
    count = weight.sum(axis=1)
    sum_x, sum_y = wx.sum(axis=1), wy.sum(axis=1)
    best = np.full(m, np.inf)
    best_x, best_y, best_tau = nan.copy(), nan.copy(), nan.copy()
    steps = int(round(float(cfg.KINK_TAU_HALF) / float(cfg.KINK_TAU_STEP)))
    shifts = [k * float(cfg.KINK_TAU_STEP) for k in range(-steps, steps + 1)]
    for sigma in cfg.KINK_SIGMAS:
        for shift in shifts:
            u = offsets - shift
            # Rounded to 1e-12 frames, so a last-bit difference of erf or exp between platform
            # libraries almost never reaches the fit (RULES.md: byte reproduction on any CPU).
            g = np.round(gauss_ramp(u, sigma), 12)
            # Normal equations of the basis (1, u, g), solved in closed form with elementwise
            # numpy only (no BLAS or LAPACK call). With g rounded above, a platform difference
            # can reach the sums only when a value lies on a rounding boundary.
            a00, a01, a02 = count, (weight * u).sum(axis=1), (weight * g).sum(axis=1)
            a11, a12, a22 = ((weight * (u * u)).sum(axis=1), (weight * (u * g)).sum(axis=1),
                             (weight * (g * g)).sum(axis=1))
            c00 = a11 * a22 - a12 * a12
            c01 = a02 * a12 - a01 * a22
            c02 = a01 * a12 - a02 * a11
            c11 = a00 * a22 - a02 * a02
            c12 = a01 * a02 - a00 * a12
            c22 = a00 * a11 - a01 * a01
            det = a00 * c00 + a01 * c01 + a02 * c02
            solvable = fitted & (np.abs(det) > 1e-9 * np.maximum(1.0, np.abs(a00 * a11 * a22)))
            safe = np.where(solvable, det, 1.0)
            resid = np.zeros(m)
            heads = []
            for vals, wv, total in ((xs, wx, sum_x), (ys, wy, sum_y)):
                r0, r1, r2 = total, (wv * u).sum(axis=1), (wv * g).sum(axis=1)
                b0 = (c00 * r0 + c01 * r1 + c02 * r2) / safe
                b1 = (c01 * r0 + c11 * r1 + c12 * r2) / safe
                b2 = (c02 * r0 + c12 * r1 + c22 * r2) / safe
                # The residual is summed directly over the frames (not as sum x^2 minus b.r), so
                # it is never negative and keeps its precision on an ill-conditioned fit.
                fit = b0[:, None] + b1[:, None] * u[None, :] + b2[:, None] * g[None, :]
                resid = resid + (weight * (vals - fit) ** 2).sum(axis=1)
                heads.append(b0)
            # A residual is a sum of squares in m^2. Below 1e-9 m^2 a difference means nothing,
            # and rounding keeps a near tie from turning on the last bits of the arithmetic.
            resid = np.round(resid, 9)
            better = solvable & (resid < best)
            best = np.where(better, resid, best)
            best_x = np.where(better, heads[0], best_x)
            best_y = np.where(better, heads[1], best_y)
            best_tau = np.where(better, centres + shift, best_tau)
    return np.round(best_x + origin_x, 6), np.round(best_y + origin_y, 6), best_tau


def kink_team_distances(points, players, cfg):
    """{"H": (M,), "A": (M,)} nearest distance of each team to each kink contact point, m.

    `points` is the (x_c, y_c, tau) of kink_points(). `players` is (px, py, visible,
    team_of_slot) in metres. The distance is the smallest over the visible players of the team
    on the frames round(tau) +- KINK_PLAYER_HALF. It is inf where the point is NaN or no player
    of the team is visible there.
    """
    xc, yc, tau = (np.asarray(v, dtype=np.float64) for v in points)
    px, py, visible, team_of_slot = players
    n = px.shape[0]
    team_of_slot = np.asarray(team_of_slot)
    out = {}
    reach = int(cfg.KINK_PLAYER_HALF)
    for team, code in (("H", data.HOME), ("A", data.AWAY)):
        cols = np.flatnonzero(team_of_slot == code)
        dist = np.full(xc.shape[0], np.inf)
        for i in np.flatnonzero(np.isfinite(xc) & np.isfinite(tau)):
            centre = int(np.round(tau[i]))
            lo, hi = max(0, centre - reach), min(n, centre + reach + 1)
            if hi <= lo or cols.size == 0:
                continue
            seen = visible[lo:hi][:, cols]
            if not seen.any():
                continue
            d = np.hypot(px[lo:hi][:, cols] - xc[i], py[lo:hi][:, cols] - yc[i])
            dist[i] = float(np.min(d[seen]))
        out[team] = dist
    return out


def _bump_contacts(ball, feats, slot_conf, alive, cfg, shape=None, players=None):
    """[(frame, kind, team, confidence, dist_m)] of the bump contacts.

    The kind is kick, receive, redirect or contested (see the module docstring). Without
    `shape`, a maximum with no player within R_TOUCH has no row. With `shape` (the team forward
    velocity of tracks.team_forward_velocity) such a maximum is a reach contact (C18) when the
    nearest player is within R_REACH_M and shape_confirms() holds for that player's team at
    the maximum. A reach contact has dist_m above R_TOUCH. The caller passes `shape` only under
    REACH_SHAPE_CONFIRM. With CONTACT_POINT "kink_min" and `players` (px, py, visible,
    team_of_slot in metres), the distance of each team is the smaller of its window distance
    and its distance to the kink contact point (kink_points, kink_team_distances).
    """
    h = int(cfg.CONTACT_HALF_WINDOW)
    size = impulse(ball, h)
    speed = np.asarray(ball["speed"], dtype=np.float64)
    d_h, d_a = _finite_or_inf(feats["dH"]), _finite_or_inf(feats["dA"])
    n = size.shape[0]
    alive = np.asarray(alive, dtype=bool)
    with np.errstate(invalid="ignore"):
        strong = np.isfinite(size) & (size >= cfg.DV_CONTACT) & alive
    dead = np.concatenate(([0], np.cumsum(~alive))) if shape is not None else None
    centres = window_maxima(size, h, np.flatnonzero(strong))
    kink = None
    if cfg.CONTACT_POINT == "kink_min" and players is not None:
        kink = kink_team_distances(kink_points(ball, centres, cfg), players, cfg)
    elif cfg.CONTACT_POINT not in CONTACT_POINTS:
        raise ValueError("CONTACT_POINT must be one of {}".format(CONTACT_POINTS))
    out = []
    for index, c in enumerate(centres):
        c = int(c)
        lo, hi = max(0, c - h), min(n, c + h + 1)
        at_h = lo + int(np.argmin(d_h[lo:hi]))
        at_a = lo + int(np.argmin(d_a[lo:hi]))
        dist = {"H": d_h[at_h], "A": d_a[at_a]}
        if kink is not None:
            dist = {"H": min(dist["H"], kink["H"][index]), "A": min(dist["A"], kink["A"][index])}
        team = "H" if dist["H"] <= dist["A"] else "A"
        other = "A" if team == "H" else "H"
        d1, d2 = dist[team], dist[other]
        if not d1 <= cfg.R_TOUCH:
            # C18: nobody within R_TOUCH. A reach contact needs the nearest player within
            # R_REACH_M and the shape response of that player's team.
            if dead is None or not d1 <= cfg.R_REACH_M:
                continue
            if not shape_confirms(shape, c, team, dead, cfg):
                continue
        margin = d2 - d1
        if d2 <= cfg.R_TOUCH and margin < cfg.CONTESTED_MARGIN_M:
            conf = confidence(d1, np.inf, 1.0, cfg)  # a contact happened, its team is unknown
            out.append((c, "contested", None, conf, float(d1)))
            continue
        at = at_h if team == "H" else at_a
        conf = confidence(d1, margin, _slot_conf_at(feats, slot_conf, team, at), cfg)
        change = speed[c + h] - speed[c - h]
        split = cfg.REDIRECT_FRACTION * size[c]
        kind = "kick" if change >= split else ("receive" if change <= -split else "redirect")
        out.append((c, kind, team, conf, float(d1)))
    return out


def shape_response(shape, frame, team, lag):
    """R_team - R_other at `frame`, m/s, with R_T = vf_T(frame + lag) - vf_T(frame) (C15).

    `shape` is tracks.team_forward_velocity(). NaN when a value is undefined or frame + lag is
    outside the arrays.
    """
    other = "A" if team == "H" else "H"
    n = shape[team].shape[0]
    end = int(frame) + int(lag)
    if not (0 <= int(frame) < n and 0 <= end < n):
        return np.nan
    own = shape[team][end] - shape[team][int(frame)]
    theirs = shape[other][end] - shape[other][int(frame)]
    return float(own - theirs)


def shape_confirms(shape, frame, team, dead_cum, cfg):
    """True when the C15 shape response confirms a redirect of `team` at `frame`.

    `dead_cum` is the cumulative count of dead frames with a leading 0 (dead_cum[t] = dead
    frames before t). The response must be defined, at least SHAPE_CONFIRM_MS, and no frame of
    [frame, frame + SHAPE_LAG_FRAMES] may be dead. Both ends must belong to the same graded block.
    Shapes without block metadata describe one synthetic block. `shape` None never confirms.
    """
    if shape is None:
        return False
    end = int(frame) + int(cfg.SHAPE_LAG_FRAMES)
    if not (0 <= int(frame) <= end < dead_cum.shape[0] - 1):
        return False
    block_id = shape.get("block_id")
    if block_id is not None and (end >= len(block_id) or block_id[int(frame)] < 0
                                 or block_id[int(frame)] != block_id[end]):
        return False
    if dead_cum[end + 1] - dead_cum[int(frame)] != 0:
        return False
    value = shape_response(shape, frame, team, cfg.SHAPE_LAG_FRAMES)
    return bool(np.isfinite(value) and value >= cfg.SHAPE_CONFIRM_MS)


def _redirect_kinds(contacts, control_rows, alive, shape=None, cfg=None):
    """{contact frame: "touch" or "control"} for the redirects of `contacts` that keep a row.

    The stream is the uncontested contacts and the control runs in frame order. On one frame
    a contact comes before a control run. The owner is the team of the last kick, receive,
    confirmed redirect or control run. A dead frame between two stream items resets the owner
    to None. A redirect by the owner is a touch. A redirect by another team is a control if
    the next stream item is by the same team with no dead frame between. With `shape` (the
    team forward velocity of tracks.team_forward_velocity) and `cfg`, it is also a control if
    shape_confirms() holds (C15). Else it has no row. `shape` None keeps the C8 rule.
    """
    dead = np.concatenate(([0], np.cumsum(~np.asarray(alive, dtype=bool))))

    def clean(a, b):
        return dead[b + 1] - dead[a] == 0

    stream = [(c[0], 0, c[2], c[1]) for c in contacts if c[1] != "contested"]
    stream += [(r[0], 1, r[2], "control") for r in control_rows]
    stream.sort(key=lambda item: (item[0], item[1]))
    kinds = {}
    owner, last = None, None
    for i, (frame, _, team, kind) in enumerate(stream):
        if last is not None and not clean(last, frame):
            owner = None
        last = frame
        if kind != "redirect":
            owner = team
            continue
        if owner == team:
            kinds[frame] = "touch"
            continue
        if i + 1 < len(stream) and stream[i + 1][2] == team and clean(frame, stream[i + 1][0]):
            kinds[frame] = "control"
            owner = team
        elif shape_confirms(shape, frame, team, dead, cfg):
            kinds[frame] = "control"
            owner = team
    return kinds


def _contact_rows(ball, feats, slot_conf, alive, cfg, control_rows=None, shape=None,
                  players=None):
    """Rows for the release, control, touch and contested events of the bump detector.

    `control_rows` are the _control_rows() of the same scene. They take part in the owner and
    in the redirect confirmation. The function does not return them. `shape` is the team
    forward velocity of the C15 and C18 shape confirmations, or None (no shape confirmation).
    REDIRECT_SHAPE_CONFIRM and REACH_SHAPE_CONFIRM switch each use of `shape` on or off. A
    reach contact (C18) gets the row of its kind, as any other contact. `players` is the
    (px, py, visible, team_of_slot) of the CONTACT_POINT "kink_min" rule, or None.
    """
    if control_rows is None:
        control_rows = _control_rows(ball, feats, slot_conf, alive, cfg)
    reach_shape = shape if cfg.REACH_SHAPE_CONFIRM else None
    contacts = _bump_contacts(ball, feats, slot_conf, alive, cfg, shape=reach_shape,
                              players=players)
    if not cfg.REDIRECT_SHAPE_CONFIRM:
        shape = None
    redirects = _redirect_kinds(contacts, control_rows, alive, shape, cfg)
    rows = []
    for frame, kind, team, conf, dist in contacts:
        if kind == "contested":
            rows.append((frame, "contested", None, conf, dist, ""))
        elif kind == "kick":
            rows.append((frame, "release", team, conf, dist, ""))
        elif kind == "receive":
            rows.append((frame, "control", team, conf, dist, RECEIVE_NOTE))
        elif frame in redirects:
            rows.append((frame, redirects[frame], team, conf, dist, REDIRECT_NOTE))
    return rows


# -- control ------------------------------------------------------------------------------------

def _control_rows(ball, feats, slot_conf, alive, cfg):
    """Rows for control events (see the module docstring)."""
    speed = np.asarray(ball["speed"], dtype=np.float64)
    d_h, d_a = _finite_or_inf(feats["dH"]), _finite_or_inf(feats["dA"])
    d1 = np.minimum(d_h, d_a)
    d2 = np.maximum(d_h, d_a)
    with np.errstate(invalid="ignore"):
        slow = np.isfinite(speed) & (speed < cfg.CONTROL_MAX_SPEED)
    with np.errstate(invalid="ignore"):  # inf - inf when neither team is visible
        contested = (d2 <= cfg.R_TOUCH) & ((d2 - d1) < cfg.CONTESTED_MARGIN_M)
    ok = np.asarray(ball["visible"], dtype=bool) & slow & (d1 <= cfg.R_CONTROL) & ~contested
    ok &= alive
    team_code = np.where(d_h <= d_a, data.HOME, data.AWAY)
    key = np.where(ok, team_code, -1).astype(np.int64)
    rows = []
    for code in sorted(TEAM_LETTER):
        starts, stops = _runs(key == code)
        letter = TEAM_LETTER[code]
        other = d_a if letter == "H" else d_h
        own = d_h if letter == "H" else d_a
        for start, stop in zip(starts, stops):
            start, stop = int(start), int(stop)
            if stop - start < cfg.CONTROL_FRAMES:
                continue
            margin = other[start] - own[start]
            conf = confidence(own[start], margin,
                              _slot_conf_at(feats, slot_conf, letter, start), cfg)
            rows.append((start, "control", letter, conf, float(own[start]),
                         "end={}".format(stop - 1)))
    return rows


# -- dribble ------------------------------------------------------------------------------------

def _constant_runs(key):
    """(starts, stops) of the maximal runs of equal values >= 0 in an int array (half-open)."""
    key = np.asarray(key, dtype=np.int64)
    if key.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    breaks = np.flatnonzero(key[1:] != key[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [key.size]))
    keep = key[starts] >= 0
    return starts[keep], stops[keep]


def _dribble_rows(ball, feats, slot_conf, alive, cfg):
    """Rows for dribble events (see the module docstring).

    The nearest player of either team defines the slot of each frame. A run is a maximal
    stretch of frames with the same slot within R_DRIBBLE of the visible ball, on alive frames.
    """
    speed = np.asarray(ball["speed"], dtype=np.float64)
    d_h, d_a = _finite_or_inf(feats["dH"]), _finite_or_inf(feats["dA"])
    home = d_h <= d_a
    d1 = np.where(home, d_h, d_a)
    slot = np.where(home, np.asarray(feats["nearest_slot_H"], dtype=np.int64),
                    np.asarray(feats["nearest_slot_A"], dtype=np.int64))
    rel = np.where(home, np.asarray(feats["relH"], dtype=np.float64),
                   np.asarray(feats["relA"], dtype=np.float64))
    ok = np.asarray(ball["visible"], dtype=bool) & (d1 <= cfg.R_DRIBBLE) & (slot >= 0) & alive
    rows = []
    for start, stop in zip(*_constant_runs(np.where(ok, slot, -1))):
        start, stop = int(start), int(stop)
        if stop - start < cfg.DRIBBLE_FRAMES:
            continue
        rel_run, speed_run = rel[start:stop], speed[start:stop]
        if not (np.isfinite(rel_run).any() and np.isfinite(speed_run).any()):
            continue
        if not (np.nanmedian(rel_run) < cfg.DRIBBLE_REL_SPEED
                and np.nanmedian(speed_run) > cfg.CONTROL_MAX_SPEED):
            continue
        letter = "H" if home[start] else "A"
        own = d_h[start] if letter == "H" else d_a[start]
        other = d_a[start] if letter == "H" else d_h[start]
        margin = other - own
        end_note = "end={}".format(stop - 1)
        if other <= cfg.R_TOUCH and margin < cfg.CONTESTED_MARGIN_M:
            conf = confidence(own, np.inf, 1.0, cfg)    # a dribble, its team is unknown
            rows.append((start, "dribble", None, conf, float(own),
                         ";".join((CONTESTED_NOTE, end_note))))
            continue
        conf = confidence(own, margin, _slot_conf_at(feats, slot_conf, letter, start), cfg)
        rows.append((start, "dribble", letter, conf, float(own), end_note))
    return rows


# -- public API ---------------------------------------------------------------------------------

def detect_events(ball, feats, tactical=None, slot_conf=None, cfg=None, e_dead=None,
                  shape=None, players=None):
    """Event table of the ball: DataFrame[frame, kind, team, confidence, dist_m, notes].

    `ball`, `feats` and `slot_conf` are the outputs of engine.tracks.prepare(). `e_dead` is the
    per-frame dead-ball evidence of engine.deadball.dead_evidence(). If `e_dead` is None and
    `tactical` is given, this function computes it from the match files of `tactical["game"]`.
    If both are None, no frame is suppressed. `shape` is the team forward velocity of
    engine.tracks.team_forward_velocity() for the C15 and C18 shape confirmations. If it is None
    and `tactical` is given, this function computes it with the graded blocks of the game. If
    both are None, no redirect is confirmed by shape and there is no reach contact. Rows are
    sorted by frame, then by kind in the order of KINDS. `team` is "H", "A" or None. `notes` is "" or "deflection" for touches and
    releases of "dv3", "receive" or "redirect" for the bump rows that are not kicks,
    "end=<frame>" for a control run or a dribble, and "contested;end=<frame>" for a contested
    dribble. A reach contact (C18) is the only row with dist_m above R_TOUCH. `players` is the
    (px, py, visible, team_of_slot) in metres of the CONTACT_POINT "kink_min" rule. If it is
    None and `tactical` is given, this function computes it (match_players). If both are None,
    the window rule decides every contact.
    """
    cfg = cfg or Config.default()
    n = np.asarray(ball["visible"]).shape[0]
    if e_dead is None and tactical is not None:
        e_dead = match_dead_evidence(ball, feats, tactical, cfg)
    if e_dead is None:
        alive = np.ones(n, dtype=bool)
    else:
        e_dead = np.asarray(e_dead, dtype=np.float64)
        alive = ~(np.isfinite(e_dead) & (e_dead > cfg.EVENT_SUPPRESS_DEAD))
    if cfg.CONTACT_DETECTOR not in DETECTORS:
        raise ValueError("CONTACT_DETECTOR must be one of {}".format(DETECTORS))
    controls = _control_rows(ball, feats, slot_conf, alive, cfg)
    if cfg.CONTACT_DETECTOR == "bump":
        wants_shape = cfg.REDIRECT_SHAPE_CONFIRM or cfg.REACH_SHAPE_CONFIRM
        if wants_shape and shape is None and tactical is not None:
            shape = match_team_shape(tactical, cfg)
        if cfg.CONTACT_POINT == "kink_min" and players is None and tactical is not None:
            players = match_players(tactical)
        rows = _contact_rows(ball, feats, slot_conf, alive, cfg, controls, shape, players)
    else:
        rows = _touch_rows(ball, feats, slot_conf, alive, cfg)
    rows += controls
    rows += _dribble_rows(ball, feats, slot_conf, alive, cfg)
    if not rows:
        return _empty()
    rows.sort(key=lambda r: (r[0], KINDS.index(r[1])))
    table = pd.DataFrame(rows, columns=COLUMNS)
    table = table.astype({"frame": np.int64, "confidence": np.float64, "dist_m": np.float64})
    table["team"] = table["team"].astype(object).where(table["team"].notna(), None)
    return table


def match_dead_evidence(ball, feats, tactical, cfg):
    """e_D for a whole match, with the Smart Tagging and the graded blocks of its game."""
    game = data.resolve_game(tactical.get("game") if isinstance(tactical, dict) else None)
    blocks = data.graded_blocks(game)
    return deadball.dead_evidence(ball, feats, data.load_tagging(game), blocks, cfg,
                                  kickoffs=deadball.kickoff_frames(game),
                                  pitch=data.pitch_m(game))


def match_players(tactical):
    """(px, py, visible, team_of_slot) of a whole match in metres, for kink_team_distances()."""
    from engine import tracks

    game = data.resolve_game(tactical.get("game") if isinstance(tactical, dict) else None)
    px, py, visible = tracks._player_metres(tactical, data.pitch_m(game))
    return px, py, visible, np.asarray(tactical["team_of_slot"])


def match_team_shape(tactical, cfg):
    """The team forward velocity of a whole match, with the graded blocks of its game (C15)."""
    from engine import tracks

    game = data.resolve_game(tactical.get("game") if isinstance(tactical, dict) else None)
    return tracks.team_forward_velocity(tactical, data.graded_blocks(game), data.pitch_m(game),
                                        cfg)


def touch_mask(events, n_frames):
    """bool (n_frames,): frames of touch, release and contested rows (not control rows).

    engine.deadball uses this mask for "no touch" and for the last toucher when it has events.
    """
    mask = np.zeros(int(n_frames), dtype=bool)
    if events is None or len(events) == 0:
        return mask
    frames = events.loc[events["kind"].isin(("touch", "release", "contested")), "frame"]
    frames = frames.to_numpy(dtype=np.int64)
    mask[frames[(frames >= 0) & (frames < n_frames)]] = True
    return mask


def event_table(tactical, cfg=None):
    """DataFrame[frame, kind, team, confidence] for the whole match, for tools/forensics.py.

    The function runs engine.tracks.prepare() and detect_events(). It keeps the last result in
    memory for the same `tactical` object, because forensics calls it once per transition. It
    writes nothing to disk.
    """
    from engine import tracks

    cfg = cfg or Config.default()
    cache = _TABLE_CACHE
    if cache["tactical"] is tactical and cache["cfg"] == cfg and cache["table"] is not None:
        return cache["table"].copy()
    ball, feats, slot_conf = tracks.prepare(tactical, cfg)
    table = detect_events(ball, feats, tactical, slot_conf, cfg)[TABLE_COLUMNS]
    cache.update({"tactical": tactical, "cfg": cfg, "table": table})
    return table.copy()

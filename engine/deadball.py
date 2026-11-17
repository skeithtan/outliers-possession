"""Dead ball: per-frame D evidence, alive->D onsets, D->alive restarts (PLAN.md 1.1, 1.4, 3.4; card P1-B).

    dead_evidence(ball, feats, tagging, blocks, cfg)   float32 (N,) e_D in [0, 1]
    dead_onsets(ball, feats, cfg)                      DataFrame[frame, mechanism, confidence]
    dead_onsets(..., events=None)                      the same, with real touch events (P1-A)
    restarts(ball, feats, events, tagging, cfg)        DataFrame[frame, team, kind, confidence, confirmed]
    chain_confirmed(frames, players, barriers, window) bool (n,) the players test plus the C11 chain
    placed_kicks(ball, stationary, rises, off_pitch)   [(frame, rest_start)] kicks of a ball at rest (C12)
    keeper_hold_gaps(ball, keepers, pitch, cfg)        [(start, stop, team, dist)] ball gaps in a goalkeeper's hands (C13)
    keeper_hold_mask(ball, keepers, pitch, cfg)        bool (N,) the frames of those gaps (C13)
    stop_signature_mask(feats, cfg, fps)               bool (N,) deep stop or collective stop frames (C13)
    kickoff_frames(game)                               {half: frame}, from games.py (not detected)
    ghost_mask(ball, feats, cfg)                       bool (N,) stuck / ghost ball frames (R-F5)
    smoothed_speed(meanv, cfg, fps)                    float (N,) centred moving mean of meanv_all (knee)
    deceleration_knee(smooth, entry, lo, cfg, fps)     frame of the players' deceleration knee, or None
    gap_starts(visible)                                int (N,) first frame of the ball gap of each frame

Inputs are the dicts of engine.tracks.prepare(): `ball` (metres, m/s, `visible`, `run_id`, `fps`)
and `feats` (`dH, dA, dmin, d_second, nearest_team, meanv_all, meanv_near, sep, ...`). NaN means
not observed and never satisfies a threshold. Every threshold and weight is in engine.config.

Semantics (PLAN 3.4, the card; do not re-derive):
  * A ball gap alone is NO evidence; gap + collective stop (meanv_all < V_STOP) is strong.
  * Ball outside the pitch rectangle (+ PITCH_MARGIN_M) is near-certain.
  * A stationary ball with one taker and everybody else away is a set-piece placement (strong);
    stationary with nobody near after a fast event is a ghost ball: no placement term there.
  * Slow players / large separation alone are weak; SET PIECES tags are a weak prior on the last
    SETPIECE_TAG_TAIL_S of the window only; GOALS windows are never a D mask (their end is a
    restart prior).
  * Pre-kickoff frames of a graded block are certain D; kickoffs come from games.py.

Touches: "no touch" (collective stop) and "last toucher" (out prior) use the event table of
engine.events when the caller gives one (`events=`, M2). With `events=None` (M1) they use the touch
proxy: dv_k >= DV_TOUCH with a player within R_TOUCH. M1 passes None and stays as it was scored.

numpy + pandas only. Deterministic. Never reads the ground truth.
"""

import numpy as np
import pandas as pd

from engine import data
from engine.config import Config
from engine.tracks import _runs, penalty_area_mask, velocity_step

ONSET_COLUMNS = ["frame", "mechanism", "confidence"]
RESTART_COLUMNS = ["frame", "team", "kind", "confidence", "confirmed"]
MECHANISMS = ("out", "pursuit_stop", "collective_stop")
TEAM_LETTER = {data.HOME: "H", data.AWAY: "A"}
OTHER_TEAM = {"H": "A", "A": "H"}
SET_PIECES_CODE = "SET PIECES"
GOALS_CODE = "GOALS"


# -- small helpers ------------------------------------------------------------------------------

def _fps(ball):
    """Frame rate stored by tracks.clean_ball, else the match fps from games.py."""
    fps = ball.get("fps") if isinstance(ball, dict) else None
    return float(fps) if fps is not None else float(data.fps())


def _frames(seconds, fps):
    """Whole frames in `seconds` (rounded, at least 1)."""
    return max(1, int(round(float(seconds) * fps)))


def _below(values, threshold):
    """values < threshold with NaN -> False."""
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(values) & (values < threshold)


def _above(values, threshold):
    """values > threshold with NaN -> False."""
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(values) & (values > threshold)


def _dilate_back(ends_mask, width):
    """Mark frames t-width+1..t for every True t of `ends_mask` (window ending at t)."""
    n = ends_mask.shape[0]
    diff = np.zeros(n + 1, dtype=np.int64)
    ends = np.flatnonzero(ends_mask)
    diff[np.maximum(ends - width + 1, 0)] += 1
    diff[ends + 1] -= 1
    return np.cumsum(diff[:n]) > 0


def _window_range(values, width):
    """max - min of values over the window ending at t (NaN if any NaN or t < width - 1)."""
    n = values.shape[0]
    out = np.full(n, np.nan)
    if n < width:
        return out
    view = np.lib.stride_tricks.sliding_window_view(values, width)
    out[width - 1:] = view.max(axis=1) - view.min(axis=1)  # NaN propagates: any gap -> NaN
    return out


def _team_letter(code):
    """'H' / 'A' for a tracks team code, None otherwise."""
    return TEAM_LETTER.get(int(code))


def touch_proxy(ball, feats, cfg):
    """bool (N,): crude touch stand-in of M1 (dv_k >= DV_TOUCH, a player within R_TOUCH)."""
    return (~_below(ball["dv_k"], cfg.DV_TOUCH) & np.isfinite(ball["dv_k"])
            & ~_above(feats["dmin"], cfg.R_TOUCH) & np.isfinite(feats["dmin"]))


def touch_frames(ball, feats, cfg, events=None):
    """bool (N,): touch frames from the event table if given, else from touch_proxy (M1)."""
    if events is None:
        return touch_proxy(ball, feats, cfg)
    from engine.events import touch_mask
    return touch_mask(events, ball["visible"].shape[0])


def outside_pitch(ball, pitch, margin):
    """bool (N,): visible ball beyond the pitch rectangle by more than `margin` metres."""
    length, width = pitch
    x, y = ball["x"], ball["y"]
    with np.errstate(invalid="ignore"):
        out = (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)
    return ball["visible"] & out


def stationary_mask(ball, cfg):
    """bool (N,): frames inside a window of STATIONARY_S with the ball visible throughout and its
    bounding box diagonal below STATIONARY_MAX_DISP_M (net displacement, PLAN 3.4)."""
    width = _frames(cfg.STATIONARY_S, _fps(ball))
    span = np.hypot(_window_range(ball["x"], width), _window_range(ball["y"], width))
    return _dilate_back(_below(span, cfg.STATIONARY_MAX_DISP_M), width)


def _prior_run_max_speed(ball):
    """Max ball speed earlier in the same visible run (inclusive), NaN outside runs."""
    speed = np.where(np.isfinite(ball["speed"]), ball["speed"], 0.0)
    out = np.full(speed.shape, np.nan)
    starts, stops = _runs(ball["visible"])
    for start, stop in zip(starts, stops):
        out[start:stop] = np.maximum.accumulate(speed[start:stop])
    return out


def ghost_mask(ball, feats, cfg):
    """bool (N,): stationary ball with nobody within GHOST_NOBODY_M after a > GHOST_PRIOR_SPEED
    event in the same visible run: the tracker is stuck on something else, typically the GK holding
    the ball after a shot (PLAN 1.1, R-F5: 13755-13800). Suppresses the placement term; the
    orchestrator freezes the last alive team there."""
    nobody = ~(np.isfinite(feats["dmin"]) & (np.nan_to_num(feats["dmin"], nan=np.inf)
                                             <= cfg.GHOST_NOBODY_M))
    after_fast = _above(_prior_run_max_speed(ball), cfg.GHOST_PRIOR_SPEED)
    return stationary_mask(ball, cfg) & nobody & after_fast


def setpiece_mask(ball, feats, cfg):
    """bool (N,): set-piece placement: stationary, one player within SETPIECE_TAKER_M, everybody
    else beyond SETPIECE_OTHERS_M (PLAN 3.4), not a ghost ball."""
    taker = ~_above(feats["dmin"], cfg.SETPIECE_TAKER_M) & np.isfinite(feats["dmin"])
    second = np.nan_to_num(feats["d_second"], nan=np.inf)
    others_away = second > cfg.SETPIECE_OTHERS_M
    return stationary_mask(ball, cfg) & taker & others_away & ~ghost_mask(ball, feats, cfg)


def _tag_rows(tagging, code):
    """Rows of the Smart Tagging table with this code (empty if no table)."""
    if tagging is None or len(tagging) == 0:
        return []
    rows = tagging[tagging["code"].astype(str).str.upper() == code]
    return list(rows.itertuples(index=False))


def tag_tail_mask(tagging, n_frames, cfg, fps):
    """bool (N,): the last SETPIECE_TAG_TAIL_S of every SET PIECES window (PLAN 1.6, 3.4)."""
    mask = np.zeros(n_frames, dtype=bool)
    tail = _frames(cfg.SETPIECE_TAG_TAIL_S, fps)
    for row in _tag_rows(tagging, SET_PIECES_CODE):
        lo = max(int(row.start_frame), int(row.end_frame) - tail, 0)
        hi = min(int(row.end_frame), n_frames)
        if hi > lo:
            mask[lo:hi] = True
    return mask


def _collective(feats, cfg):
    """bool (N,): meanv_all < V_STOP AND sep > SEP_DEAD (PLAN 3.4 collective stop, per frame)."""
    return _below(feats["meanv_all"], cfg.V_STOP) & _above(feats["sep"], cfg.SEP_DEAD)


def _stop_runs(feats, cfg, fps):
    """(starts, stops) of collective-stop runs lasting at least COLLECTIVE_STOP_S."""
    starts, stops = _runs(_collective(feats, cfg))
    keep = (stops - starts) >= _frames(cfg.COLLECTIVE_STOP_S, fps)
    return starts[keep], stops[keep]


def _confirmed(onset, stop_starts, stop_stops, horizon):
    """True if a qualifying collective stop overlaps [onset, onset + horizon]."""
    return bool(np.any((stop_starts <= onset + horizon) & (stop_stops > onset)))


# -- evidence -----------------------------------------------------------------------------------

def dead_evidence(ball, feats, tagging, blocks, cfg=None, kickoffs=None, pitch=None):
    """Per-frame dead-ball evidence e_D in [0, 1], float32 (N,).

    e_D = sigmoid(DEAD_LOGIT_BIAS + sum of active term weights) with the terms of PLAN 3.4:
      W_GAP_STOP  ball missing and meanv_all < V_STOP (a gap alone adds nothing)
      W_SLOW      meanv_all < V_STOP;  W_SEP  sep > SEP_DEAD  (weak, also active inside gaps)
      W_OUT       ball outside the rectangle by more than PITCH_MARGIN_M
      W_SETPIECE  set-piece placement (suppressed on ghost-ball frames)
      W_TAG_TAIL  last SETPIECE_TAG_TAIL_S of a SET PIECES window
    Then e_D = 1 on graded frames before the kickoff of their block: `blocks` is
    [(half, first, last_inclusive)] (engine.data.graded_blocks()) and `kickoffs` {half: frame}
    (default engine.data.kickoff_frames()); blocks=None skips the term.
    """
    cfg = cfg or Config.default()
    pitch = pitch or data.pitch_m()
    fps = _fps(ball)
    n = ball["visible"].shape[0]
    slow = _below(feats["meanv_all"], cfg.V_STOP)
    logit = np.full(n, cfg.DEAD_LOGIT_BIAS, dtype=np.float64)
    logit += cfg.W_GAP_STOP * (~ball["visible"] & slow)
    logit += cfg.W_SLOW * slow
    logit += cfg.W_SEP * _above(feats["sep"], cfg.SEP_DEAD)
    logit += cfg.W_OUT * outside_pitch(ball, pitch, cfg.PITCH_MARGIN_M)
    logit += cfg.W_SETPIECE * setpiece_mask(ball, feats, cfg)
    logit += cfg.W_TAG_TAIL * tag_tail_mask(tagging, n, cfg, fps)
    e_dead = 1.0 / (1.0 + np.exp(-logit))
    if blocks is not None:
        kickoffs = kickoff_frames() if kickoffs is None else kickoffs
        for half, first, last in blocks:
            kick = kickoffs.get(half)
            if kick is not None and kick > first:
                e_dead[first:min(kick, last + 1, n)] = 1.0
    return e_dead.astype(np.float32)


# -- deceleration knee (COMPONENT_KNEE.md, IDEAS 2) ---------------------------------------------

def smoothed_speed(meanv, cfg=None, fps=25.0):
    """float64 (N,): centred moving mean of `meanv` over KNEE_SMOOTH_S, NaN frames skipped.

    The window has 2 x round(KNEE_SMOOTH_S x fps / 2) + 1 frames (13 at 25 fps) and is cut at
    the array ends. A window with no finite value gives NaN.
    """
    cfg = cfg or Config.default()
    values = np.asarray(meanv, dtype=np.float64)
    n = values.shape[0]
    half = int(round(float(cfg.KNEE_SMOOTH_S) * fps / 2.0))
    finite = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    sums, counts = total[hi] - total[lo], count[hi] - count[lo]
    out = np.full(n, np.nan)
    has = counts > 0
    out[has] = sums[has] / counts[has]
    return out


def deceleration_knee(smooth, entry, lo, cfg=None, fps=25.0):
    """Frame of the players' deceleration knee before a dwell entry, or None (no knee).

    `smooth` is smoothed_speed() of meanv_all, `entry` the entry frame and `lo` the earliest
    frame the knee may take. The speed before the fall, `pre`, is the median of the finite
    `smooth` over [entry - KNEE_PRE_FROM_S, entry - KNEE_PRE_TO_S). The level is
    V_STOP + KNEE_LEVEL_FRAC x (pre - V_STOP). The knee is the last frame of [lo, entry] with
    `smooth` at or above the level, else `lo` (the fall began before the window). No knee when
    lo >= entry, when `pre` is undefined or when `pre` is not above V_STOP (the players were
    already slow).
    """
    cfg = cfg or Config.default()
    entry, lo = int(entry), int(lo)
    if lo >= entry:
        return None
    smooth = np.asarray(smooth, dtype=np.float64)
    first = max(0, entry - int(round(float(cfg.KNEE_PRE_FROM_S) * fps)))
    last = max(0, entry - int(round(float(cfg.KNEE_PRE_TO_S) * fps)))
    pre = smooth[first:last]
    pre = pre[np.isfinite(pre)]
    if pre.size == 0:
        return None
    pre = float(np.median(pre))
    if not pre > cfg.V_STOP:
        return None
    level = cfg.V_STOP + float(cfg.KNEE_LEVEL_FRAC) * (pre - cfg.V_STOP)
    window = smooth[lo:entry + 1]
    with np.errstate(invalid="ignore"):
        hits = np.flatnonzero(np.isfinite(window) & (window >= level))
    return int(lo + hits[-1]) if hits.size else lo


def gap_starts(visible):
    """int64 (N,): for each frame, the last visible frame at or before it plus 1 (0 if none).

    On a missing frame this is the first frame of its ball gap, the frame after the last
    sighting.
    """
    visible = np.asarray(visible, dtype=bool)
    idx = np.where(visible, np.arange(visible.shape[0]), -1)
    return (np.maximum.accumulate(idx) + 1).astype(np.int64)


# -- onsets -------------------------------------------------------------------------------------

def _boundary_distance(x, y, ux, uy, pitch):
    """Distance along the unit heading (ux, uy) from (x, y) to the first pitch line it meets."""
    length, width = pitch
    best = np.inf
    for pos, step, lo, hi in ((x, ux, 0.0, length), (y, uy, 0.0, width)):
        if step > 0:
            best = min(best, (hi - pos) / step)
        elif step < 0:
            best = min(best, (lo - pos) / step)
    return best


def _is_throw_in_reappearance(ball, frame, pitch, cfg):
    """True if the excursion beyond the line that starts at `frame` is a throw-in.

    A ball that goes out crosses the line from inside the pitch. The tracker first sees a
    thrown-in ball in the hands of the thrower, beyond the touchline. The rule holds when all of
    these are true:
      * The ball is not visible at frame - 1. The run starts outside, after a gap.
      * The ball is beyond a touchline (y < 0 or y > width) with 0 <= x <= length.
      * y moves toward the pitch from `frame` to the last visible frame of the next
        OUT_INWARD_FRAMES frames. When none of them is visible, vy points toward the pitch.
    A reappearance beyond a goal line keeps its onset: the ball comes out of the net after a goal.
    """
    length, width = pitch
    x, y, visible = ball["x"], ball["y"], ball["visible"]
    n = visible.shape[0]
    if frame > 0 and visible[frame - 1]:
        return False  # the crossing was seen from inside: a real out
    if not (0.0 <= x[frame] <= length) or (0.0 <= y[frame] <= width):
        return False  # beyond a goal line, or not beyond a touchline
    last = min(frame + int(cfg.OUT_INWARD_FRAMES), n - 1)
    while last > frame and not visible[last]:
        last -= 1
    dy = y[last] - y[frame] if last > frame else ball["vy"][frame]
    if not np.isfinite(dy):
        return False
    return bool(dy > 0.0) if y[frame] < 0.0 else bool(dy < 0.0)


def _out_onsets(ball, cfg, pitch, fps, stop_starts, stop_stops, horizon):
    """Out onsets: observed crossings (no confirmation) and missing-ball extrapolations."""
    rows = []
    length, width = pitch
    x, y, visible = ball["x"], ball["y"], ball["visible"]
    n = visible.shape[0]
    # (a) observed: the ball is seen beyond the line + margin; onset = first frame of the
    # excursion beyond the line itself (walk back while visible and outside the rectangle).
    # A throw-in reappearance is not an onset (C10, see _is_throw_in_reappearance).
    beyond_line = outside_pitch(ball, pitch, 0.0)
    starts, _ = _runs(outside_pitch(ball, pitch, cfg.PITCH_MARGIN_M))
    for start in starts:
        frame = int(start)
        while frame - 1 >= 0 and beyond_line[frame - 1]:
            frame -= 1
        if _is_throw_in_reappearance(ball, frame, pitch, cfg):
            continue
        rows.append((frame, "out", cfg.CONF_OUT_OBSERVED))
    # (b) the ball goes missing heading for a line within OUT_MAX_RANGE_M (PLAN 1.4: 12851).
    run_starts, run_stops = _runs(visible)
    for idx, stop in enumerate(run_stops):
        last = int(stop) - 1
        if stop >= n:
            continue  # the run reaches the end of the array: no gap follows
        next_visible = int(run_starts[idx + 1]) if idx + 1 < len(run_starts) else n
        if not (0.0 <= x[last] <= length and 0.0 <= y[last] <= width):
            continue  # already outside: case (a)
        speed = ball["speed"][last]
        if not np.isfinite(speed) or speed < cfg.HEADING_MIN_SPEED:
            continue
        ux, uy = ball["vx"][last] / speed, ball["vy"][last] / speed
        dist = _boundary_distance(x[last], y[last], ux, uy, pitch)
        if dist > cfg.OUT_MAX_RANGE_M:
            continue
        decel = cfg.FRICTION_DECEL
        if speed * speed >= 2.0 * decel * dist:  # friction-decayed roll reaches the line
            seconds = (speed - np.sqrt(speed * speed - 2.0 * decel * dist)) / decel
            onset, conf = last + int(round(seconds * fps)), cfg.CONF_OUT_EXTRAPOLATED
        else:
            onset, conf = last + int(cfg.DELTA_OUT_LAG), cfg.CONF_OUT_LAG
        if onset >= next_visible:
            continue  # seen again inside the pitch before the crossing / lag: not out
        if _confirmed(onset, stop_starts, stop_stops, horizon):
            rows.append((int(onset), "out", conf))
    return rows


def _first_peak(speed, start, stop):
    """First local maximum of speed in [start, stop): the end of the release (a re-play later in
    the same fast run, as at 17388-17442, must not move the onset)."""
    for frame in range(start, stop - 1):
        if speed[frame + 1] < speed[frame]:
            return frame
    return stop - 1


def _pursuit_onsets(ball, feats, cfg, stop_starts, stop_stops, horizon):
    """Pursuit-stop onsets (PLAN 3.4; 15889, 17380): fast ball, kicking team falling behind, the
    players nearest the ball dropping below PURSUIT_STOP_SPEED within ONSET_CONFIRM_S (they must
    be at or above it in the PURSUIT_KICK_LOOKBACK frames before the run: "falling below").

    Onset = the start of the nearest players' final deceleration into that drop, but not before
    the ball's peak speed (the release must be complete). Only the kicking team's distance must
    rise: the other team may be near the landing (17380). The kicking team is the one clearly
    nearer the ball (mean distances differ by >= CONTESTED_MARGIN_M) over PURSUIT_KICK_LOOKBACK
    frames before the run, and none of its players may come within R_TOUCH of the ball in the
    COLLECTIVE_STOP_S after the onset ("nobody chases it"; a pass received by a teammate is not
    a stop: P1-B forensics 18612)."""
    rows = []
    speed = ball["speed"]
    near = feats["meanv_near"]
    n = speed.shape[0]
    chase = _frames(cfg.COLLECTIVE_STOP_S, _fps(ball))
    fast = ball["visible"] & _above(speed, cfg.PURSUIT_MIN_BALL_SPEED)
    starts, stops = _runs(fast)
    for start, stop in zip(starts, stops):
        start, stop = int(start), int(stop)
        lo = max(0, start - int(cfg.PURSUIT_KICK_LOOKBACK))
        if not ball["visible"][lo:start].all() or start == 0:
            continue  # the kick must be seen to know the kicking team (not a reappearance)
        peak = _first_peak(speed, start, stop)
        before = near[lo:start + 1]
        if not np.isfinite(before).any() or np.nanmax(before) < cfg.PURSUIT_STOP_SPEED:
            continue  # the nearest players must FALL below the stop speed, not be below already
        with np.errstate(invalid="ignore"):
            mean_h = np.nanmean(feats["dH"][lo:start + 1]) if np.isfinite(
                feats["dH"][lo:start + 1]).any() else np.inf
            mean_a = np.nanmean(feats["dA"][lo:start + 1]) if np.isfinite(
                feats["dA"][lo:start + 1]).any() else np.inf
        if not np.isfinite(min(mean_h, mean_a)):
            continue
        if abs(mean_h - mean_a) < cfg.CONTESTED_MARGIN_M:
            continue  # contested kick: the kicking team is unknown (17318: 1.3 vs 1.2 m)
        d_kick = feats["dH"] if mean_h <= mean_a else feats["dA"]
        end = min(n, start + horizon)
        slow = np.flatnonzero(_below(near[start:end], cfg.PURSUIT_STOP_SPEED))
        if slow.size == 0:
            continue
        drop = start + int(slow[0])
        descent = drop
        while descent - 1 >= 0 and np.isfinite(near[descent - 1]) and \
                near[descent - 1] > near[descent]:
            descent -= 1
        onset = max(peak, descent)
        if not (np.isfinite(d_kick[onset]) and np.isfinite(d_kick[start])
                and d_kick[onset] > d_kick[start]):
            continue
        after = d_kick[onset:min(n, onset + chase)]
        if np.isfinite(after).any() and np.nanmin(after) <= cfg.R_TOUCH:
            continue  # a kicking-team player reaches the ball: a pass, not a stop (18612)
        if _confirmed(onset, stop_starts, stop_stops, horizon):
            rows.append((onset, "pursuit_stop", cfg.CONF_PURSUIT))
    return rows


def _collective_onsets(ball, feats, cfg, fps, events=None):
    """Collective stop with the ball visible and no touch for >= COLLECTIVE_STOP_S (PLAN 3.4);
    onset = first frame of the touch-free stretch. Lowest confidence. Touches come from
    touch_frames(): the event table if given, else the M1 proxy."""
    quiet = _collective(feats, cfg) & ball["visible"] & ~touch_frames(ball, feats, cfg, events)
    starts, stops = _runs(quiet)
    keep = (stops - starts) >= _frames(cfg.COLLECTIVE_STOP_S, fps)
    return [(int(s), "collective_stop", cfg.CONF_COLLECTIVE) for s in starts[keep]]


def dead_onsets(ball, feats, cfg=None, pitch=None, events=None):
    """alive->D onset events: DataFrame[frame, mechanism, confidence], sorted by frame.

    mechanism:
      out              observed beyond the line (+ margin; onset = first frame beyond the line,
                       but not when the ball reappears beyond a touchline and moves inward: a
                       throw-in, C10),
                       or the ball goes missing heading for a line within OUT_MAX_RANGE_M: onset =
                       friction-decayed crossing (FRICTION_DECEL) if the roll reaches the line,
                       else last_visible + DELTA_OUT_LAG (no 1 s limit, PLAN 1.4: 12851)
      pursuit_stop     see _pursuit_onsets
      collective_stop  see _collective_onsets
    Missing-ball outs and pursuit stops are kept only if a collective stop starts within
    ONSET_CONFIRM_S (config provenance: the two false outs inside 16400-17293).
    `events` (engine.events.detect_events) gives the touches of the collective stop. None keeps
    the M1 touch proxy.
    """
    cfg = cfg or Config.default()
    pitch = pitch or data.pitch_m()
    fps = _fps(ball)
    horizon = _frames(cfg.ONSET_CONFIRM_S, fps)
    stop_starts, stop_stops = _stop_runs(feats, cfg, fps)
    rows = _out_onsets(ball, cfg, pitch, fps, stop_starts, stop_stops, horizon)
    rows += _pursuit_onsets(ball, feats, cfg, stop_starts, stop_stops, horizon)
    rows += _collective_onsets(ball, feats, cfg, fps, events)
    table = pd.DataFrame(sorted(set(rows), key=lambda r: (r[0], MECHANISMS.index(r[1]))),
                         columns=ONSET_COLUMNS)
    return table.astype({"frame": np.int64, "confidence": np.float64})


# -- restarts -----------------------------------------------------------------------------------

def _rise_start(speed, first, last_exclusive):
    """Frame where the speed rise begins: from the largest one-frame increase in
    [first, last_exclusive), walk back while the speed was still increasing."""
    seg = speed[first:last_exclusive + 1]
    accel = np.diff(seg)
    if accel.size == 0 or not np.isfinite(accel).any():
        return first
    frame = first + int(np.nanargmax(accel))
    while frame - 1 >= first and np.isfinite(speed[frame - 1]) and speed[frame - 1] < speed[frame]:
        frame -= 1
    return frame


def _nearest_in_window(feats, lo, hi):
    """(min distance, team letter or None) of the nearest player over frames [lo, hi)."""
    window = feats["dmin"][lo:hi]
    if window.size == 0 or not np.isfinite(window).any():
        return np.inf, None
    at = lo + int(np.nanargmin(window))
    return float(feats["dmin"][at]), _team_letter(feats["nearest_team"][at])


def _last_toucher_before(ball, feats, cfg, frame, events=None):
    """Team letter of the last touch before `frame`, or None.

    With `events` it is the team of the last touch or release row with a team before `frame`.
    With None it is the nearest team on the last touch-proxy frame (M1).
    """
    if events is not None:
        rows = events[(events["frame"] < frame) & events["kind"].isin(("touch", "release"))
                      & events["team"].isin(("H", "A"))]
        return str(rows["team"].iloc[-1]) if len(rows) else None
    touches = np.flatnonzero(touch_proxy(ball, feats, cfg)[:max(frame, 0)])
    return _team_letter(feats["nearest_team"][touches[-1]]) if touches.size else None


def _restart_priors(frame, team, onsets, ball, feats, tagging, cfg, fps, events=None):
    """Net confidence adjustment from the out prior and Smart Tagging (boosts only)."""
    step, adjust = cfg.RESTART_PRIOR_STEP, 0.0
    if onsets is not None and len(onsets):
        before = onsets[onsets["frame"] < frame]
        if len(before) and before["mechanism"].iloc[-1] == "out" and team is not None:
            toucher = _last_toucher_before(ball, feats, cfg, int(before["frame"].iloc[-1]),
                                           events)
            if toucher is not None:
                adjust += step if OTHER_TEAM[toucher] == team else -step
    tail = _frames(cfg.SETPIECE_TAG_TAIL_S, fps)
    for row in _tag_rows(tagging, SET_PIECES_CODE):
        if int(row.start_frame) <= frame < int(row.end_frame) + tail:
            adjust += step  # a tagged set piece contains (or just precedes) this restart
            if team is not None and row.team in OTHER_TEAM:
                adjust += step if row.team == team else -step
            break
    for row in _tag_rows(tagging, GOALS_CODE):
        if abs(frame - int(row.end_frame)) <= tail:  # GOALS end = kick-off after the goal
            adjust += step
            break
    return adjust


def placed_kicks(ball, stationary, rises, off_pitch=None):
    """[(frame, rest_start)]: the kicks of a ball that lay at rest (C12, F1 rank 1).

    `stationary` is stationary_mask(). `rises` is [(frame, run_stop)] in frame order. The frame
    is the rise start of a run of dv_RESTART_K above DV_RESTART. The run_stop is the end of
    that run. A rise is a placed-ball kick when a stationary frame lies between the end of the
    previous rise run and the rise start, in the visible run of the rise. So the ball lay at
    rest, nobody played it after that, and the tracker saw it all the time.

    rest_start is the first frame of the stationary stretch that ends last before the kick.
    That is the frame where the ball came to rest. The kick can come some time after the last
    stationary frame. The taker often touches the ball first, and the smoothed track drifts at
    up to 2 m/s for about 1 s. In 77859 the last stationary frame lies 38 frames before the
    rise. In 82254 and 86635 it lies 10 frames before.

    `off_pitch` is a bool (N,) mask from outside_pitch() with PITCH_MARGIN_M. A rise on such a
    frame is no placed-ball kick. Laws 13, 16 and 17 put the ball of a free kick, a goal kick
    and a corner on the pitch. A ball at rest beyond a line that then moves comes back from off
    the pitch (186261: 6.6 m beyond the touchline, nobody within 12 m). None skips the test.
    """
    run_id = ball["run_id"]
    out = []
    after = 0                        # first frame after the end of the previous rise run
    for frame, run_stop in rises:
        frame, run_stop = int(frame), int(run_stop)
        lo, hi = after, frame + 1
        after = max(after, run_stop)
        if hi <= lo or (off_pitch is not None and off_pitch[frame]):
            continue
        rest = np.flatnonzero(stationary[lo:hi])
        if rest.size == 0:
            continue
        last = lo + int(rest[-1])
        if run_id[last] < 0 or run_id[last] != run_id[frame]:
            continue                 # a gap lies between the rest and the kick
        start = last
        while start - 1 >= 0 and stationary[start - 1]:
            start -= 1
        out.append((frame, start))
    return out


def _comes_to_rest(ball, rest_starts, frame, window):
    """True if the ball that reappears at `frame` comes to rest within `window` frames.

    `rest_starts` is the sorted array of the rest_start frames of placed_kicks(). The rest must
    start in (frame, frame + window] and in the visible run of `frame`. Then a placed-ball kick
    follows the reappearance, and the reappearance is not the restart (C12 (b): 82254, 143837).
    An empty `rest_starts` gives False.
    """
    run_id = ball["run_id"]
    lo = np.searchsorted(rest_starts, frame, side="right")
    hi = np.searchsorted(rest_starts, frame + int(window), side="right")
    return any(run_id[int(rest)] == run_id[frame] for rest in rest_starts[lo:hi])


def keeper_hold_gaps(ball, keepers, pitch, cfg=None):
    """[(start, stop, team, dist)]: the ball gaps that start in a goalkeeper's hands (C13).

    `keepers` is tracks.goalkeeper_positions(). A gap [start, stop) starts in the hands of a
    goalkeeper when, on the last visible frame before the gap, all of these are true:
      * The ball is inside the own penalty area of a team's goalkeeper-like slot.
      * That slot is within R_TOUCH of the ball.
      * The ball is slower than DV_RESTART. A goalkeeper catches or picks up a slow ball.
    `team` is the column of that team in `keepers` (0 for H, 1 for A) and `dist` the distance
    in metres from the slot to the ball. Physics: Law 12 lets a goalkeeper hold the ball in his
    hands, and play goes on. The tracker does not see a ball in the hands, so the hold is a gap
    (133932, 187146, F1 rank 2). A gap at the start of the array is never a hold.
    """
    cfg = cfg or Config.default()
    out = []
    starts, stops = _runs(~ball["visible"])
    for start, stop in zip(starts, stops):
        last = int(start) - 1
        if last < 0 or not _below(ball["speed"][last:last + 1], cfg.DV_RESTART)[0]:
            continue
        bx, by = ball["x"][last], ball["y"][last]
        for team in range(keepers["x"].shape[1]):
            goal_x = keepers["own_goal_x"][last, team]
            if not np.isfinite(goal_x):
                continue
            dist = np.hypot(keepers["x"][last, team] - bx, keepers["y"][last, team] - by)
            in_box = penalty_area_mask(bx, by, goal_x, pitch, cfg)
            if bool(in_box) and np.isfinite(dist) and dist <= cfg.R_TOUCH:
                out.append((int(start), int(stop), int(team), float(dist)))
                break
    return out


def keeper_hold_mask(ball, keepers, pitch, cfg=None):
    """bool (N,): the frames of the gaps of keeper_hold_gaps() (C13)."""
    out = np.zeros(ball["visible"].shape[0], dtype=bool)
    for start, stop, _, _ in keeper_hold_gaps(ball, keepers, pitch, cfg):
        out[start:stop] = True
    return out


def stop_signature_mask(feats, cfg=None, fps=None):
    """bool (N,): frames inside a deep stop or a collective stop (C13).

    A deep stop is a run of meanv_all below GK_HOLD_DEEP_STOP_SPEED that lasts at least
    GK_HOLD_DEEP_STOP_S. A collective stop is a run of meanv_all below V_STOP and sep above
    SEP_DEAD that lasts at least COLLECTIVE_STOP_S (PLAN 3.4). Either one marks a stoppage. A
    goalkeeper hold with neither one is play (keeper_hold_mask).
    """
    cfg = cfg or Config.default()
    fps = float(fps) if fps is not None else float(data.fps())
    n = feats["meanv_all"].shape[0]
    out = np.zeros(n, dtype=bool)
    starts, stops = _runs(_below(feats["meanv_all"], cfg.GK_HOLD_DEEP_STOP_SPEED))
    for start, stop in zip(starts, stops):
        if stop - start >= _frames(cfg.GK_HOLD_DEEP_STOP_S, fps):
            out[int(start):int(stop)] = True
    for start, stop in zip(*_stop_runs(feats, cfg, fps)):
        out[int(start):int(stop)] = True
    return out


def chain_confirmed(frames, players, barriers, window):
    """bool (n,): the players test of each candidate, extended by the "play goes on" chain.

    `frames` are the candidate frames in table order (sorted by frame). `players[i]` is True when
    meanv_all rises above V_STOP within RESTART_CONFIRM_S of candidate i. A candidate that fails
    that test is still confirmed when the next later candidate that passes it starts at most
    `window` frames after it, and no frame of `barriers` (stoppage onsets) lies in
    (frames[i], that frame]. The chain uses the players test of the next candidate only, so a
    chain never confirms through another chained candidate. A window of 0 or less turns the
    chain off.

    Physics (C11, RESEARCH-2 1b, M15): whether play resumed is a property of the passing
    sequence after a restart, not of its first kick. After a short goal kick the players can
    walk for more than 6 s, but the next kick of the sequence soon follows (18355, then a
    confirmed candidate 60 frames later). A re-play inside a stoppage is an isolated contact.
    No players-confirmed candidate follows it for 13 to 16 s (16058: 335 frames, 17425: 401).
    """
    frames = np.asarray(frames, dtype=np.int64)
    players = np.asarray(players, dtype=bool)
    out = players.copy()
    if int(window) <= 0 or frames.size == 0:
        return out
    barriers = np.sort(np.asarray(barriers, dtype=np.int64))
    following = None                     # frame of the next later players-confirmed candidate
    i = frames.size - 1
    while i >= 0:
        j = i
        while j > 0 and frames[j - 1] == frames[i]:
            j -= 1                       # rows j..i share one frame
        frame = int(frames[i])
        if following is not None and following - frame <= int(window):
            lo = np.searchsorted(barriers, frame, side="right")
            hi = np.searchsorted(barriers, following, side="right")
            if hi == lo:
                out[j:i + 1] = True
        if players[j:i + 1].any():
            following = frame
        i = j - 1
    return out


def restarts(ball, feats, events, tagging, cfg=None, onsets=None, pitch=None):
    """D->alive restart candidates: DataFrame[frame, team, kind, confidence, confirmed].

    kind:
      kick      signed dv_RESTART_K (velocity_step over RESTART_K frames, same visible run) above
                DV_RESTART, independent of the pre-kick speed; frame = start of the speed rise;
                a player within RESTART_PLAYER_M over [frame - RESTART_K, frame + RESTART_K];
                team = that nearest player's team. A placed-ball kick (placed_kicks(), C12 (a))
                needs no player within RESTART_PLAYER_M: its team is the nearest player within
                REAPPEAR_TEAM_M, else None
      reappear  first frame of a visible run after a gap, already faster than REAPPEAR_SPEED;
                team = nearest player's team if within REAPPEAR_TEAM_M over the first RESTART_K
                frames, else None. No candidate when the ball comes to rest within
                RESTART_CONFIRM_S in the same visible run (C12 (b)): the placed-ball kick that
                follows is the restart
    PLACED_BALL_RESTART False turns both C12 clauses off.
    confirmed: meanv_all rises above V_STOP within RESTART_CONFIRM_S (players re-engage; config
    provenance: re-plays inside D at 16064 and 17428), or the chain rule of chain_confirmed():
    the next candidate that players confirm starts within RESTART_CHAIN_FRAMES and no stoppage
    onset (out or pursuit_stop of `onsets`) lies between the two (C11, RESEARCH-2 1b).
    Candidates are NOT armed here (this function has no state): the caller accepts a candidate
    only after the ball has been dead for RESTART_ARM_S. `onsets` (dead_onsets()) enables the
    out -> other-team prior; Smart Tagging SET PIECES Type/Team and the GOALS end adjust the
    confidence only. `events` (engine.events.detect_events) gives the last toucher of the out
    prior. None keeps the M1 touch proxy. No timing offsets (P2). `pitch` (length, width) in
    metres, default engine.data.pitch_m(), places the ball of a placed-ball kick on the pitch.
    """
    cfg = cfg or Config.default()
    fps = _fps(ball)
    speed = ball["speed"]
    n = speed.shape[0]
    k = int(cfg.RESTART_K)
    confirm = _frames(cfg.RESTART_CONFIRM_S, fps)
    moving = _above(feats["meanv_all"], cfg.V_STOP)
    rows = []

    rise = _above(velocity_step(speed, k, ball["run_id"]), cfg.DV_RESTART)
    starts, stops = _runs(rise)
    rises = [(_rise_start(speed, int(start), min(n - 1, int(stop) + k)), int(stop))
             for start, stop in zip(starts, stops)]
    placed = []
    if cfg.PLACED_BALL_RESTART:
        off_pitch = outside_pitch(ball, pitch or data.pitch_m(), cfg.PITCH_MARGIN_M)
        placed = placed_kicks(ball, stationary_mask(ball, cfg), rises, off_pitch)
    placed_frames = set(frame for frame, _ in placed)
    for frame, _ in rises:
        dist, team = _nearest_in_window(feats, max(0, frame - k), min(n, frame + k + 1))
        if dist > cfg.RESTART_PLAYER_M:
            if frame not in placed_frames:
                continue
            team = team if dist <= cfg.REAPPEAR_TEAM_M else None   # C12 (a)
        rows.append((frame, team, "kick", cfg.CONF_RESTART_KICK))

    run_starts, _ = _runs(ball["visible"])
    rest_starts = np.array(sorted(rest for _, rest in placed), dtype=np.int64)
    for start in run_starts:
        start = int(start)
        if start == 0 or not _above(speed[start:start + 1], cfg.REAPPEAR_SPEED)[0]:
            continue
        if _comes_to_rest(ball, rest_starts, start, confirm):
            continue                     # C12 (b): the placed-ball kick is the restart
        dist, team = _nearest_in_window(feats, start, min(n, start + k + 1))
        rows.append((start, team if dist <= cfg.REAPPEAR_TEAM_M else None, "reappear",
                     cfg.CONF_RESTART_REAPPEAR))

    out = []
    for frame, team, kind, conf in sorted(rows, key=lambda r: (r[0], r[2])):
        conf = conf + _restart_priors(frame, team, onsets, ball, feats, tagging, cfg, fps,
                                      events)
        confirmed = bool(moving[frame:min(n, frame + confirm + 1)].any())
        out.append((int(frame), team, kind, float(np.clip(conf, 0.0, 1.0)), confirmed))
    barriers = []
    if onsets is not None and len(onsets):
        barriers = onsets.loc[onsets["mechanism"] != "collective_stop", "frame"].to_numpy()
    chained = chain_confirmed([row[0] for row in out], [row[4] for row in out], barriers,
                              int(cfg.RESTART_CHAIN_FRAMES))
    out = [row[:4] + (bool(flag),) for row, flag in zip(out, chained)]
    table = pd.DataFrame(out, columns=RESTART_COLUMNS)
    table = table.astype({"frame": np.int64, "confidence": np.float64, "confirmed": bool})
    table["team"] = table["team"].astype(object).where(table["team"].notna(), None)
    return table


# -- kickoff ------------------------------------------------------------------------------------

def kickoff_frames(game=None):
    """{half: kickoff video frame}, from games.py via engine.data (R-F12: read, not detected)."""
    return data.kickoff_frames(game)

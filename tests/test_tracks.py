"""P0-B: ball cleaning, kinematics and per-frame features on synthetic sequences.

No data bundle needed. Units under test: metres, m/s, dv_k in m/s over k frames (indexed at the
start of the window), headings in degrees (docs/plan/tasks/P0-B-tracks.md).
"""

import dataclasses

import numpy as np
import pytest

from engine import tracks
from engine.config import Config, load

FPS = 25.0
PITCH = (100.0, 50.0)   # synthetic pitch: normalised x * 100 m, y * 50 m
CFG = Config.default()


def ball_from_metres(x, y):
    """Normalised ball arrays from metre positions on PITCH."""
    return np.asarray(x, float) / PITCH[0], np.asarray(y, float) / PITCH[1]


def integrate(speeds, heading_deg=0.0, x0=10.0, y0=25.0):
    """Positions of a ball moving at speeds[t] m/s between frames t and t + 1."""
    step = np.asarray(speeds, float) / FPS
    rad = np.radians(heading_deg)
    x = x0 + np.concatenate(([0.0], np.cumsum(step * np.cos(rad))))
    y = y0 + np.concatenate(([0.0], np.cumsum(step * np.sin(rad))))
    return x, y


def clean(x, y, cfg=CFG, dmin=None):
    bx, by = ball_from_metres(x, y)
    return tracks.clean_ball(bx, by, PITCH, cfg, dmin=dmin, fps=FPS)


def tactical_from(px_m, py_m, team_of_slot, ball_x=None, ball_y=None, ps=None):
    """Synthetic tactical dict from metre positions (N, S)."""
    px_m = np.asarray(px_m, float)
    n = px_m.shape[0]
    if ball_x is None:
        ball_x = np.full(n, np.nan)
        ball_y = np.full(n, np.nan)
    bx, by = ball_from_metres(ball_x, ball_y)
    return {
        "px": (px_m / PITCH[0]).astype(np.float32),
        "py": (np.asarray(py_m, float) / PITCH[1]).astype(np.float32),
        "ps": (np.zeros_like(px_m) if ps is None else np.asarray(ps, float)).astype(np.float32),
        "bx": bx.astype(np.float32), "by": by.astype(np.float32),
        "team_of_slot": np.asarray(team_of_slot, dtype=np.int8),
        "n_frames": n, "n_slots": px_m.shape[1], "fps": int(FPS),
    }


# -- ball kinematics ---------------------------------------------------------------------------

def test_rolling_deceleration_recovered():
    """5 m/s decelerating at 0.5 m/s^2: speed and deceleration within 5%."""
    t = np.arange(101) / FPS
    x = 10.0 + 5.0 * t - 0.25 * t ** 2
    ball = clean(x, np.full_like(x, 25.0))
    expected = 5.0 - 0.5 * t
    assert ball["visible"].all()
    assert np.all(np.abs(ball["speed"] - expected) / expected < 0.05)
    decel = np.gradient(ball["speed"]) * FPS
    assert np.all(np.abs(decel[2:-2] + 0.5) < 0.05 * 0.5)


def test_kick_dv3_peak_and_heading():
    """1 -> 12 m/s over 2 frames: dv_3 peaks ~11 m/s at the kick, heading change ~0 deg."""
    speeds = np.r_[np.full(50, 1.0), 6.5, np.full(50, 12.0)]
    x, y = integrate(speeds, heading_deg=30.0)
    ball = clean(x, y)
    dv = ball["dv_3"]
    peak = int(np.nanargmax(dv))
    assert dv[peak] == pytest.approx(11.0, abs=0.3)
    # dv_3 is indexed at the start of its window: the ramp is frames 50-51, so the peak sits on
    # the plateau just before it.
    assert 47 <= peak <= 51
    assert ball["dv_3"] is ball["dv_k"]
    assert np.nanmax(ball["heading_change_3"]) < 1.0
    assert np.all(ball["dv_signed_3"][np.isfinite(ball["dv_signed_3"])] >= -1e-9)


def test_turn_heading_in_degrees():
    """90-degree turn at constant 8 m/s: heading_change_3 reaches ~90 (degrees, not radians)."""
    xa, ya = integrate(np.full(40, 8.0), heading_deg=0.0)
    xb, yb = integrate(np.full(40, 8.0), heading_deg=90.0, x0=xa[-1], y0=ya[-1])
    x, y = np.r_[xa, xb[1:]], np.r_[ya, yb[1:]]
    ball = clean(x, y)
    heading = ball["heading_change_3"]
    assert np.nanmax(heading) == pytest.approx(90.0, abs=1.0)
    assert np.nanmax(ball["dv_3"]) < 2.5      # central difference cuts the corner briefly
    straight = heading[:30]
    assert np.nanmax(straight) < 1e-6


def test_teleport_marked_and_neighbours_unaffected():
    """One frame 10 m away: that frame is missing, the frames around it keep their velocity."""
    x, y = integrate(np.full(100, 10.0))
    y = y.copy()
    y[50] += 10.0
    ball = clean(x, y)
    assert ball["teleport"].sum() == 1 and ball["teleport"][50]
    assert not ball["visible"][50]
    assert ball["visible"][49] and ball["visible"][51]
    assert np.isnan(ball["speed"][50])
    for t in (48, 49, 51, 52):
        assert ball["speed"][t] == pytest.approx(10.0, rel=1e-6)
        assert ball["vy"][t] == pytest.approx(0.0, abs=1e-6)
    # The gap splits the run: no velocity step or heading across it.
    assert np.isnan(ball["dv_3"][48]) and np.isnan(ball["heading_change_3"][48])
    assert ball["x_raw"][50] == pytest.approx(x[50]) and ball["y_raw"][50] == pytest.approx(y[50])


def test_fast_streak_rejected_until_plausible():
    """A streak of > 3 m/frame motion is rejected frame by frame; plausible steps after it stay."""
    speeds = np.r_[np.full(20, 5.0), np.full(6, 100.0), np.full(20, 5.0)]
    x, y = integrate(speeds)
    ball = clean(x, y)
    assert ball["teleport"].sum() == 6
    assert np.flatnonzero(ball["teleport"]).tolist() == list(range(21, 27))


def test_run_after_gap_is_not_a_jump():
    """A visible run that starts 20 m away after a gap is accepted (it is not a teleport)."""
    x, y = integrate(np.full(30, 5.0))
    x = x.copy()
    x[10:13] = np.nan
    x[13:] += 20.0
    ball = clean(x, y)
    assert not ball["teleport"].any()
    assert ball["visible"][13:].all()


def test_short_runs_dropped():
    """Visible runs shorter than BALL_MIN_RUN (3) frames are dropped."""
    x, y = integrate(np.full(40, 5.0))
    x = x.copy()
    keep = np.zeros(x.size, bool)
    keep[5:7] = True        # 2 frames: dropped
    keep[15:16] = True      # 1 frame: dropped
    keep[25:28] = True      # 3 frames: kept
    x[~keep] = np.nan
    ball = clean(x, y)
    assert np.flatnonzero(ball["visible"]).tolist() == [25, 26, 27]
    # One-sided velocity at run ends: speed defined on every kept frame.
    assert np.allclose(ball["speed"][25:28], 5.0)


def test_velocity_step_signed_and_same_run():
    speed = np.array([1.0, 1.0, 2.0, 5.0, 9.0, np.nan, 3.0, 3.0, 10.0])
    run_id = np.array([0, 0, 0, 0, 0, -1, 1, 1, 1])
    step = tracks.velocity_step(speed, 2, run_id)
    assert step[0] == pytest.approx(1.0) and step[1] == pytest.approx(4.0)
    assert step[2] == pytest.approx(7.0)
    assert np.isnan(step[3]) and np.isnan(step[4])   # crosses the gap
    assert step[6] == pytest.approx(7.0)
    assert np.isnan(step[7]) and np.isnan(step[8])   # beyond the end
    falling = tracks.velocity_step(speed[::-1], 1)
    assert falling[0] == pytest.approx(-7.0)


def test_airborne_suspect():
    """Fast ball, or a > 40% drop in 3 frames with nobody within 2 m; not with a player near."""
    fast_x, fast_y = integrate(np.full(30, 18.0))
    assert clean(fast_x, fast_y)["airborne_suspect"].all()

    speeds = np.r_[np.full(30, 10.0), 7.0, 5.0, np.full(30, 4.0)]
    x, y = integrate(speeds)
    far = clean(x, y, dmin=np.full(x.size, 5.0))
    assert far["airborne_suspect"].any()
    assert not far["airborne_fast"].any()
    near = clean(x, y, dmin=np.full(x.size, 1.0))
    assert not near["airborne_suspect"].any()
    no_dmin = clean(x, y)
    assert not no_dmin["airborne_suspect"].any()     # without dmin only the speed clause

    slow = np.r_[np.full(30, 2.0), 1.0, 0.5, np.full(30, 0.5)]   # below AIRBORNE_DECEL_MIN_SPEED
    xs, ys = integrate(slow)
    assert not clean(xs, ys, dmin=np.full(xs.size, 5.0))["airborne_suspect"].any()


# -- features ----------------------------------------------------------------------------------

def test_hull_area_unit_square_and_degenerate():
    xs = np.array([[0, 1, 1, 0, 0.5, np.nan],
                   [0, 1, 2, np.nan, np.nan, np.nan],        # collinear
                   [0, 1, np.nan, np.nan, np.nan, np.nan],   # 2 points
                   [0, 0, 1, 1, 0, 1]], dtype=float)          # duplicates
    ys = np.array([[0, 0, 1, 1, 0.5, np.nan],
                   [0, 1, 2, np.nan, np.nan, np.nan],
                   [0, 1, np.nan, np.nan, np.nan, np.nan],
                   [0, 0, 0, 1, 1, 1]], dtype=float)
    area = tracks.hull_area(xs, ys)
    assert area[0] == pytest.approx(1.0)
    assert area[1] == pytest.approx(0.0, abs=1e-12)
    assert area[2] == 0.0
    assert area[3] == pytest.approx(1.0)


def _monotone_chain_area(points):
    """Reference hull area (Andrew's monotone chain + shoelace) in plain Python."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return 0.5 * abs(sum(hull[i][0] * hull[i - 1][1] - hull[i - 1][0] * hull[i][1]
                         for i in range(len(hull))))


def test_hull_area_matches_monotone_chain():
    rng = np.random.RandomState(0)
    n_rows, n_pts = 300, 14
    xs = rng.uniform(0, 100, (n_rows, n_pts))
    ys = rng.uniform(0, 60, (n_rows, n_pts))
    xs[rng.uniform(size=xs.shape) < 0.3] = np.nan
    xs[:20] = np.round(xs[:20] / 10) * 10      # lattice rows: collinear and duplicate points
    ys[:20] = np.round(ys[:20] / 10) * 10
    area = tracks.hull_area(xs, ys)
    for i in range(n_rows):
        ok = np.isfinite(xs[i])
        ref = _monotone_chain_area(list(zip(xs[i, ok], ys[i, ok])))
        assert area[i] == pytest.approx(ref, rel=1e-9, abs=1e-9)


def test_frame_features_hand_built_frame():
    """Centroids, spread, sep, nearest distances, counts, d_second, relT on a hand frame."""
    # slots 0-2 home, 3-4 away; 6 frames, players still except slot 0 moving at 2 m/s in x.
    n = 6
    team = [0, 0, 0, 1, 1]
    px = np.zeros((n, 5))
    py = np.zeros((n, 5))
    base_x = [50.0, 40.0, 40.0, 60.0, 70.0]
    base_y = [25.0, 20.0, 30.0, 25.0, 25.0]
    for s in range(5):
        px[:, s] = base_x[s]
        py[:, s] = base_y[s]
    px[:, 0] = 50.0 + 2.0 * np.arange(n) / FPS
    ball_x = px[:, 0] + 1.0
    ball_y = np.full(n, 25.0)
    ps = np.zeros((n, 5))
    ps[:, 0] = np.nan                       # no ATD speed: fall back to position-derived
    tac = tactical_from(px, py, team, ball_x, ball_y, ps=ps)
    ball = tracks.clean_ball(tac["bx"], tac["by"], PITCH, CFG, fps=FPS)
    feats = tracks.frame_features(tac, ball, PITCH, CFG, fps=FPS)

    f = 3
    assert feats["dH"][f] == pytest.approx(1.0, abs=1e-4)
    assert feats["nearest_slot_H"][f] == 0
    assert feats["nH_15"][f] == 1 and feats["nH_30"][f] == 1
    expected_dA = 60.0 - ball_x[f]
    assert feats["dA"][f] == pytest.approx(expected_dA, abs=1e-3)
    assert feats["nearest_slot_A"][f] == 3
    assert feats["nA_15"][f] == 0
    assert feats["d_second"][f] == pytest.approx(expected_dA, abs=1e-3)
    assert feats["dmin"][f] == pytest.approx(1.0, abs=1e-4)
    assert feats["nearest_team"][f] == 0
    # Ball and slot 0 both move at 2 m/s in x: relative speed ~0; vH_near from positions.
    assert feats["relH"][f] == pytest.approx(0.0, abs=0.05)
    assert feats["vH_near"][f] == pytest.approx(2.0, abs=0.05)
    assert feats["cxH"][f] == pytest.approx((px[f, 0] + 80.0) / 3, abs=1e-3)
    assert feats["cyH"][f] == pytest.approx(25.0, abs=1e-3)
    assert feats["cxA"][f] == pytest.approx(65.0, abs=1e-3)
    assert feats["sxA"][f] == pytest.approx(5.0, abs=1e-3)
    assert feats["syA"][f] == pytest.approx(0.0, abs=1e-3)
    assert feats["sep"][f] == pytest.approx(abs(feats["cxH"][f] - 65.0), abs=1e-6)
    assert feats["nvisH"][f] == 3 and feats["nvisA"][f] == 2 and feats["n_tracks"][f] == 5
    assert feats["hullA"][f] == 0.0
    assert feats["hullH"][f] == pytest.approx(0.5 * (px[f, 0] - 40.0) * 10.0, rel=1e-4)
    assert feats["meanvA"][f] == pytest.approx(0.0)


def test_frame_features_meanv_near():
    """meanv_near = mean speed of the PURSUIT_N_NEAREST players nearest the ball (P1-B)."""
    n = 6
    team = [0, 0, 0, 1, 1]
    px = np.tile([50.0, 40.0, 40.0, 60.0, 70.0], (n, 1))
    py = np.tile([25.0, 20.0, 30.0, 25.0, 25.0], (n, 1))
    ps = np.tile([2.0, 4.0, 4.0, 0.0, 6.0], (n, 1))
    tac = tactical_from(px, py, team, np.full(n, 51.0), np.full(n, 25.0), ps=ps)
    ball = tracks.clean_ball(tac["bx"], tac["by"], PITCH, CFG, fps=FPS)
    two = dataclasses.replace(CFG, PURSUIT_N_NEAREST=2)
    feats = tracks.frame_features(tac, ball, PITCH, two, fps=FPS)
    assert feats["meanv_near"][2] == pytest.approx(1.0)        # slot 0 (1 m) and slot 3 (9 m)
    feats = tracks.frame_features(tac, ball, PITCH, CFG, fps=FPS)   # 6 > 5 players: all of them
    assert feats["meanv_near"][2] == pytest.approx(16.0 / 5)
    tac = tactical_from(px, py, team, np.full(n, np.nan), np.full(n, np.nan), ps=ps)
    ball = tracks.clean_ball(tac["bx"], tac["by"], PITCH, CFG, fps=FPS)
    assert np.isnan(tracks.frame_features(tac, ball, PITCH, CFG, fps=FPS)["meanv_near"]).all()


def test_frame_features_missing_ball_and_team():
    n = 4
    px = np.full((n, 3), np.nan)
    py = np.full((n, 3), np.nan)
    px[:, 0], py[:, 0] = 10.0, 10.0          # one home player; away team absent
    tac = tactical_from(px, py, [0, 1, 1], np.full(n, 12.0), np.full(n, 10.0))
    tac["bx"][1] = np.nan
    tac["by"][1] = np.nan
    ball = tracks.clean_ball(tac["bx"], tac["by"], PITCH, CFG, fps=FPS)
    feats = tracks.frame_features(tac, ball, PITCH, CFG, fps=FPS)
    # frame 1 missing raw, so frames 0 and 2-3 are short runs (< 3): ball never visible
    assert not ball["visible"].any()
    assert np.isnan(feats["dH"]).all() and np.isnan(feats["dA"]).all()
    assert (feats["nearest_slot_H"] == -1).all() and (feats["nearest_slot_A"] == -1).all()
    assert (feats["nearest_team"] == -1).all()
    assert np.isnan(feats["d_second"]).all()
    assert np.isnan(feats["cxA"]).all() and (feats["nvisA"] == 0).all()
    assert np.isnan(feats["sep"]).all()
    assert feats["cxH"][0] == pytest.approx(10.0, abs=1e-4)


# -- slot confidence ---------------------------------------------------------------------------

def test_slot_confidence_overcount_halves_team():
    """A frame with 12 visible home tracks halves the home slots' confidence in that frame."""
    n, n_home, n_away = 200, 12, 11
    team = [0] * n_home + [1] * n_away
    px = np.tile(np.linspace(5, 95, n_home + n_away), (n, 1))
    py = np.full((n, n_home + n_away), 30.0)
    px[:, 11] = np.nan                    # 12th home slot hidden except on frame 100
    px[100, 11] = 50.0
    conf = tracks.slot_confidence(tactical_from(px, py, team), CFG)
    assert conf.dtype == np.float32 and conf.shape == (n, n_home + n_away)
    assert np.all(conf[100, :11] == 0.5)
    assert np.all(conf[100, n_home:] == 1.0)       # away unaffected
    assert np.all(conf[99, :11] == 1.0) and np.all(conf[101, :11] == 1.0)
    # The 1-frame tracklet of slot 11 is also short: 0.5 x 0.5.
    assert conf[100, 11] == pytest.approx(0.25)


def test_slot_confidence_short_tracklet():
    n = 200
    px = np.full((n, 2), 20.0)
    py = np.full((n, 2), 30.0)
    px[:, 1] = np.nan
    px[10:59, 1] = 30.0                    # 49 frames: short
    px[100:150, 1] = 30.0                  # 50 frames: not short
    conf = tracks.slot_confidence(tactical_from(px, py, [0, 1]), CFG)
    assert np.all(conf[10:59, 1] == 0.5)
    assert np.all(conf[100:150, 1] == 1.0)
    assert np.all(conf[:, 0] == 1.0)


# -- goalkeeper slot-swap guard (C9) -----------------------------------------------------------

REAL_PITCH = (105.3, 68.0)          # the pitch of games.py, so the half-way line is at 52.65 m
GK_N = 2000                         # frames
GK_BLOCKS = [(1, 0, GK_N - 1)]
SWAP = slice(1000, 1100)            # the frames where a slot can carry the wrong body
FROZEN = slice(950, 1150)           # the frames where slot 11 can be frozen


def gk_scene(slot0=None, slot11=None, slot11_speed=None, slot3=None, slot0_in_box=GK_N):
    """Synthetic match for the C9 guard on the real pitch: H slots 0-10, A slots 11-21.

    H defends x = 0. Slot 0 (H) and slot 11 (A) stay in their own penalty areas and move. The
    outfield players stay between the boxes. The arguments change the scene:
      slot0         (x, y) of slot 0 on SWAP, moving at 2 m/s
      slot11        (x, y) of slot 11 on FROZEN, still with NaN speed (the frozen artefact)
      slot11_speed  with slot11: slot 11 moves along y at this speed (m/s) instead
      slot3         (x, y) of the H outfield slot 3 on SWAP, moving at 2 m/s
      slot0_in_box  slot 0 is visible only on its first frames, this many of them in its box
    The ball is at (95.5, 34) on SWAP and missing elsewhere.
    """
    t = np.arange(GK_N)
    wave = np.sin(t / 20.0)
    px = np.zeros((GK_N, 22))
    py = np.zeros((GK_N, 22))
    ps = np.full((GK_N, 22), 1.0)
    px[:, 0], py[:, 0] = 5.0 + 0.5 * wave, 34.0
    px[:, 11], py[:, 11] = 100.0 - 0.5 * wave, 34.0
    for i in range(1, 11):
        px[:, i], py[:, i] = 20.0 + 3.0 * i + wave, 6.0 * i
        px[:, 11 + i], py[:, 11 + i] = 55.0 + 3.0 * i - wave, 6.0 * i
    if slot0 is not None:
        px[SWAP, 0], py[SWAP, 0], ps[SWAP, 0] = slot0[0], slot0[1], 2.0
    if slot11 is not None:
        px[FROZEN, 11], py[FROZEN, 11] = slot11
        if slot11_speed is None:
            ps[FROZEN, 11] = np.nan
        else:
            steps = np.arange(FROZEN.stop - FROZEN.start)
            py[FROZEN, 11] = slot11[1] + slot11_speed * steps / FPS
            ps[FROZEN, 11] = slot11_speed
    if slot3 is not None:
        px[SWAP, 3], py[SWAP, 3], ps[SWAP, 3] = slot3[0], slot3[1], 2.0
    if slot0_in_box < GK_N:
        px[slot0_in_box:, 0] = np.nan
        py[slot0_in_box:, 0] = np.nan
    ball_x = np.full(GK_N, np.nan)
    ball_y = np.full(GK_N, np.nan)
    ball_x[SWAP], ball_y[SWAP] = 95.5, 34.0
    team = [0] * 11 + [1] * 11
    return {
        "px": (px / REAL_PITCH[0]).astype(np.float32),
        "py": (py / REAL_PITCH[1]).astype(np.float32),
        "ps": ps.astype(np.float32),
        "bx": (ball_x / REAL_PITCH[0]).astype(np.float32),
        "by": (ball_y / REAL_PITCH[1]).astype(np.float32),
        "team_of_slot": np.asarray(team, dtype=np.int8),
        "n_frames": GK_N, "n_slots": 22, "fps": int(FPS),
    }


def guard(tac):
    return tracks.guard_goalkeeper_swaps(tac, GK_BLOCKS, CFG, pitch=REAL_PITCH)


def test_goalkeeper_slots_own_box_rule():
    """Slot 0 is H's goalkeeper-like slot (share 0.95), slot 11 is A's (share 1.0)."""
    tac = gk_scene(slot0=(95.0, 34.0), slot11=(98.0, 36.0))
    keepers = tracks.goalkeeper_slots(tac, GK_BLOCKS, REAL_PITCH, CFG)
    assert keepers[(1, 0)]["slot"] == 0 and keepers[(1, 1)]["slot"] == 11
    assert keepers[(1, 0)]["own_goal_x"] == 0.0
    assert keepers[(1, 1)]["own_goal_x"] == REAL_PITCH[0]
    assert keepers[(1, 0)]["share"] == pytest.approx(0.95)
    assert keepers[(1, 0)]["own_box"] == 1900
    assert keepers[(1, 1)]["share"] == pytest.approx(1.0)


def test_goalkeeper_slots_need_enough_own_box_frames():
    """Slot 0 is in its box on 1000 frames only (< GK_MIN_OWN_BOX_FRAMES): H has no keeper."""
    tac = gk_scene(slot0=(95.0, 34.0), slot11=(98.0, 36.0), slot0_in_box=1000)
    keepers = tracks.goalkeeper_slots(tac, GK_BLOCKS, REAL_PITCH, CFG)
    assert keepers[(1, 0)]["slot"] == -1 and keepers[(1, 0)]["own_box"] == 1000
    assert keepers[(1, 1)]["slot"] == 11
    _, swaps, _ = guard(tac)
    assert swaps.shape == (0, 3)


def test_frozen_mask_needs_nan_speed_and_no_move():
    tac = gk_scene(slot11=(98.0, 36.0))
    frozen = tracks.frozen_mask(tac, REAL_PITCH, CFG)
    half = CFG.FROZEN_WIN // 2
    expected = np.zeros(GK_N, dtype=bool)
    expected[FROZEN.start + half:FROZEN.stop - half] = True
    assert np.array_equal(frozen[:, 11], expected)
    assert not frozen[:, :11].any() and not frozen[:, 12:].any()


def test_guard_moves_the_live_keeper_into_the_frozen_slot():
    """H slot 0 plays in the A box next to frozen A slot 11: the body is the A goalkeeper."""
    tac = gk_scene(slot0=(95.0, 34.0), slot11=(98.0, 36.0))
    raw = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tac.items()}
    patched, swaps, _ = guard(tac)
    assert np.array_equal(swaps[:, 0], np.arange(SWAP.start, SWAP.stop))
    assert np.all(swaps[:, 1] == 0) and np.all(swaps[:, 2] == 11)
    # Slot 11 holds the positions and speed of slot 0 on the swap frames. Slot 0 is missing.
    assert np.array_equal(patched["px"][SWAP, 11], tac["px"][SWAP, 0])
    assert np.array_equal(patched["py"][SWAP, 11], tac["py"][SWAP, 0])
    assert np.array_equal(patched["ps"][SWAP, 11], tac["ps"][SWAP, 0])
    assert np.isnan(patched["px"][SWAP, 0]).all() and np.isnan(patched["ps"][SWAP, 0]).all()
    # Every other frame and slot is unchanged, and the input dict is not changed.
    other = np.ones(GK_N, dtype=bool)
    other[SWAP] = False
    assert np.array_equal(patched["px"][other], tac["px"][other], equal_nan=True)
    assert np.array_equal(patched["px"][:, 1:11], tac["px"][:, 1:11], equal_nan=True)
    for key in ("px", "py", "ps"):
        assert np.array_equal(tac[key], raw[key], equal_nan=True)
    # The nearest player to the ball at (95.5, 34) is now on A.
    before = tracks.prepare(tac, CFG, pitch=REAL_PITCH)[1]
    after = tracks.prepare(patched, CFG, pitch=REAL_PITCH)[1]
    assert np.all(before["nearest_team"][SWAP] == 0)
    assert np.all(after["nearest_team"][SWAP] == 1)
    assert np.all(after["nearest_slot_A"][SWAP] == 11)
    assert np.all(after["nvisH"][SWAP] == 10) and np.all(after["nvisA"][SWAP] == 11)


def test_guard_off_when_the_keeper_is_in_its_own_half():
    tac = gk_scene(slot0=(50.0, 34.0), slot11=(53.0, 36.0))
    assert tracks.goalkeeper_slots(tac, GK_BLOCKS, REAL_PITCH, CFG)[(1, 1)]["slot"] == 11
    _, swaps, _ = guard(tac)
    assert swaps.shape == (0, 3)


def test_guard_off_when_the_other_keeper_moves():
    tac = gk_scene(slot0=(95.0, 34.0), slot11=(98.0, 30.0), slot11_speed=1.5)
    assert not tracks.frozen_mask(tac, REAL_PITCH, CFG)[:, 11].any()
    _, swaps, _ = guard(tac)
    assert swaps.shape == (0, 3)


def test_guard_ignores_an_outfield_slot():
    """H outfield slot 3 beyond half-way next to frozen slot 11: only keeper slots swap."""
    tac = gk_scene(slot11=(98.0, 36.0), slot3=(95.0, 34.0))
    keepers = tracks.goalkeeper_slots(tac, GK_BLOCKS, REAL_PITCH, CFG)
    assert keepers[(1, 0)]["slot"] == 0 and keepers[(1, 1)]["slot"] == 11
    patched, swaps, _ = guard(tac)
    assert swaps.shape == (0, 3)
    assert np.array_equal(patched["px"], tac["px"], equal_nan=True)


def test_guard_off_for_an_upfield_keeper_and_two_close_outfield_players():
    """Neither a keeper upfield without a frozen partner nor two close outfield players swap.

    CODE-REVIEW-RESPONSE finding 7. H keeper slot 0 plays at (95, 34), beyond half-way and
    5 m from A keeper slot 11, which moves (not frozen). H outfield slot 3 stands 0.6 m from
    A outfield slot 12 beyond half-way, and slot 12 is frozen (NaN speed, no move).
    """
    tac = gk_scene(slot0=(95.0, 34.0))
    px = tac["px"].astype(np.float64) * REAL_PITCH[0]
    py = tac["py"].astype(np.float64) * REAL_PITCH[1]
    ps = tac["ps"].astype(np.float64)
    px[FROZEN, 12], py[FROZEN, 12], ps[FROZEN, 12] = 80.0, 20.0, np.nan
    px[SWAP, 3], py[SWAP, 3], ps[SWAP, 3] = 80.6, 20.0, 2.0
    tac.update(px=(px / REAL_PITCH[0]).astype(np.float32),
               py=(py / REAL_PITCH[1]).astype(np.float32), ps=ps.astype(np.float32))
    keepers = tracks.goalkeeper_slots(tac, GK_BLOCKS, REAL_PITCH, CFG)
    assert keepers[(1, 0)]["slot"] == 0 and keepers[(1, 1)]["slot"] == 11
    frozen = tracks.frozen_mask(tac, REAL_PITCH, CFG)
    assert not frozen[:, 11].any() and frozen[SWAP, 12].all()
    dist = np.hypot(px[SWAP, 0] - px[SWAP, 11], py[SWAP, 0] - py[SWAP, 11])
    assert np.all(dist <= CFG.GK_SWAP_R) and np.all(px[SWAP, 0] > REAL_PITCH[0] / 2.0)
    patched, swaps, _ = guard(tac)
    assert swaps.shape == (0, 3)
    for key in ("px", "py", "ps"):
        assert np.array_equal(patched[key], tac[key], equal_nan=True)


def test_guard_deterministic():
    tac = gk_scene(slot0=(95.0, 34.0), slot11=(98.0, 36.0))
    first, swaps_1, keepers_1 = guard(tac)
    second, swaps_2, keepers_2 = guard(tac)
    assert np.array_equal(swaps_1, swaps_2) and keepers_1 == keepers_2
    for key in ("px", "py", "ps"):
        assert np.array_equal(first[key], second[key], equal_nan=True)


def test_prepare_uses_full_airborne_rule():
    n = 60
    speeds = np.r_[np.full(30, 10.0), 7.0, 5.0, np.full(n - 33, 4.0)]
    bx_m, by_m = integrate(speeds)
    px = np.full((n, 2), np.nan)
    py = np.full((n, 2), np.nan)
    px[:, 0], py[:, 0] = 5.0, 5.0
    px[:, 1], py[:, 1] = 95.0, 45.0
    tac = tactical_from(px, py, [0, 1], bx_m, by_m)
    ball, feats, conf = tracks.prepare(tac, CFG, pitch=PITCH)
    assert ball["airborne_suspect"].any()
    assert np.array_equal(ball["airborne_suspect"], feats["airborne_suspect"])
    assert conf.shape == (n, 2)


# -- config ------------------------------------------------------------------------------------

def test_config_hash_stable_and_sensitive(tmp_path):
    cfg = Config.default()
    assert cfg.config_hash() == Config.default().config_hash()
    assert len(cfg.config_hash()) == 10
    changed = dataclasses.replace(cfg, DV_TOUCH=2.0)
    assert changed.config_hash() != cfg.config_hash()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.DV_TOUCH = 2.0
    path = tmp_path / "cfg.json"
    path.write_text('{"DV_TOUCH": 2}')
    loaded = load(str(path))
    assert loaded.DV_TOUCH == 2.0 and isinstance(loaded.DV_TOUCH, float)
    assert loaded.config_hash() == changed.config_hash()
    path.write_text('{"DV_TUOCH": 2}')
    with pytest.raises(ValueError):
        load(str(path))


def test_config_card_constants():
    """The values the task cards name (P0-B, P1-A, P1-B, P1-C, P2)."""
    cfg = Config.default()
    assert (cfg.BALL_MAX_JUMP_M, cfg.BALL_MIN_RUN, cfg.DV_K, cfg.AIRBORNE_SPEED) == (3.0, 3, 3, 15.0)
    assert cfg.MIN_TRACKLET_FRAMES == 50 and cfg.MAX_TEAM_TRACKS == 11
    # DV_TOUCH starts at 1.5 on the P1-A card. P1-A sets it to 1.0 from the measured touch rate.
    # CONTROL_FRAMES 8: the 8 Nov protocol (grid-edge rule, grid centre), was the card value 5.
    assert (cfg.DV_TOUCH, cfg.R_TOUCH, cfg.DV_RELEASE, cfg.R_CONTROL, cfg.CONTROL_FRAMES) == (
        1.0, 1.5, 3.0, 1.2, 8)
    assert (cfg.V_STOP, cfg.SEP_DEAD, cfg.DV_RESTART, cfg.RESTART_K) == (1.6, 13.0, 3.0, 9)
    assert (cfg.C_SWITCH, cfg.C_SWITCH_EVENT, cfg.MIN_POSS, cfg.MIN_DEAD) == (8.0, 0.5, 10, 25)
    assert cfg.POSSESSION_RULE == "release_or_control" and cfg.FLIGHT_ASSIGNMENT is False
    # R1 removes P2's public-derived reappearance offset (calibration/CANDIDATE_B.md).
    # C11 keeps the kick offset at 12 from the bump-centre measurement M16.
    assert (cfg.DELTA_SWITCH, cfg.DELTA_DEAD, cfg.DELTA_RESTART_KICK,
            cfg.DELTA_RESTART_REAPPEAR) == (0, 0, 12, 0)
    assert cfg.DELTA_OUT_LAG == 0
    assert cfg.RESTART_CHAIN_FRAMES == 87

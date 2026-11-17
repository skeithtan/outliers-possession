"""Synthetic tests for the K1 kink contact point (engine/events.py, CONTACT_POINT "kink_min").

The scenes follow tests/test_events_reach.py. The ball rests at (10, 10) m, is kicked at frame 40
and turns 60 degrees at frame 100, at 14.4 m/s. The velocity is smoothed with a Gaussian kernel
of sigma 9.5 frames (tests/test_events.py smooth_path), the kernel of RESEARCH-2 M13. The smoothed
path then never comes closer than about 2.2 m to the corner where the ball turned. An H player
stands on the corner and plays the ball. An A player stands inside the turn, near the smoothed
path.
"""

import dataclasses

import numpy as np
import pytest

from engine import events, tracks
from tests.test_events import A, BUMP, FAST, FPS, H, PITCH, far, piecewise, smooth_path, static

N = 200
KICK = 40
TURN = 100
KINK = dataclasses.replace(BUMP, CONTACT_POINT="kink_min")
WINDOW = dataclasses.replace(BUMP, CONTACT_POINT="window")   # the C8 rule, c21_compliance


def true_path(n=N, kick=KICK, turn=TURN, angle=60.0, x0=10.0, y0=10.0):
    """Ball positions (m) of the unsmoothed path, the same integration as smooth_path."""
    speeds = piecewise(n, [(kick, FAST)])
    heading = np.radians(piecewise(n, [(turn, angle)]))
    x = x0 + np.concatenate(([0.0], np.cumsum((speeds * np.cos(heading))[:-1]) / FPS))
    y = y0 + np.concatenate(([0.0], np.cumsum((speeds * np.sin(heading))[:-1]) / FPS))
    return x, y


def scene(inside_m=0.8, smooth=True, h_on_corner=True, h_kicker=False):
    """(ball, feats, slot_conf, players) of the turn scene.

    The A player stands `inside_m` from the smoothed ball at TURN, on the inner side of the turn.
    With smooth=False the ball follows the unsmoothed path. With h_kicker an H player stands on
    the kick spot, so H owns the ball when it turns.
    """
    if smooth:
        bx, by = smooth_path(piecewise(N, [(KICK, FAST)]), piecewise(N, [(TURN, 60.0)]))
    else:
        bx, by = true_path()
    tx, ty = true_path()
    corner = np.array([tx[TURN], ty[TURN]])
    inward = np.array([bx[TURN], by[TURN]]) - corner
    norm = np.hypot(*inward)
    inward = inward / norm if norm > 0 else np.array([-0.5, np.sqrt(3.0) / 2.0])
    spot = np.array([bx[TURN], by[TURN]]) + inside_m * inward
    players = [static(A, spot[0], spot[1], N), far(H, N)]
    if h_on_corner:
        players.insert(0, static(H, corner[0], corner[1], N))
    if h_kicker:
        players.append(static(H, 10.0, 9.6, N))
    px = np.column_stack([p[1] for p in players]).astype(np.float64)
    py = np.column_stack([p[2] for p in players]).astype(np.float64)
    tactical = {"px": px / PITCH[0], "py": py / PITCH[1],
                "team_of_slot": np.array([p[0] for p in players], dtype=np.int8),
                "ps": np.full(px.shape, np.nan), "fps": FPS}
    ball = tracks.clean_ball(np.asarray(bx) / PITCH[0], np.asarray(by) / PITCH[1], PITCH, BUMP,
                             fps=FPS)
    feats = tracks.frame_features(tactical, ball, PITCH, BUMP, fps=FPS)
    ball["airborne_suspect"] = feats["airborne_suspect"].copy()
    px_m, py_m, visible = tracks._player_metres(tactical, PITCH)
    return ball, feats, tracks.slot_confidence(tactical, BUMP), (px_m, py_m, visible,
                                                                  tactical["team_of_slot"])


def centres(ball, cfg=BUMP):
    h = int(cfg.CONTACT_HALF_WINDOW)
    size = events.impulse(ball, h)
    with np.errstate(invalid="ignore"):
        strong = np.isfinite(size) & (size >= cfg.DV_CONTACT)
    return events.window_maxima(size, h, np.flatnonzero(strong))


def contacts(sc, cfg):
    ball, feats, slot_conf, players = sc
    return events._bump_contacts(ball, feats, slot_conf, np.ones(N, dtype=bool), cfg,
                                 players=players)


def at_turn(rows):
    return [r for r in rows if abs(int(r[0]) - TURN) <= 3]


def test_gauss_ramp_is_the_smoothed_hinge():
    u = np.linspace(-40.0, 40.0, 161)
    for sigma in (1.0, 3.0, 9.5):
        g = events.gauss_ramp(u, sigma)
        assert np.allclose(g - events.gauss_ramp(-u, sigma), u)          # G(u) - G(-u) = u
        assert np.isclose(events.gauss_ramp(0.0, sigma), sigma / np.sqrt(2.0 * np.pi))
        assert np.all(g >= np.maximum(u, 0.0) - 1e-12)                   # above the hinge
    # Sharp limit: the largest gap is sigma / sqrt(2 pi), at u = 0.
    assert np.allclose(events.gauss_ramp(u, 1e-3), np.maximum(u, 0.0), atol=1e-3)


def test_smoothed_path_misses_the_corner_by_more_than_the_touch_radius():
    ball = scene()[0]
    tx, ty = true_path()
    lo, hi = TURN - BUMP.CONTACT_HALF_WINDOW, TURN + BUMP.CONTACT_HALF_WINDOW + 1
    gap = np.hypot(ball["x"][lo:hi] - tx[TURN], ball["y"][lo:hi] - ty[TURN])
    assert gap.min() > BUMP.R_TOUCH


def test_kink_points_recover_the_corner_and_the_kick_spot():
    ball = scene()[0]
    found = centres(ball)
    assert [int(c) for c in found] == [KICK, TURN]
    xc, yc, tau = events.kink_points(ball, found, KINK)
    tx, ty = true_path()
    assert np.hypot(xc[1] - tx[TURN], yc[1] - ty[TURN]) < 0.2
    assert abs(tau[1] - TURN) <= 1.0
    assert np.hypot(xc[0] - 10.0, yc[0] - 10.0) < 0.2 and abs(tau[0] - KICK) <= 1.0


def test_window_rule_gives_the_turn_to_the_player_inside_it():
    rows = at_turn(contacts(scene(), WINDOW))
    assert [(r[1], r[2]) for r in rows] == [("redirect", "A")]


def test_kink_min_gives_the_turn_to_the_player_on_the_corner():
    rows = at_turn(contacts(scene(), KINK))
    assert [(r[1], r[2]) for r in rows] == [("redirect", "H")]
    assert rows[0][4] < 0.2                      # dist_m is the distance to the contact point


def test_kink_min_finds_a_toucher_that_the_window_rule_cannot_reach():
    sc = scene(inside_m=5.0)                     # A far from the ball: H is the only candidate
    assert at_turn(contacts(sc, WINDOW)) == []     # 2.2 m from the smoothed path: no row
    assert [(r[1], r[2]) for r in at_turn(contacts(sc, KINK))] == [("redirect", "H")]


def test_kink_min_keeps_the_window_rule_on_a_sharp_path():
    sc = scene(inside_m=0.8, smooth=False)
    assert contacts(sc, KINK) == contacts(sc, WINDOW)


def test_kink_min_never_adds_distance():
    sc = scene()
    window, kink = contacts(sc, WINDOW), contacts(sc, KINK)
    by_frame = dict((r[0], r[4]) for r in window)
    for row in kink:
        if row[0] in by_frame:
            assert row[4] <= by_frame[row[0]] + 1e-12


def test_without_players_kink_min_is_the_window_rule():
    ball, feats, slot_conf, _ = scene()
    alive = np.ones(N, dtype=bool)
    plain = events._bump_contacts(ball, feats, slot_conf, alive, KINK)
    assert plain == events._bump_contacts(ball, feats, slot_conf, alive, WINDOW)


def test_detect_events_passes_players_through():
    # H kicks and owns the ball. Under the window rule the turn is an A redirect that nothing
    # confirms, so it has no row (C8). Under kink_min it is a redirect by the owner: a touch.
    ball, feats, slot_conf, players = scene(h_kicker=True)
    window = events.detect_events(ball, feats, None, slot_conf, WINDOW, players=players)
    kink = events.detect_events(ball, feats, None, slot_conf, KINK, players=players)
    turn_w = window[(window["frame"] - TURN).abs() <= 3]
    turn_k = kink[(kink["frame"] - TURN).abs() <= 3]
    assert turn_w.empty
    assert turn_k[["kind", "team", "notes"]].values.tolist() == [["touch", "H", "redirect"]]
    kick = kink[(kink["frame"] - KICK).abs() <= 3]
    assert kick[["kind", "team"]].values.tolist() == [["release", "H"]]
    assert events.detect_events(ball, feats, None, slot_conf, KINK).equals(
        events.detect_events(ball, feats, None, slot_conf, WINDOW))


def test_kink_points_need_both_sides_and_enough_frames():
    ball = scene()[0]
    short = dict(ball)
    run_id = np.array(ball["run_id"], copy=True)
    run_id[TURN + 2:] = -1                       # the ball vanishes 2 frames after the turn
    short["run_id"] = run_id
    xc, yc, tau = events.kink_points(short, np.array([TURN]), KINK)
    assert np.isnan(xc[0]) and np.isnan(yc[0]) and np.isnan(tau[0])
    empty = events.kink_points(ball, np.zeros(0, dtype=np.int64), KINK)
    assert all(v.shape == (0,) for v in empty)


def test_default_is_kink_min():
    assert BUMP.CONTACT_POINT == "kink_min"


def test_unknown_contact_point_rule_raises():
    ball, feats, slot_conf, players = scene()
    bad = dataclasses.replace(WINDOW, CONTACT_POINT="nearest")
    with pytest.raises(ValueError):
        events._bump_contacts(ball, feats, slot_conf, np.ones(N, dtype=bool), bad,
                              players=players)


def test_kink_contacts_are_deterministic():
    sc = scene()
    assert contacts(sc, KINK) == contacts(sc, KINK)
    first = events.kink_points(sc[0], centres(sc[0]), KINK)
    second = events.kink_points(sc[0], centres(sc[0]), KINK)
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(first, second))


def test_kink_points_do_not_depend_on_where_the_scene_is():
    # Far from the origin, with the ball lost 5 frames after the turn (a one-sided fit, as at
    # the edge of a ball gap), the fit gives the same point, moved, and the same tau.
    ball = scene()[0]
    found = centres(ball)
    cut = dict(ball)
    run_id = np.array(ball["run_id"], copy=True)
    run_id[TURN + 5:] = -1
    cut["run_id"] = run_id
    far_scene = dict(cut)
    far_scene["x"] = np.asarray(cut["x"], dtype=np.float64) + 55.0
    far_scene["y"] = np.asarray(cut["y"], dtype=np.float64) + 50.0
    near_x, near_y, near_tau = events.kink_points(cut, found, KINK)
    far_x, far_y, far_tau = events.kink_points(far_scene, found, KINK)
    assert np.isfinite(near_x[1])
    assert np.array_equal(near_tau, far_tau, equal_nan=True)
    assert np.allclose(far_x - 55.0, near_x, atol=2e-6, equal_nan=True)
    assert np.allclose(far_y - 50.0, near_y, atol=2e-6, equal_nan=True)
    tx, ty = true_path()
    assert np.hypot(near_x[1] - tx[TURN], near_y[1] - ty[TURN]) < 0.2

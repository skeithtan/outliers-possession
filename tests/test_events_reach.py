"""Synthetic tests for the C18 reach contact (engine/events.py, _bump_contacts).

The scenes follow tests/test_events_shape.py. H kicks at frame 40. At frame 100 the ball turns
60 degrees with no player of either team within R_TOUCH. An A player stands outside the turn,
beyond R_TOUCH. Six more players of each team stand away from the ball. In the "response"
scenes both teams run in +x from frame 110: A goes forward and H goes back, so the shape
response of A is about 2.4 m/s.
"""

import dataclasses

import numpy as np

from engine import events, tracks
from tests.test_events import A, BUMP, FAST, FPS, H, PITCH, far, piecewise, smooth_path, static
from tests.test_events_shape import N, REDIRECT, contact_rows, runner, shape_of

OFF = dataclasses.replace(BUMP, REACH_SHAPE_CONFIRM=False)
OUTSIDE = np.array([0.8, -2.0]) / np.hypot(0.8, -2.0)   # a unit step outside the turn


def scene(reach_m=2.4, response=True, owner_near=False):
    """(ball, feats, slot_conf, tactical) with an A player `reach_m` from the ball at REDIRECT.

    With `owner_near` an H player stands 0.5 m from the ball at REDIRECT.
    """
    bx, by = smooth_path(piecewise(N, [(40, FAST)]), piecewise(N, [(REDIRECT, 60.0)]))
    speed = 1.5 if response else 0.0
    spot = np.array([bx[REDIRECT], by[REDIRECT]]) + reach_m * OUTSIDE
    players = [static(H, 10.0, 10.3, N), static(A, spot[0], spot[1], N), far(H, N)]
    if owner_near:
        players.append(static(H, bx[REDIRECT] + 0.25, by[REDIRECT] - 0.4, N))
    players += [runner(A, 5.0 + 6.0 * i, 2.0, speed=speed) for i in range(6)]
    players += [runner(H, 70.0 + 5.0 * i, 2.0, speed=speed) for i in range(6)]
    px = np.column_stack([p[1] for p in players]).astype(np.float64)
    py = np.column_stack([p[2] for p in players]).astype(np.float64)
    tactical = {"px": px / PITCH[0], "py": py / PITCH[1],
                "team_of_slot": np.array([p[0] for p in players], dtype=np.int8),
                "ps": np.full(px.shape, np.nan), "fps": FPS}
    ball = tracks.clean_ball(np.asarray(bx) / PITCH[0], np.asarray(by) / PITCH[1], PITCH, BUMP,
                             fps=FPS)
    feats = tracks.frame_features(tactical, ball, PITCH, BUMP, fps=FPS)
    ball["airborne_suspect"] = feats["airborne_suspect"].copy()
    return ball, feats, tracks.slot_confidence(tactical, BUMP), tactical


def detect(sc, cfg=BUMP, shape="auto"):
    ball, feats, slot_conf, tactical = sc
    if isinstance(shape, str):
        shape = shape_of(tactical, cfg)
    return events.detect_events(ball, feats, None, slot_conf, cfg, shape=shape)


def near_redirect(table):
    rows = contact_rows(table)
    return rows[(rows["frame"] - REDIRECT).abs() <= 3]


def window_distance(sc, frame, team="A"):
    """The team's smallest ball distance over [frame - h, frame + h] (the _bump_contacts rule)."""
    h = int(BUMP.CONTACT_HALF_WINDOW)
    key = "dA" if team == "A" else "dH"
    return float(np.min(sc[1][key][frame - h:frame + h + 1]))


def test_reach_contact_with_a_shape_response_is_a_row_of_the_reach_team():
    sc = scene(reach_m=2.4, response=True)
    rows = near_redirect(detect(sc))
    assert len(rows) == 1
    row = rows.iloc[0]
    frame = int(row["frame"])
    assert BUMP.R_TOUCH < window_distance(sc, frame, "A") <= BUMP.R_REACH_M
    assert window_distance(sc, frame, "H") > window_distance(sc, frame, "A")
    assert (row["kind"], row["team"]) == ("control", "A")     # an other-team redirect, confirmed
    assert row["notes"] == events.REDIRECT_NOTE
    assert np.isclose(row["dist_m"], window_distance(sc, frame, "A"))
    assert row["dist_m"] > BUMP.R_TOUCH                       # dist_m marks the reach row
    assert row["confidence"] <= events.confidence(BUMP.R_TOUCH, np.inf, 1.0, BUMP) + 1e-12
    value = events.shape_response(shape_of(sc[3]), frame, "A", BUMP.SHAPE_LAG_FRAMES)
    assert value >= BUMP.SHAPE_CONFIRM_MS


def test_reach_contact_without_a_shape_response_has_no_row():
    assert near_redirect(detect(scene(reach_m=2.4, response=False))).empty


def test_player_beyond_the_reach_gives_no_row():
    sc = scene(reach_m=3.5, response=True)
    tables = detect(sc)
    assert near_redirect(tables).empty
    h = int(BUMP.CONTACT_HALF_WINDOW)
    size = events.impulse(sc[0], h)
    maxima = events.window_maxima(size, h, np.flatnonzero(size >= BUMP.DV_CONTACT))
    turn = [int(c) for c in maxima if abs(int(c) - REDIRECT) <= 3]
    assert len(turn) == 1 and window_distance(sc, turn[0], "A") > BUMP.R_REACH_M


def test_owner_within_the_touch_radius_keeps_its_row():
    sc = scene(reach_m=2.4, response=True, owner_near=True)
    rows = near_redirect(detect(sc))
    assert rows["kind"].tolist() == ["touch"] and rows["team"].tolist() == ["H"]
    assert rows["notes"].tolist() == [events.REDIRECT_NOTE]
    assert detect(sc).equals(detect(sc, cfg=OFF))


def test_switch_off_and_no_shape_give_no_reach_row():
    sc = scene(reach_m=2.4, response=True)
    assert near_redirect(detect(sc, cfg=OFF)).empty
    assert near_redirect(detect(sc, shape=None)).empty
    plain = events.detect_events(*sc[:2], None, sc[2], BUMP)
    assert plain.equals(detect(sc, cfg=OFF))
    # Without a shape array _bump_contacts keeps the C8 rule: no row with nobody near.
    alive = np.ones(N, dtype=bool)
    rows = events._bump_contacts(sc[0], sc[1], sc[2], alive, BUMP)
    assert [(r[1], r[2]) for r in rows] == [("kick", "H")]


def test_reach_rows_are_deterministic():
    sc = scene(reach_m=2.4, response=True)
    assert detect(sc).equals(detect(sc))

"""Synthetic tests for the C15 team-shape confirmation of a redirect (engine/events.py, tracks.py).

The scenes reuse the smoothed-ball builders of the C8 tests. H kicks at frame 40, and an A
player redirects the ball at frame 100. No player of A plays the ball after that, so the C8
"next contact" clause never confirms the redirect. Six more players of each team stand away
from the ball. A defends x = 0 and H defends x = length (the median x of each team). In the "response"
scenes both teams run in +x from frame 110: A goes forward and H goes back.
"""

import dataclasses

import numpy as np
import pytest

from engine import events, tracks
from tests.test_events import (A, BUMP, FAST, FPS, H, PITCH, far, piecewise, smooth_path,
                               static)

N = 220
REDIRECT = 100
RUN_FROM = 110
RUN_SPEED = 1.5          # m/s, run by 6 of the 7 A players and 6 of the 8 H players
A_MEAN = RUN_SPEED * 6 / 7.0
H_MEAN = RUN_SPEED * 6 / 8.0
BLOCKS = [(1, 0, N - 1)]
OFF = dataclasses.replace(BUMP, REDIRECT_SHAPE_CONFIRM=False)


def runner(team, x0, y0, n=N, speed=RUN_SPEED, start=RUN_FROM):
    """A player who stands at (x0, y0) and runs in +x at `speed` m/s from frame `start`."""
    t = np.arange(n, dtype=np.float64)
    x = x0 + speed * np.clip(t - start, 0.0, None) / FPS
    return team, x, np.full(n, float(y0))


def scene(redirector=A, response=True, a_players=6, h_players=6):
    """(ball, feats, slot_conf, tactical) of a redirect at REDIRECT with no later contact."""
    speeds = piecewise(N, [(40, FAST)])
    headings = piecewise(N, [(REDIRECT, 60.0)])
    bx, by = smooth_path(speeds, headings)
    speed = RUN_SPEED if response else 0.0
    players = [static(H, 10.0, 10.3, N),
               static(redirector, bx[REDIRECT] + 0.25, by[REDIRECT] - 0.45, N),
               far(H if redirector == A else A, N)]
    players += [runner(A, 5.0 + 6.0 * i, 2.0, speed=speed) for i in range(a_players)]
    players += [runner(H, 70.0 + 5.0 * i, 2.0, speed=speed) for i in range(h_players)]
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


def shape_of(tactical, cfg=BUMP):
    return tracks.team_forward_velocity(tactical, BLOCKS, PITCH, cfg, fps=FPS)


def contact_rows(table):
    """The rows of `table` that are not control runs or dribbles."""
    runs = table["notes"].astype(str).str.startswith("end=") | (table["kind"] == "dribble")
    return table[~runs.to_numpy()]


def detect(sc, cfg=BUMP, e_dead=None, shape="auto"):
    ball, feats, slot_conf, tactical = sc
    if isinstance(shape, str):
        shape = shape_of(tactical, cfg)
    return events.detect_events(ball, feats, None, slot_conf, cfg, e_dead=e_dead, shape=shape)


def redirect_row(table, team="A"):
    rows = contact_rows(table)
    return rows[(rows["team"] == team) & ((rows["frame"] - REDIRECT).abs() <= 3)]


def test_forward_velocity_follows_the_attack_direction():
    vf = shape_of(scene()[3])
    assert abs(vf["A"][60]) < 1e-9 and abs(vf["H"][60]) < 1e-9
    assert abs(vf["A"][160] - A_MEAN) < 1e-6             # A defends x = 0: +x is forward
    assert abs(vf["H"][160] + H_MEAN) < 1e-6             # H defends x = length: +x is back
    outside = tracks.team_forward_velocity(scene()[3], [(1, 0, 99)], PITCH, BUMP, fps=FPS)
    assert np.isnan(outside["A"][150]) and np.isnan(outside["H"][150])


def test_redirect_with_a_shape_response_and_no_next_contact_is_a_control():
    sc = scene(response=True)
    row = redirect_row(detect(sc))
    assert len(row) == 1
    assert (row["kind"].iloc[0], row["notes"].iloc[0]) == ("control", events.REDIRECT_NOTE)
    frame = int(row["frame"].iloc[0])
    value = events.shape_response(shape_of(sc[3]), frame, "A", BUMP.SHAPE_LAG_FRAMES)
    assert value >= BUMP.SHAPE_CONFIRM_MS and abs(value - (A_MEAN + H_MEAN)) < 1e-6


def test_redirect_without_a_shape_response_has_no_row():
    assert redirect_row(detect(scene(response=False))).empty


def test_switch_off_keeps_the_c8_rule():
    sc = scene(response=True)
    assert redirect_row(detect(sc, cfg=OFF)).empty
    assert redirect_row(detect(sc, shape=None)).empty   # no shape: the C8 rule as it was
    plain = events.detect_events(*sc[:2], None, sc[2], BUMP)
    assert plain.equals(detect(sc, cfg=OFF))


def test_undefined_centroid_does_not_confirm():
    sc = scene(response=True, a_players=BUMP.SHAPE_MIN_PLAYERS - 2)
    vf = shape_of(sc[3])
    assert np.isnan(vf["A"]).all()                       # 1 + 4 A players: below the minimum
    assert redirect_row(detect(sc)).empty


def test_dead_frame_inside_the_lag_does_not_confirm():
    sc = scene(response=True)
    e_dead = np.zeros(N)
    e_dead[REDIRECT + 15:REDIRECT + 18] = 0.9
    assert redirect_row(detect(sc, e_dead=e_dead)).empty


def test_redirect_by_the_owner_stays_a_touch():
    rows = contact_rows(detect(scene(redirector=H, response=True)))
    owner = rows[(rows["team"] == "H") & ((rows["frame"] - REDIRECT).abs() <= 3)]
    assert owner["kind"].tolist() == ["touch"]
    assert owner["notes"].tolist() == [events.REDIRECT_NOTE]


def test_shape_response_is_nan_off_the_array_and_deterministic():
    sc = scene(response=True)
    vf = shape_of(sc[3])
    assert np.isnan(events.shape_response(vf, N - 5, "A", BUMP.SHAPE_LAG_FRAMES))
    assert detect(sc).equals(detect(sc))


@pytest.mark.parametrize("dropout", [REDIRECT, REDIRECT + BUMP.SHAPE_LAG_FRAMES])
def test_smoothing_keeps_a_player_dropout_undefined(dropout):
    tactical = scene()[3]
    cols = np.flatnonzero(tactical["team_of_slot"] == A)
    tactical["px"][dropout, cols[:2]] = np.nan
    shape = shape_of(tactical)
    assert np.isnan(shape["A"][dropout])
    assert np.isfinite(shape["A"][dropout - 3])
    assert np.isfinite(shape["H"][dropout])
    assert not events.shape_confirms(shape, REDIRECT, "A", np.zeros(N + 1), BUMP)


def test_smoothing_keeps_frames_outside_a_block_undefined():
    shape = tracks.team_forward_velocity(scene()[3], [(1, 10, 150)], PITCH, BUMP, fps=FPS)
    for team in ("H", "A"):
        assert np.isnan(shape[team][:10]).all()
        assert np.isnan(shape[team][151:]).all()
        assert np.isfinite(shape[team][10:151]).all()


@pytest.mark.parametrize("second_start", [115, 116])
def test_shape_lag_cannot_cross_adjacent_or_separated_blocks(second_start):
    blocks = [(1, 0, second_start - 1 if second_start == 115 else 114),
              (2, second_start, N - 1)]
    keepers = {(half, team): {"own_goal_x": PITCH[0] if team == H else 0.0}
               for half in (1, 2) for team in (H, A)}
    shape = tracks.team_forward_velocity(scene()[3], blocks, PITCH, BUMP, fps=FPS,
                                        keepers=keepers)
    assert events.shape_response(shape, REDIRECT, "A", BUMP.SHAPE_LAG_FRAMES) >= BUMP.SHAPE_CONFIRM_MS
    assert not events.shape_confirms(shape, REDIRECT, "A", np.zeros(N + 1), BUMP)


@pytest.mark.parametrize("frame", [-1, N - 1, N])
def test_shape_confirmation_rejects_frames_outside_the_available_lag(frame):
    assert not events.shape_confirms(shape_of(scene()[3]), frame, "A", np.zeros(N + 1), BUMP)

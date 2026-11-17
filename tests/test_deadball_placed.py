"""Synthetic tests for the placed-ball restart of engine/deadball.py (cycle C12, F1 rank 1).

A ball at rest inside a stoppage that rises by dv_9 above DV_RESTART is a restart candidate
whatever the distance of the nearest tracked player (clause a). A reappearance whose ball comes
to rest within RESTART_CONFIRM_S is no candidate, because the placed-ball kick is (clause b).
A rise with the ball beyond a line is no placed-ball kick. PLACED_BALL_RESTART False gives the
candidates of C11 exactly. No match data is needed.
"""

import dataclasses

import numpy as np
import pandas as pd

from engine import deadball, possession
from engine.tracks import velocity_step
from tests.test_deadball import (CFG, FPS, PITCH, _candidates, _onsets, make_ball, make_feats,
                                 positions_from_speed)

OFF = dataclasses.replace(CFG, PLACED_BALL_RESTART=False)


def _resting_ball(kick, n=600, y=30.0, rest_from=0, first_visible=0, start_speed=0.0):
    """A ball at rest that a kick raises to 5 m/s over 9 frames, holds 10 frames, then stops.

    The ball is missing before `first_visible`. It rolls from `start_speed` (m/s) to rest over
    `rest_from` frames when `rest_from` is above 0. Positions are metres along y = `y`.
    """
    t = np.arange(n)
    rise = np.clip((t - kick + 1) / 9.0, 0.0, 1.0) * 5.0
    fall = np.clip(1.0 - (t - kick - 19) / 10.0, 0.0, 1.0)
    speed = np.where(t >= kick, rise * fall, 0.0)
    if rest_from > 0:
        roll = np.clip(start_speed * (1.0 - (t - first_visible) / rest_from), 0.0, None)
        speed = np.maximum(speed, np.where(t >= first_visible, roll, 0.0))
    x, yy = positions_from_speed(speed, (30.0, y), heading=(1.0, 0.0))
    x = np.where(t >= first_visible, x, np.nan)
    yy = np.where(t >= first_visible, yy, np.nan)
    return make_ball(x, yy)


def _players(n, distance, moving_from=350):
    """feats: the nearest player is an A player at `distance` m, the players move from a frame."""
    t = np.arange(n)
    return make_feats(n, dA=distance, dH=distance + 10.0, d_second=distance + 10.0,
                      meanv_all=np.where(t >= moving_from, 2.0, 1.0), sep=14.0)


def _kicks(table):
    kicks = table[table["kind"] == "kick"]
    return kicks["frame"].tolist(), kicks["team"].tolist(), kicks["confirmed"].tolist()


# -- clause (a): the placed ball is kicked ------------------------------------------------------

def test_placed_ball_kick_is_a_restart_with_nobody_within_1_5_m():
    """The ball lies still, the nearest tracked player is 20 m away, the ball is kicked at 300."""
    ball, feats = _resting_ball(300), _players(600, 20.0)
    frames, teams, confirmed = _kicks(deadball.restarts(ball, feats, None, None, CFG,
                                                        pitch=PITCH))
    assert len(frames) == 1 and abs(frames[0] - 300) <= 3
    assert teams == [None] and confirmed == [True]
    assert _kicks(deadball.restarts(ball, feats, None, None, OFF, pitch=PITCH)) == ([], [], [])


def test_placed_ball_kick_takes_the_team_of_a_player_within_the_reappear_radius():
    """A taker 2.2 m away gives the team (REAPPEAR_TEAM_M = 3 m). One 3.4 m away does not."""
    for distance, team in ((2.2, "A"), (3.4, None)):
        ball, feats = _resting_ball(300), _players(600, distance)
        frames, teams, _ = _kicks(deadball.restarts(ball, feats, None, None, CFG, pitch=PITCH))
        assert len(frames) == 1 and teams == [team]


def test_nudged_ball_without_player_movement_is_no_confirmed_restart():
    """The players stand still after the nudge: the candidate exists and is not confirmed."""
    ball, feats = _resting_ball(300), _players(600, 20.0, moving_from=10 ** 6)
    frames, _, confirmed = _kicks(deadball.restarts(ball, feats, None, None, CFG, pitch=PITCH))
    assert len(frames) == 1 and confirmed == [False]
    n = 600
    cands = deadball.restarts(ball, feats, None, None, CFG, pitch=PITCH)
    dead, restart_list = possession.dead_intervals(n, _onsets([(50, "out", 1.0)]), np.zeros(n),
                                                   cands, [(1, 0, n - 1)], {1: 0}, CFG, FPS)
    assert dead[50:].all() and restart_list == []               # no restart ends the stoppage


def test_placed_ball_kick_ends_a_stoppage_in_dead_intervals():
    """The stoppage that starts at an out onset at 50 ends at the kick at 300, not never."""
    ball, feats = _resting_ball(300), _players(600, 20.0)
    n = 600
    for cfg, end in ((CFG, 300), (OFF, None)):
        cands = deadball.restarts(ball, feats, None, None, cfg, pitch=PITCH)
        dead, restart_list = possession.dead_intervals(
            n, _onsets([(50, "out", 1.0)]), np.zeros(n), cands, [(1, 0, n - 1)], {1: 0}, cfg,
            FPS)
        if end is None:
            assert dead[50:].all() and restart_list == []
        else:
            assert abs(restart_list[-1][0] - end) <= 3 and not dead[end + 3:].any()


def test_normal_kick_with_a_player_within_1_5_m_is_unchanged():
    """A kick with an A player 0.8 m away gives the same table with the switch on and off."""
    ball, feats = _resting_ball(300), _players(600, 0.8)
    new = deadball.restarts(ball, feats, None, None, CFG, pitch=PITCH)
    old = deadball.restarts(ball, feats, None, None, OFF, pitch=PITCH)
    pd.testing.assert_frame_equal(new, old)
    frames, teams, confirmed = _kicks(new)
    assert len(frames) == 1 and teams == ["A"] and confirmed == [True]


# -- clause (b): a reappearance followed by a placed ball ---------------------------------------

def _reappearing_ball(rest_at, kick=400, n=700):
    """The ball is missing until 100, reappears at 6 m/s and rolls to rest by `rest_at`."""
    return _resting_ball(kick, n=n, first_visible=100, rest_from=rest_at - 100, start_speed=6.0)


def test_reappearance_followed_by_a_placed_ball_kick_is_not_the_restart():
    """82254: the ball reappears moving, lies still, and is kicked. The kick is the restart."""
    ball, feats = _reappearing_ball(rest_at=130), _players(700, 20.0, moving_from=450)
    assert ball["speed"][100] > CFG.REAPPEAR_SPEED
    new = deadball.restarts(ball, feats, None, None, CFG, pitch=PITCH)
    assert new[new["kind"] == "reappear"].empty
    frames, _, confirmed = _kicks(new)
    assert len(frames) == 1 and abs(frames[0] - 400) <= 3 and confirmed == [True]
    old = deadball.restarts(ball, feats, None, None, OFF, pitch=PITCH)
    assert old[old["kind"] == "reappear"]["frame"].tolist() == [100]
    assert _kicks(old) == ([], [], [])


def test_reappearance_stays_a_candidate_when_the_rest_comes_too_late():
    """The ball rolls for more than RESTART_CONFIRM_S: the reappearance is still a candidate."""
    late = 100 + int(round(CFG.RESTART_CONFIRM_S * FPS)) + 100
    ball = _reappearing_ball(rest_at=late, kick=late + 200, n=late + 400)
    feats = _players(late + 400, 20.0, moving_from=late + 250)
    table = deadball.restarts(ball, feats, None, None, CFG, pitch=PITCH)
    assert table[table["kind"] == "reappear"]["frame"].tolist() == [100]


def test_reappearance_stays_a_candidate_when_the_ball_never_rests():
    """No stationary frame follows: no placed-ball kick, the clause does not touch it."""
    n = 200
    t = np.arange(n)
    x = np.where(t < 60, np.nan, 40.0 + 9.0 * (t - 60) / FPS)
    y = np.where(t < 60, np.nan, 30.0)
    ball = make_ball(x, y)
    feats = make_feats(n, dH=np.where(t >= 60, 2.0, np.nan), dA=np.where(t >= 60, 6.0, np.nan))
    for cfg in (CFG, OFF):
        table = deadball.restarts(ball, feats, None, None, cfg, pitch=PITCH)
        assert table[table["kind"] == "reappear"]["frame"].tolist() == [60]


# -- the pitch clause ---------------------------------------------------------------------------

def test_rise_beyond_the_touchline_is_no_placed_ball_kick():
    """186261: a ball at rest 2 m beyond the touchline that moves again comes from off the pitch."""
    inside = deadball.restarts(_resting_ball(300, y=60.0), _players(600, 20.0), None, None, CFG,
                               pitch=PITCH)
    beyond = deadball.restarts(_resting_ball(300, y=70.0), _players(600, 20.0), None, None, CFG,
                               pitch=PITCH)
    assert len(_kicks(inside)[0]) == 1
    assert _kicks(beyond) == ([], [], [])
    ball = _resting_ball(300, y=70.0)
    rises = _rises(ball)
    stationary = deadball.stationary_mask(ball, CFG)
    assert len(deadball.placed_kicks(ball, stationary, rises, None)) == 1
    off_pitch = deadball.outside_pitch(ball, PITCH, CFG.PITCH_MARGIN_M)
    assert deadball.placed_kicks(ball, stationary, rises, off_pitch) == []


def _rises(ball):
    """[(frame, run_stop)] as restarts() builds them."""
    from engine.tracks import _runs
    speed, k = ball["speed"], int(CFG.RESTART_K)
    starts, stops = _runs(deadball._above(velocity_step(speed, k, ball["run_id"]),
                                          CFG.DV_RESTART))
    return [(deadball._rise_start(speed, int(a), min(len(speed) - 1, int(b) + k)), int(b))
            for a, b in zip(starts, stops)]


# -- placed_kicks and _comes_to_rest on plain arrays --------------------------------------------

def _plain(n=200, run_id=None, stationary=()):
    ball = {"run_id": np.zeros(n, dtype=np.int64) if run_id is None else np.asarray(run_id)}
    mask = np.zeros(n, dtype=bool)
    for lo, hi in stationary:
        mask[lo:hi] = True
    return ball, mask


def test_placed_kicks_edge_cases_on_plain_arrays():
    ball, still = _plain(stationary=[(20, 100)])
    assert deadball.placed_kicks(ball, still, [(110, 125)]) == [(110, 20)]
    assert deadball.placed_kicks(ball, still, []) == []
    # Only the first rise after the rest counts: the second has no rest of its own.
    assert deadball.placed_kicks(ball, still, [(110, 125), (180, 195)]) == [(110, 20)]
    # A new rest after the first rise makes the second rise a placed-ball kick too.
    ball, again = _plain(stationary=[(20, 100), (140, 175)])
    assert deadball.placed_kicks(ball, again, [(110, 125), (180, 195)]) == [(110, 20), (180, 140)]
    # The rest begins at frame 0 (first frame) and the rise frame itself is stationary.
    ball, first = _plain(stationary=[(0, 112)])
    assert deadball.placed_kicks(ball, first, [(110, 125)]) == [(110, 0)]
    # No stationary frame before the rise.
    ball, none = _plain()
    assert deadball.placed_kicks(ball, none, [(110, 125)]) == []
    # The off-pitch mask removes the kick, a mask that is False keeps it.
    ball, still = _plain(stationary=[(20, 100)])
    off = np.zeros(200, dtype=bool)
    assert deadball.placed_kicks(ball, still, [(110, 125)], off) == [(110, 20)]
    off[110] = True
    assert deadball.placed_kicks(ball, still, [(110, 125)], off) == []


def test_placed_kicks_ignore_a_rest_in_another_visible_run():
    """A gap (run_id -1) and a new run between the rest and the rise: not a placed ball."""
    run_id = np.zeros(200, dtype=np.int64)
    run_id[100:105] = -1
    run_id[105:] = 1
    ball, still = _plain(run_id=run_id, stationary=[(20, 100)])
    assert deadball.placed_kicks(ball, still, [(110, 125)]) == []


def test_comes_to_rest_on_plain_arrays():
    ball, _ = _plain()
    rests = np.array([130, 400], dtype=np.int64)
    assert deadball._comes_to_rest(ball, rests, 100, 150)
    assert not deadball._comes_to_rest(ball, rests, 100, 20)          # rest after the window
    assert not deadball._comes_to_rest(ball, rests, 130, 150)         # rest at the frame itself
    assert not deadball._comes_to_rest(ball, np.array([], dtype=np.int64), 100, 150)
    other = {"run_id": np.where(np.arange(200) >= 120, 1, 0)}
    assert not deadball._comes_to_rest(other, rests, 100, 150)        # rest in another run


# -- determinism and the switch -----------------------------------------------------------------

def test_placed_ball_restart_is_deterministic():
    ball, feats = _reappearing_ball(rest_at=130), _players(700, 20.0, moving_from=450)
    first = deadball.restarts(ball, feats, None, None, CFG, pitch=PITCH)
    second = deadball.restarts(ball, feats, None, None, CFG, pitch=PITCH)
    pd.testing.assert_frame_equal(first, second)


def test_the_switch_is_a_detection_field_and_defaults_to_true():
    """The cache key of possession.prepare() must change with the switch."""
    from engine import possession as pos
    assert CFG.PLACED_BALL_RESTART is True
    assert pos.detection_key(CFG) != pos.detection_key(OFF)

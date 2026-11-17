"""Synthetic tests for the goalkeeper hold of engine/deadball.py and engine/possession.py (C13).

A ball gap that starts with a slow ball inside a goalkeeper's own penalty area, within R_TOUCH
of his slot, is a goalkeeper hold. Law 12 keeps the ball in play while the goalkeeper holds it.
An evidence entry in such a gap starts no `D` unless the stoppage it would start holds a deep
stop or a collective stop. GK_HOLD_IS_PLAY False gives the stoppages of C12 exactly. A hold
that stays alive gives the goalkeeper's team a `control` row at the start of the gap (C13b).
GK_HOLD_OWNER False gives the event table of C13 exactly. No match data is needed.
"""

import dataclasses

import numpy as np

from engine import deadball, events, possession, tracks
from tests.test_deadball import (CFG, FPS, PITCH, _candidates, _onsets, make_ball, make_feats,
                                 positions_from_speed)

OFF = dataclasses.replace(CFG, GK_HOLD_IS_PLAY=False)
OWNER_OFF = dataclasses.replace(CFG, GK_HOLD_OWNER=False)
N = 800
LAST = 99               # last visible frame before the gap
REAPPEAR = 600          # the ball comes back here (a punt), so the gap lasts 20 s


def _held_ball(end_speed=1.0, start_x=14.0):
    """A ball that rolls toward the goal at x = 0, vanishes after LAST and returns at REAPPEAR.

    The speed falls from 4 m/s to `end_speed` over frames 0..LAST. The ball comes back at
    (20, 34) moving at 10 m/s.
    """
    t = np.arange(N)
    speed = np.where(t <= LAST, 4.0 + (end_speed - 4.0) * t / LAST, 0.0)
    x, y = positions_from_speed(speed, (start_x, 34.0), heading=(-1.0, 0.0))
    x = np.where(t <= LAST, x, np.nan)
    y = np.where(t <= LAST, y, np.nan)
    x[REAPPEAR:] = 20.0 + 10.0 * (t[REAPPEAR:] - REAPPEAR) / FPS
    y[REAPPEAR:] = 34.0
    return make_ball(x, y)


def _keeper_at(ball, offset=(0.5, 0.0), team=0, own_goal_x=0.0):
    """goalkeeper_positions(): the slot of `team` stands `offset` metres from the last ball."""
    out = {key: np.full((N, 2), np.nan) for key in ("x", "y", "own_goal_x")}
    out["x"][:, team] = ball["x"][LAST] + offset[0]
    out["y"][:, team] = ball["y"][LAST] + offset[1]
    out["own_goal_x"][:, team] = own_goal_x
    out["slot"] = np.full((N, 2), -1, dtype=np.int64)
    out["slot"][:, team] = 0
    return out


def _feats(meanv_gap=1.3, sep_gap=8.0):
    """Players walk at `meanv_gap` m/s with centroids `sep_gap` m apart while the ball is away."""
    t = np.arange(N)
    gap = (t > LAST) & (t < REAPPEAR)
    return make_feats(N, meanv_all=np.where(gap, meanv_gap, 2.5), sep=np.where(gap, sep_gap, 8.0))


def _stoppage(ball, feats, keepers, cfg=CFG, onsets=None):
    """dead_intervals() with the e_D of dead_evidence() and one confirmed reappear at REAPPEAR."""
    e_dead = deadball.dead_evidence(ball, feats, None, None, cfg, pitch=PITCH)
    cands = _candidates([(REAPPEAR, "H", "reappear", 0.6, True)])
    holds = None
    if cfg.GK_HOLD_IS_PLAY:
        holds = (deadball.keeper_hold_mask(ball, keepers, PITCH, cfg),
                 deadball.stop_signature_mask(feats, cfg, FPS))
    onsets = _onsets([]) if onsets is None else onsets
    dead, _ = possession.dead_intervals(N, onsets, e_dead, cands, [(1, 0, N - 1)], {1: 0}, cfg,
                                        FPS, holds)
    return dead


# -- the hold mask ------------------------------------------------------------------------------

def test_slow_ball_at_the_goalkeeper_inside_his_box_starts_a_hold():
    """The ball stops at 1 m/s 0.5 m from the goalkeeper, 4 m from his goal line."""
    ball = _held_ball()
    hold = deadball.keeper_hold_mask(ball, _keeper_at(ball), PITCH, CFG)
    assert ball["x"][LAST] < CFG.PENALTY_AREA_DEPTH_M
    assert np.flatnonzero(hold).tolist() == list(range(LAST + 1, REAPPEAR))


def test_vanish_near_a_player_who_is_not_the_goalkeeper_is_no_hold():
    """The goalkeeper-like slot stands 5 m from the ball. A defender at the ball does not count."""
    ball = _held_ball()
    assert not deadball.keeper_hold_mask(ball, _keeper_at(ball, (5.0, 0.0)), PITCH, CFG).any()


def test_vanish_outside_the_penalty_area_is_no_hold():
    """The ball vanishes 18.5 m from the goal line with the goalkeeper 0.5 m away."""
    ball = _held_ball(start_x=28.5)
    assert ball["x"][LAST] > CFG.PENALTY_AREA_DEPTH_M
    assert not deadball.keeper_hold_mask(ball, _keeper_at(ball), PITCH, CFG).any()


def test_vanish_in_the_other_teams_box_or_at_speed_is_no_hold():
    """The slot of the team that defends the far goal, or a ball last seen at 5 m/s."""
    ball = _held_ball()
    far = _keeper_at(ball, own_goal_x=PITCH[0])
    assert not deadball.keeper_hold_mask(ball, far, PITCH, CFG).any()
    fast = _held_ball(end_speed=5.0, start_x=22.0)
    assert fast["speed"][LAST] > CFG.DV_RESTART
    assert not deadball.keeper_hold_mask(fast, _keeper_at(fast), PITCH, CFG).any()


def test_stop_signature_needs_a_long_deep_stop_or_a_collective_stop():
    """9 s below 1 m/s is not a deep stop, 10 s is. 2 s of slow, separated players is one."""
    t = np.arange(N)
    short = make_feats(N, meanv_all=np.where((t >= 100) & (t < 325), 0.5, 2.5))
    long_ = make_feats(N, meanv_all=np.where((t >= 100) & (t < 350), 0.5, 2.5))
    split = make_feats(N, meanv_all=np.where((t >= 100) & (t < 150), 1.2, 2.5),
                       sep=np.where((t >= 100) & (t < 150), 14.0, 8.0))
    assert not deadball.stop_signature_mask(short, CFG, FPS).any()
    assert np.flatnonzero(deadball.stop_signature_mask(long_, CFG, FPS)).tolist() == \
        list(range(100, 350))
    assert np.flatnonzero(deadball.stop_signature_mask(split, CFG, FPS)).tolist() == \
        list(range(100, 150))


# -- the stoppage -------------------------------------------------------------------------------

def test_hold_with_players_still_moving_is_play():
    """Players walk at 1.3 m/s, teams 8 m apart: the evidence entry starts no D (133932)."""
    ball = _held_ball()
    dead = _stoppage(ball, _feats(), _keeper_at(ball))
    assert not dead.any()
    assert _stoppage(ball, _feats(), _keeper_at(ball), OFF).any()     # C12: D in the gap


def test_hold_followed_by_a_deep_stop_is_dead():
    """Players stand at 0.5 m/s for the whole 20 s gap: D from the entry to the reappearance."""
    ball = _held_ball()
    dead = _stoppage(ball, _feats(meanv_gap=0.5), _keeper_at(ball))
    on = np.flatnonzero(dead)
    assert on.size and on[-1] == REAPPEAR - 1 and on[0] > LAST
    assert np.array_equal(dead, _stoppage(ball, _feats(meanv_gap=0.5), _keeper_at(ball), OFF))


def test_hold_followed_by_separation_is_dead():
    """Players walk at 1.3 m/s with the team centroids 15 m apart: a collective stop."""
    ball = _held_ball()
    feats = _feats(sep_gap=15.0)
    dead = _stoppage(ball, feats, _keeper_at(ball))
    assert dead.any() and np.array_equal(dead, _stoppage(ball, feats, _keeper_at(ball), OFF))


def test_vanish_near_a_non_goalkeeper_or_outside_the_box_keeps_the_stoppage():
    """Without a hold the gap with slow players enters D as in C12."""
    ball = _held_ball()
    dead = _stoppage(ball, _feats(), _keeper_at(ball, (5.0, 0.0)))
    assert dead.any() and np.array_equal(dead, _stoppage(ball, _feats(), _keeper_at(ball), OFF))
    outside = _held_ball(start_x=28.5)
    dead = _stoppage(outside, _feats(), _keeper_at(outside))
    assert dead.any() and np.array_equal(dead, _stoppage(outside, _feats(),
                                                         _keeper_at(outside), OFF))


def test_onset_entry_inside_a_hold_is_kept():
    """An out onset in the hold gap enters D. The rule skips evidence entries only."""
    ball = _held_ball()
    dead = _stoppage(ball, _feats(), _keeper_at(ball), onsets=_onsets([(300, "out", 1.0)]))
    assert np.flatnonzero(dead).tolist() == list(range(300, REAPPEAR))


def test_switch_off_gives_no_holds_and_the_old_stoppage():
    """GK_HOLD_IS_PLAY False: keeper_holds() is None and dead_intervals() ignores the hold."""
    ball = _held_ball()
    tactical = {"px": np.full((N, 1), 0.1), "py": np.full((N, 1), 0.5),
                "team_of_slot": np.array([0])}
    assert possession.keeper_holds(tactical, ball, _feats(), [(1, 0, N - 1)], PITCH, OFF) is None
    e_dead = deadball.dead_evidence(ball, _feats(), None, None, CFG, pitch=PITCH)
    cands = _candidates([(REAPPEAR, "H", "reappear", 0.6, True)])
    plain, _ = possession.dead_intervals(N, _onsets([]), e_dead, cands, [(1, 0, N - 1)], {1: 0},
                                         CFG, FPS)
    assert np.array_equal(plain, _stoppage(ball, _feats(), _keeper_at(ball), OFF))


def test_hold_is_deterministic():
    ball = _held_ball()
    first = deadball.keeper_hold_mask(ball, _keeper_at(ball), PITCH, CFG)
    second = deadball.keeper_hold_mask(ball, _keeper_at(ball), PITCH, CFG)
    assert np.array_equal(first, second)
    assert np.array_equal(_stoppage(ball, _feats(), _keeper_at(ball)),
                          _stoppage(ball, _feats(), _keeper_at(ball)))


# -- goalkeeper positions -----------------------------------------------------------------------

def test_goalkeeper_positions_follow_the_slot_inside_the_block_only():
    """Slot 1 is the H goalkeeper of block (1, 2, 5). A has no goalkeeper-like slot."""
    n = 8
    px = np.full((n, 2), 0.5)
    px[:, 1] = 0.05
    py = np.full((n, 2), 0.5)
    py[4, 1] = np.nan
    px[4, 1] = np.nan
    tactical = {"px": px, "py": py, "team_of_slot": np.array([1, 0])}
    slots = {(1, 0): {"slot": 1, "own_goal_x": 0.0}, (1, 1): {"slot": -1, "own_goal_x": 0.0}}
    out = tracks.goalkeeper_positions(tactical, slots, [(1, 2, 5)], PITCH)
    seen = np.flatnonzero(np.isfinite(out["x"][:, 0])).tolist()
    assert seen == [2, 3, 5]
    assert np.allclose(out["x"][seen, 0], 0.05 * PITCH[0])
    assert np.allclose(out["own_goal_x"][seen, 0], 0.0)
    assert not np.isfinite(out["x"][:, 1]).any()


# -- the owner of a hold (C13b) -----------------------------------------------------------------

def _table(rows):
    """An event table with the columns of engine.events."""
    import pandas as pd
    table = pd.DataFrame(rows, columns=events.COLUMNS)
    return table.astype({"frame": np.int64, "confidence": np.float64, "dist_m": np.float64})


def test_catch_by_the_away_goalkeeper_switches_the_owner_to_a():
    """H releases at 40, the A goalkeeper catches: a control A at the first gap frame sets A."""
    ball = _held_ball()
    keepers = _keeper_at(ball, team=1)
    rows = possession.keeper_hold_rows(ball, _feats(), None, keepers, PITCH, CFG)
    assert rows["frame"].tolist() == [LAST + 1] and rows["team"].tolist() == ["A"]
    assert rows["kind"].tolist() == ["control"]
    assert rows["notes"].tolist() == ["keeper_hold;end={}".format(REAPPEAR - 1)]
    base = _table([(40, "release", "H", 0.8, 0.5, "")])
    table = possession.add_keeper_rows(base, rows, np.zeros(N, dtype=bool))
    assert possession.owner_events(table, CFG) == [(40, "H"), (LAST + 1, "A")]
    assert (LAST + 1, "A") in possession.team_event_rows(table)


def test_hold_owner_switch_off_gives_the_table_of_c13():
    ball = _held_ball()
    keepers = _keeper_at(ball, team=1)
    assert possession.keeper_hold_rows(ball, _feats(), None, keepers, PITCH, OWNER_OFF) is None
    base = _table([(40, "release", "H", 0.8, 0.5, "")])
    assert possession.add_keeper_rows(base, None, np.zeros(N, dtype=bool)) is base


def test_hold_that_becomes_dead_gets_no_owner_row():
    """A stoppage frame inside the gap: no row, so the owner after the restart does not change."""
    ball = _held_ball()
    rows = possession.keeper_hold_rows(ball, _feats(), None, _keeper_at(ball, team=1), PITCH, CFG)
    stoppage = np.zeros(N, dtype=bool)
    stoppage[300:REAPPEAR] = True
    base = _table([(40, "release", "H", 0.8, 0.5, "")])
    table = possession.add_keeper_rows(base, rows, stoppage)
    assert possession.owner_events(table, CFG) == [(40, "H")]


def test_hold_owner_confidence_is_a_control_confidence():
    """The goalkeeper 0.5 m from the ball, no opponent seen: EVENT_CONF_MAX x distance factor."""
    ball = _held_ball()
    feats = make_feats(N, dH=np.inf, dA=0.5)
    rows = possession.keeper_hold_rows(ball, feats, None, _keeper_at(ball, team=1), PITCH, CFG)
    expected = events.confidence(0.5, np.inf, 1.0, CFG)
    assert abs(rows["confidence"].iloc[0] - expected) < 1e-9
    assert rows["confidence"].iloc[0] >= CFG.OWNER_MIN_CONFIDENCE

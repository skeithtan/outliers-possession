"""The kickoff team rule of mode "model" (Config.LEARNED_KICKOFF_TEAM) on synthetic sequences.

The rule forces the first LEARNED_KICKOFF_FLOOR_S of each half from its kickoff frame to the team
that takes the kickoff: in half 1 the team of the first alive frame of the decode without the
rule, in half 2 the other team (Law 8). On by default since round 8 (candidate E3); off writes
E2's output (tests/test_learned.py).
"""

import dataclasses

import numpy as np

from engine import learned
from engine.config import Config

H, A, D = learned.H, learned.A, learned.D
BLOCKS = [(1, 0, 99), (2, 120, 219)]
KICKOFFS = {1: 0, 2: 140}


def test_on_by_default():
    cfg = Config.default()
    assert cfg.LEARNED_KICKOFF_TEAM is True
    assert cfg.LEARNED_KICKOFF_FLOOR_S == 1.0
    assert learned.kickoff_team_on(cfg) is True
    assert learned.kickoff_team_on(dataclasses.replace(cfg, LEARNED_KICKOFF_TEAM=False)) is False


def test_kickoff_teams_follow_law_8():
    codes = np.full(220, 3, dtype=np.int8)
    codes[0:5] = D
    codes[5:100] = A
    codes[120:220] = A
    assert learned.kickoff_teams(codes, BLOCKS, KICKOFFS) == {1: A, 2: H}
    codes[5:100] = H
    assert learned.kickoff_teams(codes, BLOCKS, KICKOFFS) == {1: H, 2: A}
    codes[0:100] = D
    assert learned.kickoff_teams(codes, BLOCKS, KICKOFFS) == {}


def test_kickoff_team_mask_covers_the_floor_only():
    mask = learned.kickoff_team_mask(220, BLOCKS, KICKOFFS, {1: A, 2: H}, 25)
    assert set(mask[0:25].tolist()) == {A}
    assert set(mask[140:165].tolist()) == {H}
    assert (mask[25:140] == -1).all() and (mask[165:] == -1).all()
    short = learned.kickoff_team_mask(220, BLOCKS, {1: 90, 2: 140}, {1: A, 2: H}, 25)
    assert set(short[90:100].tolist()) == {A} and short[100] == -1     # clipped to the block


def _probs(n, team_a_until):
    """Frame probabilities: D before frame 140 of half 2, then A, then H after team_a_until."""
    p = np.tile([0.1, 0.8, 0.1], (n, 1))
    p[120:140] = [0.05, 0.05, 0.9]
    p[140:team_a_until] = [0.2, 0.7, 0.1]
    p[team_a_until:] = [0.8, 0.1, 0.1]
    return p


def test_forced_team_none_or_empty_changes_nothing():
    p = _probs(220, 150)
    forced = learned.pre_kickoff_mask(220, BLOCKS, KICKOFFS)
    base = learned.decode_model(p, BLOCKS, 12.0, 12.0, forced)
    empty = np.full(220, -1, dtype=np.int8)
    assert np.array_equal(base, learned.decode_model(p, BLOCKS, 12.0, 12.0, forced,
                                                     forced_team=empty))


def test_second_half_kickoff_takes_the_other_team():
    p = _probs(220, 150)
    forced = learned.pre_kickoff_mask(220, BLOCKS, KICKOFFS)
    base = learned.decode_model(p, BLOCKS, 12.0, 12.0, forced)
    assert base[0] == A and base[140] == A                     # the decoder alone: A at 140
    teams = learned.kickoff_teams(base, BLOCKS, KICKOFFS)
    assert teams == {1: A, 2: H}
    mask = learned.kickoff_team_mask(220, BLOCKS, KICKOFFS, teams, 25)
    out = learned.decode_model(p, BLOCKS, 12.0, 12.0, forced, forced_team=mask)
    assert set(out[140:165].tolist()) == {H}
    assert np.array_equal(out[:140], base[:140])                 # nothing before the kickoff
    changed = np.flatnonzero(out != base)
    assert changed.min() == 140 and changed.max() < 150          # only the old A run

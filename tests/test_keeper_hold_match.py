"""C13: the goalkeeper holds of this match (F1 Section 6 rank 2).

The frames of this match live here as assertions and nowhere in engine/ (PLAN.md 5.5). The
tests need the match data bundle and are skipped without it.
"""

import os

import numpy as np
import pytest

import games
from engine import data, possession, tracks
from engine.config import Config

CFG = Config.default()
_ASSETS = games.asset_paths()
HAVE_DATA = all(os.path.isfile(_ASSETS[k]) for k in ("atd_meta", "atd_raw"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present")

# Ball gaps that start in a goalkeeper's hands, half-open (the C13 measurement).
HOLD_GAPS = [(122672, 123048), (130598, 131127), (133932, 134453), (135279, 135448),
             (140087, 140216), (147511, 147643), (175384, 175578), (187127, 187264)]


@pytest.fixture(scope="module")
def holds():
    tactical = possession.engine_tactical(CFG)
    ball, feats, _ = tracks.prepare(tactical, CFG)
    return possession.keeper_holds(tactical, ball, feats, data.graded_blocks(), data.pitch_m(),
                                   CFG)


@pytest.mark.slow
@needs_data
def test_hold_gaps_of_this_match(holds):
    hold, _ = holds
    starts, stops = tracks._runs(hold)
    assert list(zip(starts.tolist(), stops.tolist())) == HOLD_GAPS
    assert not hold[23831:24097].any()       # ball last seen at 5.0 m/s
    assert not hold[119589:119812].any()     # ball last seen 0.4 m outside the box


@pytest.mark.slow
@needs_data
def test_stop_signature_decides_the_holds_of_this_match(holds):
    _, stop = holds
    assert not stop[133932:134453].any()     # catch, 21 s hold and punt: play
    assert not stop[187146:187264].any()     # play
    assert stop[175384:176993].any()         # 64 s of standing players: a stoppage
    assert np.count_nonzero(stop) > 0


@pytest.mark.slow
@needs_data
def test_goalkeeper_team_of_each_hold_of_this_match():
    """C13b: every hold but the last is the away goalkeeper's (slot 45), the last the home one."""
    from engine import deadball

    tactical = possession.engine_tactical(CFG)
    ball, _, _ = tracks.prepare(tactical, CFG)
    keepers = possession.keeper_positions(tactical, data.graded_blocks(), data.pitch_m(), CFG)
    gaps = deadball.keeper_hold_gaps(ball, keepers, data.pitch_m(), CFG)
    assert [(a, b) for a, b, _, _ in gaps] == HOLD_GAPS
    assert [team for _, _, team, _ in gaps] == [1] * 7 + [0]
    assert abs(gaps[2][3] - 1.04) < 0.01               # 133932: 1.04 m from the A goalkeeper
    assert [int(keepers["slot"][a - 1, team]) for a, _, team, _ in gaps] == [45] * 7 + [23]

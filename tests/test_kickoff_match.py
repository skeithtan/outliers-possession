"""C6: kickoff teams on this match (RESEARCH 4 rank 3, M9; Law 8).

The match-specific frames and teams live here as assertions and nowhere in engine/ (PLAN.md
5.5). The engine rule is the centre mark rule with the Law 8 fallback for the other half. Two
expectations of this match are assertions only:
  * the second-half kickoff team is not the first-half kickoff team (Law 8),
  * after a goal, the kickoff team is not the team of the Smart Tagging GOALS window. This
    assumes that the GOALS team is the scoring team, which the organisers' documentation does
    not say (RESEARCH open question 2).
The tests need the match data bundle and are skipped without it.
"""

import dataclasses
import os

import numpy as np
import pandas as pd
import pytest

import games
from engine import data, possession, tracks
from engine.config import Config

CFG = Config.default()
_ASSETS = games.asset_paths()
HAVE_DATA = all(os.path.isfile(_ASSETS[k])
                for k in ("atd_meta", "atd_raw", "homography", "smart_tagging"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present")

KICKOFF_1, KICKOFF_2 = 11185, 113741
# Frames of A at the start of the second half in the output of candidate E2 (a known limit of E2,
# removed in the default since round 8 by Config.LEARNED_KICKOFF_TEAM; E2 = that setting off).
E2_KICKOFF_2_LAG = 21
E2_CFG = dataclasses.replace(CFG, LEARNED_KICKOFF_TEAM=False)
# The restart that ends the stoppage after each GOALS window, as the c6 engine finds it:
# (kickoff frame, GOALS team, rule). At 62467 the nearest tracked player is 3.6 m from the mark
# (the taker is not tracked), so the restart candidate gives the team.
GOAL_KICKOFFS = [(49723, "H", "mark"), (62467, "A", "restart"), (75542, "A", "mark"),
                 (163093, "H", "mark")]
OTHER = {"H": "A", "A": "H"}
# P2's historical reappearance offset in frames, included in the c6 frames above (PLAN 1.4).
P2_REAPPEAR_OFFSET = 10
# The 62467 kickoff is a kick exit. C11 adds the measured kick offset.
# The other historical C6 frames include P2's +10 reappearance offset. R1 removes that delay.
# Compare the engine edges with the observed reappearance anchors and the configured offset.
KICK_EXIT = {frame: frame - P2_REAPPEAR_OFFSET + CFG.DELTA_RESTART_REAPPEAR
             for frame, _, rule in GOAL_KICKOFFS if rule == "mark"}
KICK_EXIT[62467] = 62467 + CFG.DELTA_RESTART_KICK


@pytest.fixture(scope="module")
def mark():
    return tracks.mark_nearest(data.load_tactical(), data.pitch_m())


def _no_candidates():
    return pd.DataFrame({"frame": np.zeros(0, dtype=np.int64), "team": np.zeros(0, dtype=object),
                         "confirmed": np.zeros(0, dtype=bool)})


@needs_data
def test_half_kickoffs_follow_law_8(mark):
    rules = possession.kickoff_decisions(data.kickoff_frames(), _no_candidates(), CFG, mark)
    assert data.kickoff_frames() == {1: KICKOFF_1, 2: KICKOFF_2}
    team_1, rule_1, dist_1 = rules[1]
    assert (team_1, rule_1) == ("A", "mark") and dist_1 < 0.5     # the taker at 0.47 m
    team_2, rule_2, dist_2 = rules[2]
    assert rule_2 == "law" and dist_2 > CFG.KICKOFF_TAKER_M      # the taker is not tracked
    assert team_2 == OTHER[team_1] == "H"


@needs_data
def test_after_goal_mark_teams_are_not_the_goals_team(mark):
    for frame, goals_team, rule in GOAL_KICKOFFS:
        team, dist = possession.mark_team(mark, frame, CFG)
        if rule == "mark":
            assert team == OTHER[goals_team] and dist < 0.5, frame
        else:
            assert team is None and dist > CFG.KICKOFF_TAKER_M, frame


@pytest.mark.slow
@needs_data
def test_engine_kickoff_teams_on_the_match():
    # The rule engine (c23) explicitly: the goal-kickoff frames below are its calibrated restart
    # frames. The default output is checked by test_default_output_kickoff_teams_on_the_match.
    labels, parts = possession.run(dataclasses.replace(CFG, LEARNED_MODE="off"),
                                   return_parts=True)
    assert {h: r[:2] for h, r in parts["kick_rules"].items()} == {1: ("A", "mark"),
                                                                  2: ("H", "law")}
    rows = [(f, t, r) for f, t, r, _ in parts["goal_kickoffs"]]
    goal_kickoffs = [(KICK_EXIT.get(f, f), g, r) for f, g, r in GOAL_KICKOFFS]
    assert rows == [(f, OTHER[g], r) for f, g, r in goal_kickoffs]
    assert labels[KICKOFF_1] == "A" and labels[KICKOFF_2] == "H"
    assert set(labels[KICKOFF_2 - 50:KICKOFF_2].tolist()) == {"D"}   # pre-kickoff frames
    for frame, goals_team, _ in goal_kickoffs:
        assert labels[frame - 1] == "D" and labels[frame] == OTHER[goals_team], frame


@pytest.mark.slow
@needs_data
def test_default_output_kickoff_teams_on_the_match():
    """The checks that hold for any shipped output, on the default configuration (since round 8
    the learned candidate E3: E2 plus the kickoff team rule): the kickoff decisions, the team at
    each half's kickoff, D before the second-half kickoff, and the kicking-off team at each goal
    kickoff's c23 edge frame. The learned candidate ends those stoppages earlier than c23 (a limit
    recorded in docs/plan/calibration/LEARNED_STAGE_RESULT_R6.md Section 9 and
    LEARNED_STAGE_RESULT_R8.md), so the D before the edge frame is checked only for the rule
    engine, above."""
    labels, parts = possession.run(CFG, return_parts=True)
    assert {h: r[:2] for h, r in parts["kick_rules"].items()} == {1: ("A", "mark"),
                                                                  2: ("H", "law")}
    rows = [(f, t, r) for f, t, r, _ in parts["goal_kickoffs"]]
    goal_kickoffs = [(KICK_EXIT.get(f, f), g, r) for f, g, r in GOAL_KICKOFFS]
    assert rows == [(f, OTHER[g], r) for f, g, r in goal_kickoffs]
    floor = int(round(CFG.LEARNED_KICKOFF_FLOOR_S * 25))
    assert labels[KICKOFF_1] == "A"
    assert set(labels[KICKOFF_2:KICKOFF_2 + max(floor, E2_KICKOFF_2_LAG + 1)].tolist()) == {"H"}
    assert set(labels[KICKOFF_2 - 50:KICKOFF_2].tolist()) == {"D"}   # pre-kickoff frames
    for frame, goals_team, _ in goal_kickoffs:
        assert labels[frame] == OTHER[goals_team], frame


@pytest.mark.slow
@needs_data
def test_default_output_second_half_kickoff_frame_is_the_kicking_team():
    """Law 8 at the second-half kickoff frame holds in the default output since round 8 (the
    expected-failure mark of E2 is removed here; it stays on E2's configuration below)."""
    labels = possession.run(CFG)
    assert labels[KICKOFF_2] == "H"


@pytest.mark.slow
@needs_data
def test_e2_output_second_half_starts_with_the_known_a_run():
    """Candidate E2 (LEARNED_KICKOFF_TEAM off) labels the first 21 frames (0.84 s) of the second
    half A, then the kicking-off team H: its known limit, disclosed (private note 6.16 of
    LEARNED_STAGE_PROTOCOL_R6.md; LEARNED_STAGE_RESULT_R6.md Section 9). Kept so that E2 stays
    reachable and described."""
    labels = possession.run(E2_CFG)
    assert labels[KICKOFF_1] == "A"
    assert set(labels[KICKOFF_2:KICKOFF_2 + E2_KICKOFF_2_LAG].tolist()) == {"A"}
    assert labels[KICKOFF_2 + E2_KICKOFF_2_LAG] == "H"
    assert set(labels[KICKOFF_2 - 50:KICKOFF_2].tolist()) == {"D"}   # pre-kickoff frames


@pytest.mark.slow
@needs_data
@pytest.mark.xfail(strict=True, reason="known limit of candidate E2 (LEARNED_KICKOFF_TEAM off): "
                   "the first 21 frames of the second half are labelled A; Law 8 says H "
                   "(private note 6.16)")
def test_e2_output_second_half_kickoff_frame_is_the_kicking_team():
    labels = possession.run(E2_CFG)
    assert labels[KICKOFF_2] == "H"


@pytest.mark.slow
@needs_data
def test_kickoff_team_setting_changes_only_the_old_a_run():
    """The default (LEARNED_KICKOFF_TEAM on) against E2 (off): the first LEARNED_KICKOFF_FLOOR_S
    from each half's kickoff take the kicking team (half 1 the decoder's first team, half 2 the
    other one), and only the frames of E2's old A run at the second-half kickoff change."""
    e2 = possession.run(E2_CFG)
    labels = possession.run(CFG)
    assert labels[KICKOFF_1] == "A" and labels[KICKOFF_2] == "H"
    floor = int(round(CFG.LEARNED_KICKOFF_FLOOR_S * 25))
    assert set(labels[KICKOFF_2:KICKOFF_2 + floor].tolist()) == {"H"}
    changed = np.flatnonzero(labels != e2)
    assert changed.tolist() == list(range(KICKOFF_2, KICKOFF_2 + E2_KICKOFF_2_LAG))
    assert set(e2[changed].tolist()) == {"A"} and set(labels[changed].tolist()) == {"H"}

"""Tests of tools/analysis/link_structure.py and count_side_repair.py on synthetic tracking."""

import numpy as np

from tools.analysis import count_side_repair as csr
from tools.analysis import link_structure as ls
from tools.analysis import tracklet_chains as tc
from tests.test_tracklet_chains import FPS, PITCH, match, put, team_sets


def test_classify_cases():
    assert ls.classify(11, 10, 11, 12) == "relabel_start"
    assert ls.classify(12, 11, 10, 11) == "relabel_end"
    assert ls.classify(11, 10, 10, 11) == "lost_found"
    assert ls.classify(12, 11, 11, 12) == "handover"
    assert ls.classify(10, 9, 9, 10) == "both_within_11"
    assert ls.classify(13, 12, 11, 12) == "other"


def test_wrong_team_episode_is_a_relabel_pair():
    tactical = match(400, 11, 11, spare=1)
    away_spare = 11 + 11 + 1
    put(tactical, 1, 200, 260, away_spare)
    result, _ = ls.analyse(tactical, PITCH, FPS, [(1, 0, 399)], [3])
    cases = result["links"]["3"]["cases"]
    assert cases == {"relabel_end": 1, "relabel_start": 1}
    assert result["links"]["3"]["gap_by_case"]["relabel_start"] == {0: 1}
    episodes = result["episodes"]["away"]
    assert episodes["episodes"] == 1
    assert episodes["start_cases"] == {"other 11->10 (relabel)": 1}
    assert episodes["end_cases"] == {"other 10->11 (relabel ends)": 1}
    assert episodes["start_with_cross_link_within_3"] == 1
    assert result["episode_anatomy"]["away"]["episode_births"]["ends_episode"] == 1


def test_count_side_moves_the_over_counted_side_and_leaves_a_clean_match():
    clean = match(400, 11, 11, spare=1)
    bad = {key: (value.copy() if hasattr(value, "copy") else value) for key, value in clean.items()}
    away_spare = 11 + 11 + 1
    put(bad, 1, 200, 260, away_spare)
    x, y, visible = tc.metres(bad, PITCH)
    spans = tc.tracklets(visible)
    succ = tc.continuation_links(x, y, spans, FPS)
    counts = tc.team_counts(visible, bad["team_of_slot"])
    moves, reasons = csr.count_side_moves(spans, succ, bad["team_of_slot"], counts)
    assert moves == [(away_spare, 200, 260)]
    assert reasons["new_side_over"] == 1 and reasons["ended_side_over"] == 1
    fixed = tc.apply_moves(bad, moves)
    for frame in (150, 230, 300):
        assert team_sets(fixed, frame) == team_sets(clean, frame)
    out = csr.run(bad, PITCH, FPS, [(1, 0, 399)], 3, True)
    assert out["moves"] == 1 and out["over_share_after"]["away"] == 0.0
    assert csr.run(clean, PITCH, FPS, [(1, 0, 399)], 3, True)["moves"] == 0


def test_count_side_skips_links_within_11():
    # A link where neither side is over: a home player lost, an away track found there.
    tactical = match(400, 10, 10, spare=1)
    away_spare = 10 + 10 + 1
    put(tactical, 1, 200, 260, away_spare)
    x, y, visible = tc.metres(tactical, PITCH)
    spans = tc.tracklets(visible)
    succ = tc.continuation_links(x, y, spans, FPS)
    counts = tc.team_counts(visible, tactical["team_of_slot"])
    moves, reasons = csr.count_side_moves(spans, succ, tactical["team_of_slot"], counts)
    assert moves == [] and reasons["neither_over"] == 2


def test_veto_blocks_a_move_that_overfills_the_receiver():
    # Home player 0 is re-labelled away on 200-300 while home player 1 is hidden until 220,
    # so the link at 200 reads new_side_over (home 10 -> 9, away 11 -> 12). A duplicate home
    # track on 230-300 fills home to 11 there, so the move back would overfill home.
    tactical = match(400, 11, 11, spare=2)
    home_spare, away_spare = 11 + 11, 11 + 11 + 2
    tactical["px"][:221, 1] = np.nan
    tactical["py"][:221, 1] = np.nan
    put(tactical, 0, 200, 300, away_spare)
    tactical["px"][230:300, home_spare] = tactical["px"][230:300, 5] + np.float32(0.01)
    tactical["py"][230:300, home_spare] = tactical["py"][230:300, 5]
    x, y, visible = tc.metres(tactical, PITCH)
    spans = tc.tracklets(visible)
    succ = tc.continuation_links(x, y, spans, FPS)
    counts = tc.team_counts(visible, tactical["team_of_slot"])
    with_veto, reasons = csr.count_side_moves(spans, succ, tactical["team_of_slot"], counts)
    without, _ = csr.count_side_moves(spans, succ, tactical["team_of_slot"], counts, veto=False)
    # Both links (200 and 300) name the same tracklet; the veto drops it both times.
    assert without == [(away_spare, 200, 300)]
    assert with_veto == [] and reasons["vetoed"] == 2

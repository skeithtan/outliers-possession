"""Tests of tools/analysis/tracklet_chains.py on synthetic tracking. They need no match data."""

import numpy as np
import pytest

import games
from tools.analysis import tracklet_chains as tc

PITCH = (105.0, 68.0)
FPS = 25


def match(n_frames, n_home, n_away, spare=2, seed=1):
    """A synthetic tactical dict. Each player walks slowly from his own start position.

    The first n_home slots are home players and the next n_away slots are away players. Then
    `spare` empty home slots and `spare` empty away slots follow.
    """
    rng = np.random.RandomState(seed)
    players = n_home + n_away
    start_x = rng.uniform(10.0, 95.0, players)
    start_y = rng.uniform(5.0, 63.0, players)
    step = rng.uniform(-0.04, 0.04, (2, players))
    frames = np.arange(n_frames)[:, None]
    x = start_x[None, :] + frames * step[0][None, :]
    y = start_y[None, :] + frames * step[1][None, :]
    empty = np.full((n_frames, 2 * spare), np.nan)
    team = np.concatenate([np.zeros(n_home), np.ones(n_away), np.zeros(spare), np.ones(spare)])
    return {
        "px": (np.concatenate([x, empty], axis=1) / PITCH[0]).astype(np.float32),
        "py": (np.concatenate([y, empty], axis=1) / PITCH[1]).astype(np.float32),
        "ps": np.full((n_frames, players + 2 * spare), 1.0, dtype=np.float32),
        "bx": np.full(n_frames, 0.5, dtype=np.float32),
        "by": np.full(n_frames, 0.5, dtype=np.float32),
        "team_of_slot": team.astype(np.int8),
        "n_slots": players + 2 * spare,
    }


def put(tactical, slot, start, stop, target):
    """Move the frames [start, stop) of `slot` into the empty slot `target`, in place."""
    for key in ("px", "py", "ps"):
        tactical[key][start:stop, target] = tactical[key][start:stop, slot]
        tactical[key][start:stop, slot] = np.nan


def team_sets(tactical, frame):
    """The rounded positions of each team on one frame, as two sorted lists."""
    out = []
    for team in (0, 1):
        cols = np.flatnonzero(tactical["team_of_slot"] == team)
        x = tactical["px"][frame, cols]
        out.append(sorted(round(float(v), 5) for v in x[np.isfinite(x)]))
    return out


def test_runs_and_tracklets():
    mask = np.array([0, 1, 1, 0, 1], dtype=bool)
    starts, stops = tc.runs(mask)
    assert starts.tolist() == [1, 4] and stops.tolist() == [3, 5]
    visible = np.stack([mask, ~mask], axis=1)
    assert tc.tracklets(visible) == [(0, 1, 3), (0, 4, 5), (1, 0, 1), (1, 3, 4)]


def test_wrong_team_episode_moves_back():
    clean = match(400, 3, 3)
    bad = {key: (value.copy() if hasattr(value, "copy") else value) for key, value in clean.items()}
    away_spare = 3 + 3 + 2
    put(bad, 1, 200, 260, away_spare)
    fixed, moves, spans, succ = tc.repair(bad, PITCH, FPS)
    assert moves == [(away_spare, 200, 260)]
    assert len(succ) == 2
    for frame in (150, 230, 300):
        assert team_sets(fixed, frame) == team_sets(clean, frame)


def test_no_link_between_far_tracklets():
    tactical = match(300, 2, 2)
    away_spare = 2 + 2 + 2
    put(tactical, 0, 100, 200, away_spare)
    tactical["px"][100:200, away_spare] += np.float32(5.0 / PITCH[0])
    _, moves, _, succ = tc.repair(tactical, PITCH, FPS)
    assert succ == {} and moves == []


def test_tracklet_between_two_episodes_keeps_its_team():
    tactical = match(400, 3, 3)
    first, second = 3 + 3 + 2, 3 + 3 + 3
    put(tactical, 1, 100, 130, first)
    put(tactical, 1, 160, 190, second)
    _, moves, _, _ = tc.repair(tactical, PITCH, FPS)
    assert moves == [(first, 100, 130), (second, 160, 190)]


def test_count_decides_a_chain_without_a_majority():
    tactical = match(200, 11, 11, spare=1)
    away_spare = 11 + 11 + 1
    put(tactical, 0, 100, 200, away_spare)
    x, y, visible = tc.metres(tactical, PITCH)
    spans = tc.tracklets(visible)
    succ = tc.continuation_links(x, y, spans, FPS)
    assert tc.team_moves(spans, succ, tactical["team_of_slot"], visible) == [(away_spare, 100, 200)]
    # Without the over-count the same chain gives no move: hide one other away player.
    tactical["px"][:, 11] = np.nan
    tactical["py"][:, 11] = np.nan
    x, y, visible = tc.metres(tactical, PITCH)
    spans = tc.tracklets(visible)
    succ = tc.continuation_links(x, y, spans, FPS)
    assert tc.team_moves(spans, succ, tactical["team_of_slot"], visible) == []


def test_injected_episodes_are_found_and_the_clean_match_has_no_move():
    tactical = match(6000, 11, 11, spare=0, seed=3)
    blocks = [(1, 0, 5999)]
    result = tc.selftest(tactical, blocks, FPS, PITCH, seed=5, rate_per_min=6.0, near_ball_m=0.0)
    assert result["episodes"] >= 10
    assert result["false_moves"] == 0 and result["missed_episodes"] == 0
    assert result["moves_on_the_clean_match"] == 0
    again = tc.selftest(tactical, blocks, FPS, PITCH, seed=5, rate_per_min=6.0, near_ball_m=0.0)
    assert again == result


def test_report_counts_the_over_count():
    tactical = match(200, 11, 11, spare=1)
    put(tactical, 0, 100, 200, 11 + 11 + 1)
    out = tc.report(tactical, PITCH, FPS, [(1, 0, 199)])
    assert out["moves"] == 1 and out["links_to_other_team"] == 1 and out["mixed_chains"] == 1
    assert out["counts_before"]["away"]["over_share"] == 0.5
    assert out["counts_before"]["away"]["other_short_when_over"] == 1.0
    assert out["counts_before"]["away"]["total_22_when_over"] == 1.0
    assert out["counts_after"]["away"]["over_share"] == 0.0


def test_selftest_refuses_the_competition_match():
    with pytest.raises(SystemExit):
        tc.main(["selftest", "--game", games.GAME])

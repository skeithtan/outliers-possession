"""Synthetic tests for K2: owner changes moved back to where the new team took the ball.

possession.backdate_changes() walks back from each direct H <-> A change while the ball is
visible and the new team is the nearer team within touch range. It gives the new team the frames
of that stretch (calibration/COMPONENT_BACKDATE.md, RESEARCH-4 Section 4.1). No match data is
needed.
"""

import dataclasses

import numpy as np
import pytest

from engine import possession
from engine.config import Config
from tests.test_calibration import BASE_EVENTS, _edges, _inputs

FPS = 25.0
N = 400
BLOCKS = [(1, 0, N - 1)]
CHANGE = 200            # the decoded H -> A change (the new team's late kick)
RECEPTION = 170         # the A player has the ball within touch range from here
ON = dataclasses.replace(Config.default(), BACKDATE_CHANGES=True, BACKDATE_RADIUS_M=1.5,
                         BACKDATE_CAP_S=3.0, BACKDATE_MIN_S=0.2,
                         BACKDATE_MIN_PRE_SPELL_S=1.0, BACKDATE_MIN_POST_SPELL_S=1.0)


def _labels(change=CHANGE, before="H", after="A", start=0, n=N):
    labels = np.full(n, before, dtype="<U1")
    labels[change:] = after
    labels[:start] = "D"
    return labels


def _scene(reception=RECEPTION, change=CHANGE, n=N):
    """(d_home, d_away, visible): H near the ball before the reception, A near after it."""
    d_home = np.full(n, 0.8)
    d_away = np.full(n, 6.0)
    d_home[reception:] = 4.0
    d_away[reception:] = 0.9
    return d_home, d_away, np.ones(n, dtype=bool)


def _backdate(labels, scene=None, cfg=ON, blocks=BLOCKS, records=None):
    d_home, d_away, visible = _scene() if scene is None else scene
    return possession.backdate_changes(labels, d_home, d_away, visible, blocks, cfg, FPS,
                                       records)


def _first_a(labels):
    return int(np.flatnonzero(np.asarray(labels) == "A")[0])


def test_a_late_change_moves_back_to_the_reception():
    records = []
    out = _backdate(_labels(), records=records)
    assert _first_a(out) == RECEPTION
    assert records == [{"t1": CHANGE, "t0": RECEPTION, "team": "A"}]
    assert set(out[:RECEPTION]) == {"H"}


def test_the_input_labels_are_not_changed():
    labels = _labels()
    _backdate(labels)
    assert _first_a(labels) == CHANGE


def test_a_change_is_not_moved_when_the_old_team_is_nearer():
    d_home, d_away, visible = _scene()
    d_home[RECEPTION:CHANGE] = 0.5          # H nearer than A (0.9 m) until the change
    out = _backdate(_labels(), (d_home, d_away, visible))
    assert _first_a(out) == CHANGE


def test_a_change_is_not_moved_when_the_new_team_is_out_of_touch_range():
    d_home, d_away, visible = _scene()
    d_home[RECEPTION:CHANGE] = 5.0
    d_away[RECEPTION:CHANGE] = 2.0          # nearer, but beyond R_TOUCH
    out = _backdate(_labels(), (d_home, d_away, visible))
    assert _first_a(out) == CHANGE


def test_a_tie_counts_as_the_nearer_team_and_a_missing_old_team_does_not_stop_the_walk():
    d_home, d_away, visible = _scene()
    d_home[RECEPTION:CHANGE] = 0.9           # equal distances
    d_home[RECEPTION:RECEPTION + 10] = np.nan  # no visible H player
    out = _backdate(_labels(), (d_home, d_away, visible))
    assert _first_a(out) == RECEPTION


def test_the_walk_stops_at_a_ball_gap():
    d_home, d_away, visible = _scene()
    visible[185] = False
    out = _backdate(_labels(), (d_home, d_away, visible))
    assert _first_a(out) == 186


def test_the_walk_stops_where_the_new_team_has_no_player():
    d_home, d_away, visible = _scene()
    d_away[190] = np.nan
    out = _backdate(_labels(), (d_home, d_away, visible))
    assert _first_a(out) == 191


def test_a_stretch_shorter_than_the_minimum_is_not_moved():
    # 4 frames of evidence: under BACKDATE_MIN_S (0.2 s = 5 frames at 25 fps).
    out = _backdate(_labels(), _scene(reception=CHANGE - 4))
    assert _first_a(out) == CHANGE
    out = _backdate(_labels(), _scene(reception=CHANGE - 5))
    assert _first_a(out) == CHANGE - 5


def test_the_cap_bounds_the_move():
    # Evidence from frame 50; the cap is 3.0 s = 75 frames.
    out = _backdate(_labels(), _scene(reception=50))
    assert _first_a(out) == CHANGE - 75


def test_the_spell_before_keeps_the_floor():
    # The H spell starts at 150 (D before it). The floor of 1 s keeps 25 H frames: 150..174.
    labels = _labels(start=150)
    out = _backdate(labels, _scene(reception=100))
    assert _first_a(out) == 175
    assert list(out[150:175]) == ["H"] * 25
    assert set(out[:150]) == {"D"}


def test_a_spell_already_shorter_than_the_floor_keeps_its_change():
    labels = _labels(start=CHANGE - 20)     # a 20-frame H spell after D
    out = _backdate(labels, _scene(reception=100))
    np.testing.assert_array_equal(out, labels)


def test_k2b_a_change_into_a_spell_shorter_than_the_floor_is_not_moved():
    # COMPONENT_BACKDATE_B.md: the A spell after the change lasts 20 frames, then D.
    labels = _labels()
    labels[CHANGE + 20:] = "D"
    out = _backdate(labels)
    np.testing.assert_array_equal(out, labels)


def test_k2b_a_change_into_a_spell_of_exactly_the_floor_is_moved():
    labels = _labels()
    labels[CHANGE + 25:] = "D"
    out = _backdate(labels)
    assert _first_a(out) == RECEPTION


def test_k2b_the_block_end_counts_as_the_end_of_the_spell_after():
    labels = _labels()
    out = _backdate(labels, blocks=[(1, 0, CHANGE + 9)])
    np.testing.assert_array_equal(out, labels)


def _spells(labels):
    labels = np.asarray(labels)
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(labels)]
    return {(int(a), int(b), str(labels[a])) for a, b in zip(starts, stops) if labels[a] in "HA"}


@pytest.mark.parametrize("post_floor", [1.0, 0.0])
def test_k2b_no_spell_shorter_than_the_floor_is_created_or_changed(post_floor):
    # With the post-spell floor no spell under 1 s is created or changed. Without it (K2) the
    # same scenes do change short spells, so the test can tell the two rules apart.
    cfg = dataclasses.replace(ON, BACKDATE_MIN_POST_SPELL_S=post_floor)
    rng = np.random.RandomState(1)
    changed = 0
    floor = 25
    for _ in range(50):
        lengths = rng.randint(5, 60, 40)
        letters = np.array(list("HAD"))[rng.randint(0, 3, 40)]
        labels = np.repeat(letters, lengths)
        n = len(labels)
        near_home = np.repeat(rng.uniform(0, 1, n) > 0.5, rng.randint(3, 40, n))[:n]
        d_home = np.where(near_home, 0.6, 3.0)
        d_away = np.where(near_home, 3.0, 0.6)
        visible = rng.uniform(0, 1, n) > 0.01
        out = possession.backdate_changes(labels, d_home, d_away, visible, [(1, 0, n - 1)], cfg,
                                          FPS)
        short_in = {s for s in _spells(labels) if s[1] - s[0] < floor}
        short_out = {s for s in _spells(out) if s[1] - s[0] < floor}
        changed += short_out != short_in
    assert (changed == 0) if post_floor else (changed > 0)


def test_no_move_into_or_out_of_d():
    labels = _labels()
    labels[CHANGE:] = "D"                   # H -> D: not a direct change
    labels[CHANGE + 20:] = "A"              # D -> A: not a direct change
    out = _backdate(labels, _scene(reception=CHANGE + 5))
    np.testing.assert_array_equal(out, labels)


def test_no_move_across_a_block_edge():
    blocks = [(1, 0, 179), (2, 180, N - 1)]
    labels = _labels()
    labels[180:CHANGE] = "H"
    out = _backdate(labels, _scene(reception=100), blocks=blocks)
    # The walk may not leave block 2 (it starts at 180) and must keep 25 H frames of it.
    assert _first_a(out) == CHANGE
    blocks = [(1, 0, 159), (2, 160, N - 1)]
    out = _backdate(labels, _scene(reception=100), blocks=blocks)
    assert _first_a(out) == 185


def test_frames_outside_the_blocks_never_change():
    labels = _labels()
    labels[:20] = "X"
    blocks = [(1, 20, N - 1)]
    out = _backdate(labels, _scene(reception=0), blocks=blocks)
    assert set(out[:20]) == {"X"}


def test_both_directions_and_several_changes_are_moved_independently():
    labels = np.full(N, "H", dtype="<U1")
    labels[100:250] = "A"
    d_home = np.full(N, 0.7)
    d_away = np.full(N, 5.0)
    d_home[80:230] = 5.0
    d_away[80:230] = 0.7                    # A has the ball from 80, H takes it back at 230
    records = []
    out = _backdate(labels, (d_home, d_away, np.ones(N, dtype=bool)), records=records)
    assert _edges(out) == {("H", "A"): 80, ("A", "H"): 230}
    assert records == [{"t1": 100, "t0": 80, "team": "A"},
                       {"t1": 250, "t0": 230, "team": "H"}]


def test_a_move_never_adds_or_removes_a_change():
    rng = np.random.RandomState(0)
    for _ in range(20):
        labels = np.array(list("HAD"))[rng.randint(0, 3, N // 20)].repeat(20)
        d_home = rng.uniform(0.0, 3.0, N)
        d_away = rng.uniform(0.0, 3.0, N)
        visible = rng.uniform(0, 1, N) > 0.05
        out = possession.backdate_changes(labels, d_home, d_away, visible, BLOCKS, ON, FPS)
        assert np.count_nonzero(out[1:] != out[:-1]) == np.count_nonzero(labels[1:] != labels[:-1])
        assert np.array_equal(out == "D", labels == "D")


def test_backdate_fields_are_semantic_and_on_since_c23():
    cfg = Config.default()
    assert cfg.BACKDATE_CHANGES is True     # COMPONENT_BACKDATE_B.md Section 7 (KEEP)
    assert cfg.BACKDATE_CAP_S == 2.8
    assert cfg.BACKDATE_RADIUS_M == cfg.R_TOUCH
    assert cfg.BACKDATE_MIN_S == pytest.approx(2.0 * cfg.R_TOUCH / cfg.AIRBORNE_SPEED)
    assert cfg.BACKDATE_MIN_PRE_SPELL_S == cfg.KNEE_MIN_PRE_SPELL_S
    assert cfg.BACKDATE_MIN_POST_SPELL_S == cfg.BACKDATE_MIN_PRE_SPELL_S
    for name in ("BACKDATE_CHANGES", "BACKDATE_RADIUS_M", "BACKDATE_CAP_S", "BACKDATE_MIN_S",
                 "BACKDATE_MIN_PRE_SPELL_S", "BACKDATE_MIN_POST_SPELL_S"):
        assert name in possession.SEMANTIC_FIELDS
    assert possession.detection_key(cfg) == possession.detection_key(ON)


def _run_inputs():
    """test_calibration's synthetic match with a ball and team distances: A near from 260."""
    inputs = _inputs(BASE_EVENTS)
    n = inputs["n_frames"]
    d_home = np.full(n, 0.8)
    d_away = np.full(n, 6.0)
    d_home[260:] = 5.0
    d_away[260:] = 0.9
    inputs["feats"] = dict(inputs["feats"], dH=d_home, dA=d_away)
    inputs["ball"] = {"visible": np.ones(n, dtype=bool), "fps": 25.0}
    return inputs


def test_run_moves_the_change_only_with_the_switch_on():
    from tests.test_calibration import A_EVENT, CFG

    base = dataclasses.replace(CFG, ONSET_KNEE=False, BACKDATE_CHANGES=False)
    inputs = _run_inputs()
    off = possession.run(base, inputs=inputs)
    assert _edges(off)[("H", "A")] == A_EVENT
    on, parts = possession.run(dataclasses.replace(base, BACKDATE_CHANGES=True),
                               return_parts=True, inputs=inputs)
    assert _edges(on)[("H", "A")] == 260
    assert parts["backdate"] == [{"t1": A_EVENT, "t0": 260, "team": "A"}]
    changed = np.flatnonzero(on != off)
    np.testing.assert_array_equal(changed, np.arange(260, A_EVENT))
    _, parts_off = possession.run(base, return_parts=True, inputs=inputs)
    assert parts_off["backdate"] == []

"""Unit tests for engine/decoder.py and the M3 evidence of engine/possession.py (card P1-C).

Every test builds a synthetic sequence. No test needs the match data bundle.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine import decoder, possession
from engine.config import Config

CFG = Config.default()
H, A, D = decoder.H, decoder.A, decoder.D
DEAD_FLOOR_LOG = float(np.log(CFG.P_FLOOR))   # log P(D) of a frame with no D evidence


def _log_e(z, log_dead=None):
    """(n, 3) log evidence from the H-versus-A log-odds z (H wins by exactly z per frame)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty((z.shape[0], 3))
    out[:, H] = -np.logaddexp(0.0, -z)
    out[:, A] = -np.logaddexp(0.0, z)
    out[:, D] = DEAD_FLOOR_LOG if log_dead is None else log_dead
    return out


def _runs(path):
    """[(start, stop, state)] of a path."""
    starts, stops, labels = decoder._runs(path)
    return [(int(a), int(b), int(c)) for a, b, c in zip(starts, stops, labels)]


def _costs(n, h=(), a=(), onsets=(), restarts=()):
    return decoder.switch_costs(n, CFG, {"H": np.array(h, dtype=np.int64),
                                         "A": np.array(a, dtype=np.int64)},
                                np.array(onsets, dtype=np.int64), list(restarts))


# -- switch costs and helpers -------------------------------------------------------------------

def test_near_mask_clips_at_both_ends():
    mask = decoder.near_mask([0, 9], 10, 2)
    assert mask.tolist() == [True] * 3 + [False] * 4 + [True] * 3
    assert not decoder.near_mask([], 5, 3).any()


def test_switch_costs_follow_the_events():
    costs = _costs(40, h=[10], a=[30], onsets=[20], restarts=[(5, "H")])
    w = CFG.SWITCH_EVENT_WINDOW
    assert costs[10 - w, A, H] == CFG.C_SWITCH_EVENT
    assert costs[10 + w, A, H] == CFG.C_SWITCH_EVENT
    assert costs[10 + w + 1, A, H] == CFG.C_SWITCH
    assert costs[10, H, A] == CFG.C_SWITCH                  # an H event does not help H -> A
    assert costs[30, H, A] == CFG.C_SWITCH_EVENT
    assert costs[20 + CFG.DEAD_EVENT_WINDOW, H, D] == CFG.C_DEAD_EVENT
    assert costs[20 + CFG.DEAD_EVENT_WINDOW + 1, A, D] == CFG.C_DEAD
    assert costs[5, D, H] == CFG.C_ALIVE_EVENT - CFG.RESTART_TEAM_BONUS
    assert costs[5, D, A] == CFG.C_ALIVE_EVENT
    assert costs[30, D, A] == CFG.C_ALIVE
    assert np.all(costs[:, [0, 1, 2], [0, 1, 2]] == 0.0)


# -- the card's synthetic cases -----------------------------------------------------------------

def test_constant_evidence_gives_one_segment():
    path = decoder.decode(_log_e(np.full(300, 1.0)), _costs(300), CFG)
    assert _runs(path) == [(0, 300, H)]
    path = decoder.decode(_log_e(np.full(300, -0.2)), _costs(300, h=[100, 200]), CFG)
    assert _runs(path) == [(0, 300, A)]


def test_flip_with_an_event_switches_at_the_event_frame():
    # 5 frames of A evidence at the end cannot pay C_SWITCH, but they pay C_SWITCH_EVENT.
    z = np.concatenate([np.full(100, 2.0), np.full(15, -0.4)])
    assert 15 * 0.4 < CFG.C_SWITCH and 15 * 0.4 > CFG.C_SWITCH_EVENT
    assert _runs(decoder.decode(_log_e(z), _costs(115), CFG)) == [(0, 115, H)]
    path = decoder.decode(_log_e(z), _costs(115, a=[100]), CFG)
    assert _runs(path) == [(0, 100, H), (100, 115, A)]


def test_flip_without_an_event_needs_c_switch_over_log_odds_frames():
    r = 0.4                                        # per-frame log-odds of the flipped evidence
    threshold = CFG.C_SWITCH / r                   # 20 frames with the card values
    short = int(np.floor(threshold)) - 2
    long = int(np.ceil(threshold)) + 2
    assert short >= CFG.MIN_POSS                   # the post-pass plays no part in the result
    for flip, switches in ((short, False), (long, True)):
        z = np.concatenate([np.full(100, 2.0), np.full(flip, -r)])
        path = decoder.decode(_log_e(z), _costs(z.size), CFG)
        expected = [(0, 100, H), (100, 100 + flip, A)] if switches else [(0, 100 + flip, H)]
        assert _runs(path) == expected, (flip, threshold)


def test_twelve_frame_possession_survives_and_eight_frame_one_is_merged():
    for length, survives in ((12, True), (8, False)):
        z = np.concatenate([np.full(50, 2.0), np.full(length, -2.0), np.full(50, 2.0)])
        costs = _costs(z.size, h=[50 + length], a=[50])
        raw = decoder.viterbi(_log_e(z), costs)
        assert _runs(raw) == [(0, 50, H), (50, 50 + length, A), (50 + length, z.size, H)]
        path = decoder.decode(_log_e(z), costs, CFG)
        if survives:
            assert _runs(path) == _runs(raw)
        else:
            assert _runs(path) == [(0, z.size, H)]


def test_eight_second_ball_gap_with_low_dead_evidence_holds_the_owner():
    fps = 25
    n = 400
    gap = slice(100, 100 + 8 * fps)                # 8 s without a ball and without an event
    owner = possession.owner_track([(0, "H")], [], [], [(1, 0, n - 1)], n)
    feats = {"dmin": np.full(n, 0.8), "nearest_team": np.zeros(n, dtype=np.int8)}
    feats["dmin"][gap] = np.nan
    feats["nearest_team"][gap] = -1
    near = possession.nearest_log_odds(feats, np.zeros(n, dtype=bool), CFG)
    unknown = possession.stretch_mean(near, owner == possession.UNKNOWN, [0])
    e_dead = np.full(n, 0.05)
    e_dead[gap] = 0.3                              # low: players still move in the gap
    p_dead = possession.dead_probability(e_dead, np.zeros(n, dtype=bool), CFG)
    log_e = possession.emissions(owner, unknown, np.zeros(n), p_dead, CFG)
    labels = np.array(decoder.STATES)[decoder.decode(log_e, _costs(n), CFG)]
    assert set(labels.tolist()) == {"H"}           # no D, no X, the owner is held


def test_dead_segment_gets_its_boundaries_at_the_onset_and_the_restart():
    n, onset, restart = 200, 60, 120
    log_dead = np.full(n, -10.0)
    log_dead[onset:restart] = 0.0
    z = np.full(n, 3.0)
    log_e = _log_e(z)
    log_e[:, D] = log_dead
    alive = log_e[onset:restart, H].mean()
    advantage = (0.0 - alive) * (restart - onset)  # D evidence summed over the stoppage
    assert advantage < CFG.C_DEAD + CFG.C_ALIVE    # without events the stoppage does not pay
    assert _runs(decoder.decode(log_e, _costs(n), CFG)) == [(0, n, H)]
    costs = _costs(n, onsets=[onset], restarts=[(restart, "H")])
    path = decoder.decode(log_e, costs, CFG)
    assert _runs(path) == [(0, onset, H), (onset, restart, D), (restart, n, H)]


def test_decoder_is_deterministic_and_breaks_ties_in_the_card_order():
    rng = np.random.RandomState(7)                 # synthetic input only, seed held at 7
    z = rng.normal(0.0, 1.5, size=2000)
    log_e = _log_e(z, log_dead=rng.normal(-1.0, 1.0, size=2000))
    costs = _costs(2000, h=rng.randint(0, 2000, 40), a=rng.randint(0, 2000, 40),
                   onsets=rng.randint(0, 2000, 10), restarts=[(500, "A"), (1500, None)])
    first = decoder.decode(log_e, costs, CFG)
    second = decoder.decode(log_e.copy(), costs.copy(), CFG)
    assert np.array_equal(first, second)
    flat = np.zeros((50, 3))
    assert _runs(decoder.decode(flat, _costs(50), CFG)) == [(0, 50, H)]   # H first on a tie
    forced = np.full(50, decoder.NO_FORCE)
    forced[:10] = D
    assert _runs(decoder.decode(flat, _costs(50), CFG, forced)) == [(0, 50, D)]  # stay on a tie


def test_forced_frames_keep_their_state():
    z = np.full(60, 3.0)
    forced = np.full(60, decoder.NO_FORCE)
    forced[:30] = D
    path = decoder.decode(_log_e(z), _costs(60, restarts=[(30, None)]), CFG, forced)
    assert _runs(path) == [(0, 30, D), (30, 60, H)]


def test_post_pass_merges_into_the_cheaper_neighbour_and_prefers_the_left_one():
    n = 60
    states = np.array([H] * 25 + [A] * 5 + [D] * 30, dtype=np.int8)
    log_e = np.zeros((n, 3))
    log_e[25:30, D] = 1.0                          # the short A run looks more like D
    out = decoder.min_durations(states, log_e, _costs(n), CFG)
    assert _runs(out) == [(0, 25, H), (25, 60, D)]
    # Equal evidence: both merges remove C_SWITCH + C_DEAD of switches and add C_DEAD back, so
    # the losses are equal and the left neighbour (H) wins.
    left = decoder._relabel_loss(np.zeros((n, 3)), _costs(n), states, 25, 30, H)
    right = decoder._relabel_loss(np.zeros((n, 3)), _costs(n), states, 25, 30, D)
    assert left == right
    out = decoder.min_durations(states, np.zeros((n, 3)), _costs(n), CFG)
    assert _runs(out) == [(0, 30, H), (30, 60, D)]
    single = np.full(5, A, dtype=np.int8)
    assert np.array_equal(decoder.min_durations(single, np.zeros((5, 3)), _costs(5), CFG), single)


# -- M3 evidence (engine/possession.py) ---------------------------------------------------------

def test_owner_track_rules():
    blocks = [(1, 0, 199)]
    owner = possession.owner_track([(20, "H"), (120, "A")], [(170, "H")], [(60, "A"), (90, "H")],
                                   blocks, 220, kick_teams={1: "A"}, kickoffs={1: 0})
    assert set(owner[0:20].tolist()) == {1}                # the kickoff team
    assert set(owner[20:60].tolist()) == {0}               # an owner event
    assert set(owner[60:120].tolist()) == {possession.UNKNOWN}  # a new-team touch
    assert set(owner[120:170].tolist()) == {1}
    assert set(owner[170:200].tolist()) == {0}             # a restart with a team
    assert set(owner[200:].tolist()) == {possession.UNKNOWN}  # outside the block
    same_team = possession.owner_track([(20, "H")], [], [(60, "H")], blocks, 200)
    assert set(same_team[20:].tolist()) == {0}             # a touch of the owner changes nothing
    assert set(same_team[:20].tolist()) == {possession.UNKNOWN}  # no kickoff team


def test_team_event_rows_and_kickoff_teams():
    table = pd.DataFrame({"frame": [5, 9, 12, 14], "kind": ["touch", "release", "contested",
                                                             "control"],
                          "team": ["H", "A", None, "A"], "confidence": [0.4, 0.3, 0.8, 0.7],
                          "dist_m": [0.5] * 4, "notes": ["", "deflection", "", "end=20"]})
    assert possession.team_event_rows(table) == [(5, "H"), (9, "A"), (14, "A")]
    assert possession.team_event_rows(table, deflections=False) == [(5, "H"), (14, "A")]
    cands = pd.DataFrame({"frame": [103, 220, 400], "team": ["A", "H", None],
                          "kind": ["reappear", "kick", "kick"], "confidence": [0.6] * 3,
                          "confirmed": [True, True, True]})
    rules = possession.kickoff_decisions({1: 100, 2: 200}, cands, CFG)
    assert rules[1][:2] == ("A", "restart")
    assert rules[2][:2] == ("H", "law")                    # 220 is outside ALIVE_EVENT_WINDOW
    assert possession.kickoff_teams({1: 100, 2: 200}, cands, CFG) == {1: "A", 2: "H"}
    assert possession.kickoff_teams({2: 200}, cands, CFG) == {2: None}   # no partner half


# -- kickoff team from the centre mark (RESEARCH 4 rank 3, Law 8) ---------------------------------

def _mark(n, slots):
    """tracks.mark_nearest() of a synthetic 100 m x 50 m pitch.

    `slots` is a list of (team, x, y) arrays of length n in metres (NaN = not visible).
    """
    from engine import tracks
    pitch = (100.0, 50.0)
    px = np.stack([np.asarray(x, float) for _, x, _ in slots], axis=1) / pitch[0]
    py = np.stack([np.asarray(y, float) for _, _, y in slots], axis=1) / pitch[1]
    tactical = {"px": px, "py": py, "team_of_slot": np.array([t for t, _, _ in slots])}
    return tracks.mark_nearest(tactical, pitch)


def test_mark_nearest_and_mark_team_find_the_taker():
    """An A taker stands 0.4 m from the mark; an H player waits 10 m away; kickoff at 100."""
    n = 200
    taker_x = np.full(n, 50.4)
    taker_x[:20] = np.nan                                  # the taker is not tracked early on
    slots = [(1, taker_x, np.full(n, 25.0)),               # A at the mark
             (0, np.full(n, 40.0), np.full(n, 25.0)),      # H outside the centre circle
             (0, np.full(n, np.nan), np.full(n, np.nan))]  # a ghost slot
    mark = _mark(n, slots)
    assert np.isclose(mark["dist"][50], 0.4) and mark["team"][50] == 1
    assert np.isclose(mark["dist"][10], 10.0) and mark["team"][10] == 0
    team, dist = possession.mark_team(mark, 100, CFG)
    assert team == "A" and np.isclose(dist, 0.4)
    rules = possession.kickoff_decisions({1: 100}, _no_candidates(), CFG, mark)
    assert rules[1][:2] == ("A", "mark")
    # Before frame 20 only the H player 10 m away is tracked: farther than KICKOFF_TAKER_M.
    assert possession.mark_team(mark, 15, CFG)[0] is None
    assert possession.mark_team(None, 100, CFG) == (None, float("inf"))


def test_mark_rule_wins_over_the_restart_candidate_and_the_law_fills_the_other_half():
    n = 400
    taker_x = np.where(np.arange(n) < 200, 50.3, 70.0)     # H at the mark before kickoff 1
    slots = [(0, taker_x, np.full(n, 25.0)), (1, np.full(n, 60.0), np.full(n, 25.0))]
    mark = _mark(n, slots)
    cands = pd.DataFrame({"frame": [102, 302], "team": ["A", "A"], "kind": ["kick", "kick"],
                          "confidence": [0.6, 0.6], "confirmed": [True, True]})
    rules = possession.kickoff_decisions({1: 100, 2: 300}, cands, CFG, mark)
    assert rules[1][:2] == ("H", "mark")                   # the mark beats the A candidate
    assert rules[2][:2] == ("A", "restart")                # nobody at the mark: the candidate
    alone = possession.kickoff_decisions({1: 100, 2: 300}, _no_candidates(), CFG, mark)
    assert alone[2][:2] == ("A", "law")                    # Law 8: the other team


def test_goal_kickoffs_give_the_restart_after_a_goals_window_the_mark_team():
    """A GOALS window ends at 150 inside a stoppage 120-180; the restart at 180 has team H, but
    an A player stands at the mark: the kickoff after the goal is A at 180."""
    n = 300
    blocks = [(1, 0, n - 1)]
    stoppage = np.zeros(n, dtype=bool)
    stoppage[120:180] = True
    slots = [(1, np.full(n, 50.5), np.full(n, 25.0)), (0, np.full(n, 30.0), np.full(n, 25.0))]
    mark = _mark(n, slots)
    tags = pd.DataFrame({"code": ["GOALS", "GOALS", "GOALS"], "team": ["H", "A", "H"],
                         "start_frame": [10, 20, 200], "end_frame": [150, 60, 260]})
    restart_list = [(180, "H")]
    rows = possession.goal_kickoffs(tags, stoppage, restart_list, blocks, CFG, mark)
    assert [(f, t, r) for f, t, r, _ in rows] == [(180, "A", "mark")]   # 60, 260: alive frames
    new_list, new_events = possession.apply_goal_kickoffs(restart_list, [(178, "H"), (180, "H")],
                                                          rows)
    assert new_list == [(180, "A")]
    assert new_events == [(178, "H"), (180, "A")]
    # Without anybody at the mark the restart keeps its own team.
    far = _mark(n, [(1, np.full(n, 70.0), np.full(n, 25.0))])
    rows = possession.goal_kickoffs(tags, stoppage, restart_list, blocks, CFG, far)
    assert [(f, t, r) for f, t, r, _ in rows] == [(180, "H", "restart")]
    assert possession.apply_goal_kickoffs(restart_list, [], rows) == (restart_list, [(180, "H")])
    # A stoppage that runs to the end of the block has no kickoff.
    stoppage[120:] = True
    assert possession.goal_kickoffs(tags, stoppage, restart_list, blocks, CFG, mark) == []


def _no_candidates():
    return pd.DataFrame({"frame": pd.Series([], dtype=np.int64), "team": pd.Series([], dtype=object),
                         "kind": pd.Series([], dtype=object),
                         "confidence": pd.Series([], dtype=np.float64),
                         "confirmed": pd.Series([], dtype=bool)})


def test_nearest_log_odds_is_delayed_and_graded_by_distance():
    n = 60
    feats = {"dmin": np.full(n, 0.5), "nearest_team": np.full(n, 1, dtype=np.int8)}
    feats["dmin"][30:] = 3.0
    feats["nearest_team"][30:] = 0
    feats["dmin"][50:] = 7.0
    ghost = np.zeros(n, dtype=bool)
    out = possession.nearest_log_odds(feats, ghost, CFG)
    lag = CFG.NEAREST_LAG
    close = np.log(CFG.P_NEAR_CLOSE / (1 - CFG.P_NEAR_CLOSE))
    far = np.log(CFG.P_NEAR_FAR / (1 - CFG.P_NEAR_FAR))
    assert np.all(out[:lag] == 0.0)
    assert np.allclose(out[lag:30 + lag], -close)          # A within 1 m, 20 frames later
    assert np.allclose(out[30 + lag:50 + lag], far)
    ghost[:] = True
    assert not possession.nearest_log_odds(feats, ghost, CFG).any()


def test_stretch_mean_is_constant_between_events():
    values = np.array([1.0, 1.0, 3.0, 3.0, -2.0, -2.0, 5.0, 5.0])
    unknown = np.array([False, True, True, True, True, True, True, False])
    out = possession.stretch_mean(values, unknown, boundaries=[4])
    assert np.allclose(out, [0.0, 7.0 / 3, 7.0 / 3, 7.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0])


def test_tag_log_odds_and_dead_probability():
    tags = pd.DataFrame({"code": ["PROGRESSION", "BUILD UP", "DEFENDING IN MIDDLE THIRD",
                                  "SET PIECES"],
                         "team": ["H", "A", "A", "H"], "start_frame": [0, 5, 0, 0],
                         "end_frame": [10, 15, 20, 20]})
    out = possession.tag_log_odds(tags, 20, CFG)
    step = np.log(CFG.TAG_TEAM_PRIOR_RATIO)
    assert np.allclose(out[:5], step) and np.allclose(out[5:10], 0.0)
    assert np.allclose(out[10:15], -step) and np.allclose(out[15:], 0.0)
    e_dead = np.array([0.05, 0.6, 0.95, 0.3])
    assert np.allclose(possession.dead_probability(e_dead, None, CFG), e_dead)
    stop = np.array([False, False, True, True])
    p = possession.dead_probability(e_dead, stop, CFG)
    assert np.allclose(p, [0.05, 1 - CFG.P_DEAD_STOPPAGE, 0.95, CFG.P_DEAD_STOPPAGE])


def test_emissions_use_the_owner_where_known_and_the_stretch_value_elsewhere():
    owner = np.array([0, 1, possession.UNKNOWN], dtype=np.int8)
    log_e = possession.emissions(owner, np.array([0.0, 0.0, 1.5]), np.zeros(3),
                                 np.array([0.0, 0.5, 1.0]), CFG)
    step = np.log(CFG.P_OWNER / (1 - CFG.P_OWNER))
    assert np.isclose(log_e[0, H] - log_e[0, A], step)
    assert np.isclose(log_e[1, A] - log_e[1, H], step)
    assert np.isclose(log_e[2, H] - log_e[2, A], 1.5)
    assert np.isclose(log_e[0, D], np.log(CFG.P_FLOOR))   # floored, finite
    assert np.all(np.isfinite(log_e))


def test_owner_events_ignore_flight_and_m2_refuses_it():
    table = pd.DataFrame({"frame": [5], "kind": ["release"], "team": ["H"], "confidence": [0.8],
                          "dist_m": [0.5], "notes": [""]})
    flight = dataclasses.replace(CFG, FLIGHT_ASSIGNMENT=True)
    assert possession.owner_events(table, flight) == [(5, "H")]   # run() applies flights later
    with pytest.raises(NotImplementedError):
        possession.run_m2(flight)                                  # M2 stays as it was scored


def test_dribble_of_the_other_team_opens_a_switch():
    """An H release, then an A dribble at frame 60 (RESEARCH 4 rank 2)."""
    n = 120
    table = pd.DataFrame({"frame": [10, 60], "kind": ["release", "dribble"],
                          "team": ["H", "A"], "confidence": [0.8, 0.78], "dist_m": [0.5, 0.9],
                          "notes": ["", "end=90"]})
    rows = possession.team_event_rows(table, kinds=possession.RULE_TEAM_KINDS[CFG.POSSESSION_RULE])
    assert rows == [(10, "H"), (60, "A")]
    owners = possession.owner_events(table, CFG)
    assert owners == [(10, "H"), (60, "A")]
    costs = _costs(n, h=[f for f, t in rows if t == "H"], a=[f for f, t in rows if t == "A"])
    assert costs[60, H, A] == CFG.C_SWITCH_EVENT
    owner = possession.owner_track(owners, [], rows, [(1, 0, n - 1)], n)
    log_e = possession.emissions(owner, np.zeros(n), np.zeros(n), np.zeros(n), CFG)
    assert _runs(decoder.decode(log_e, costs, CFG)) == [(0, 60, H), (60, n, A)]
    release = dataclasses.replace(CFG, POSSESSION_RULE="release")
    assert possession.owner_events(table, release) == [(10, "H")]
    assert possession.team_event_rows(table, kinds=possession.RULE_TEAM_KINDS["release"]) == [
        (10, "H")]


def test_bare_touch_of_the_other_team_does_not_release_the_owner():
    """An H release at 10, then an A touch at 60 with full confidence (C7, RESEARCH 4 rank 4).

    The delayed nearest player favours A after the touch. Only a release, control or dribble of
    A that is not an owner event makes the owner unknown. A bare touch keeps the owner H.
    """
    n = 200
    table = pd.DataFrame({"frame": [10, 60, 120], "kind": ["release", "touch", "control"],
                          "team": ["H", "A", "A"], "confidence": [0.8, 0.85, 0.3],
                          "dist_m": [0.5, 0.4, 0.6], "notes": ["", "", "end=130"]})
    assert "touch" not in CFG.OWNER_RELEASE_KINDS
    assert possession.release_rows(table, CFG) == [(10, "H"), (120, "A")]
    owners = possession.owner_events(table, CFG)
    assert owners == [(10, "H")]                           # the 0.3 control sets no owner
    rows = possession.team_event_rows(table, kinds=possession.RULE_TEAM_KINDS[CFG.POSSESSION_RULE])
    assert rows == [(10, "H"), (60, "A"), (120, "A")]      # the touch still opens a cheap switch
    blocks = [(1, 0, n - 1)]
    near = np.where(np.arange(n) >= 60, -2.0, 0.0)         # A is nearest after the touch

    def decoded(cfg):
        owner = possession.owner_track(owners, [], possession.release_rows(table, cfg), blocks, n)
        unknown_lo = possession.stretch_mean(near, owner == possession.UNKNOWN,
                                             [f for f, _ in rows])
        log_e = possession.emissions(owner, unknown_lo, np.zeros(n), np.zeros(n), cfg)
        costs = _costs(n, h=[f for f, t in rows if t == "H"], a=[f for f, t in rows if t == "A"])
        return owner, _runs(decoder.decode(log_e, costs, cfg))

    owner, path = decoded(CFG)
    assert set(owner[10:120].tolist()) == {0}              # the touch keeps the owner H
    assert set(owner[120:].tolist()) == {possession.UNKNOWN}   # the control releases it
    assert path == [(0, 120, H), (120, n, A)]
    with_touch = dataclasses.replace(CFG, OWNER_RELEASE_KINDS=CFG.OWNER_RELEASE_KINDS + ("touch",))
    owner, path = decoded(with_touch)                      # the c6_kickoff behaviour
    assert set(owner[60:].tolist()) == {possession.UNKNOWN}
    assert path == [(0, 60, H), (60, n, A)]
    release = dataclasses.replace(CFG, POSSESSION_RULE="release")
    assert possession.release_rows(table, release) == [(10, "H")]
    assert "OWNER_RELEASE_KINDS" in possession.SEMANTIC_FIELDS


def test_growing_team_confidence_cannot_shorten_a_stoppage():
    """IDEAS-REVIEW Blend D: owner evidence inside a stoppage never shortens it.

    The dead-ball module sets the stoppage [200, 400). run() floors P(D) there at
    P_DEAD_STOPPAGE (dead_probability). log P(H) and log P(A) are at most log(1 - P(D)),
    whatever the team log-odds, and log(1 - P(D)) < log P(D) when P_DEAD_STOPPAGE > 0.5. The
    test raises P_OWNER and the tag prior inside the stoppage step by step, and adds team
    events inside it. The D run keeps its two edges every time.
    """
    n, onset, restart = 600, 200, 400
    blocks = [(1, 0, n - 1)]
    stoppage = np.zeros(n, dtype=bool)
    stoppage[onset:restart] = True
    e_dead = np.full(n, 0.02)
    e_dead[onset:restart] = 0.2                    # weak D evidence: the floor carries it
    assert CFG.P_DEAD_STOPPAGE > 0.5
    owner = possession.owner_track([(0, "H")], [(restart, "H")], [], blocks, n)
    expected = [(0, onset, H), (onset, restart, D), (restart, n, H)]
    for p_owner in (0.6, 0.9, 0.99, 0.999999):
        cfg = dataclasses.replace(CFG, P_OWNER=p_owner)
        p_dead = possession.dead_probability(e_dead, stoppage, cfg)
        for boost in (0.0, 5.0, 50.0, 500.0):
            tag_lo = np.zeros(n)
            tag_lo[onset:restart] = boost          # more H confidence inside the stoppage only
            log_e = possession.emissions(owner, np.zeros(n), tag_lo, p_dead, cfg)
            alive_best = np.maximum(log_e[onset:restart, H], log_e[onset:restart, A])
            assert np.all(alive_best < log_e[onset:restart, D])
            for events in ((), (250, 300, 350)):
                costs = _costs(n, h=list(events), a=list(events), onsets=[onset],
                               restarts=[(restart, "H")])
                path = decoder.decode(log_e, costs, cfg)
                assert _runs(path) == expected, (p_owner, boost, events)

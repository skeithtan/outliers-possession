"""Unit tests for the GT-semantics switches of engine/possession.py (card P2).

Every test builds a synthetic sequence. No test needs the match data bundle. The synthetic
match has one graded block of N frames. An H release at H_KICK starts the H possession, an A
event at A_EVENT gives the ball to A, an out onset at ONSET starts a stoppage, and a confirmed
restart candidate at RESTART ends it.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine import possession
from engine.config import Config

N = 600
BLOCKS = [(1, 0, N - 1)]
H_KICK, A_EVENT, ONSET, RESTART = 100, 300, 400, 480
# The rule engine (c23) explicitly: these are tests of its possession steps, and the default
# LEARNED_MODE is the learned candidate since branch r6e-adopt.
CFG = dataclasses.replace(Config.default(), LEARNED_MODE="off",
                          POSSESSION_RULE="release_or_control", FLIGHT_ASSIGNMENT=False,
                          DELTA_SWITCH=0, DELTA_DEAD=0, DELTA_RESTART_KICK=0,
                          DELTA_RESTART_REAPPEAR=0)


def _events(rows):
    """Event table from (frame, kind, team, confidence, notes) tuples."""
    table = pd.DataFrame([(f, k, t, c, 0.5, n) for f, k, t, c, n in rows],
                         columns=["frame", "kind", "team", "confidence", "dist_m", "notes"])
    table = table.astype({"frame": np.int64, "confidence": np.float64, "dist_m": np.float64})
    table["team"] = table["team"].astype(object).where(table["team"].notna(), None)
    return table


def _inputs(events, onsets=((ONSET, "out"),), restarts=((RESTART, "reappear", "H"),)):
    """A synthetic prepare() result for run()."""
    onset_table = pd.DataFrame([(f, m, 1.0) for f, m in onsets],
                               columns=["frame", "mechanism", "confidence"])
    onset_table = onset_table.astype({"frame": np.int64, "confidence": np.float64})
    cands = pd.DataFrame([(f, t, k, 0.6, True) for f, k, t in restarts],
                         columns=["frame", "team", "kind", "confidence", "confirmed"])
    cands = cands.astype({"frame": np.int64, "confidence": np.float64, "confirmed": bool})
    return {"key": possession.detection_key(CFG), "n_frames": N, "blocks": BLOCKS,
            "kickoffs": {1: 0}, "fps": 25.0,
            "feats": {"dmin": np.full(N, np.nan), "nearest_team": np.full(N, -1)},
            "tagging": pd.DataFrame(columns=["code", "team", "start_frame", "end_frame"]),
            "e_dead": np.full(N, 0.05), "events": _events(events), "onsets": onset_table,
            "restarts": cands, "ghost": np.zeros(N, dtype=bool)}


BASE_EVENTS = [(H_KICK, "release", "H", 0.8, ""), (A_EVENT, "release", "A", 0.8, "")]


def _edges(labels):
    """{(from, to): first frame of the new label} of a label vector (one change per pair)."""
    labels = np.asarray(labels)
    idx = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return {(str(labels[i - 1]), str(labels[i])): int(i) for i in idx}


def _run(inputs, **changes):
    return possession.run(dataclasses.replace(CFG, **changes), inputs=inputs)


# -- shifts ------------------------------------------------------------------------------------

def test_shift_frames_stays_inside_the_block_of_each_frame():
    blocks = [(1, 10, 19), (2, 30, 39)]
    out = possession.shift_frames([12, 18, 31, 25, 39], 5, blocks)
    assert out.tolist() == [17, 19, 36, 25, 39]          # 25 is outside every block
    assert possession.shift_frames([12, 31], -5, blocks).tolist() == [10, 30]


def test_shift_restarts_moves_each_kind_by_its_own_offset():
    cands = _inputs([])["restarts"]
    cands = pd.concat([cands, pd.DataFrame([(200, "A", "kick", 0.6, True)],
                                           columns=cands.columns)], ignore_index=True)
    cfg = dataclasses.replace(CFG, DELTA_RESTART_KICK=7, DELTA_RESTART_REAPPEAR=-3)
    out = possession.shift_restarts(cands, cfg, BLOCKS)
    assert out[["frame", "kind"]].values.tolist() == [[207, "kick"], [RESTART - 3, "reappear"]]
    assert possession.shift_restarts(cands, CFG, BLOCKS) is cands


def test_reference_run_has_the_expected_edges():
    edges = _edges(_run(_inputs(BASE_EVENTS)))
    assert edges[("H", "A")] == A_EVENT
    assert edges[("A", "D")] == ONSET
    assert edges[("D", "H")] == RESTART


@pytest.mark.parametrize("field, delta, moved", [
    ("DELTA_SWITCH", 3, ("H", "A")),
    ("DELTA_SWITCH", -4, ("H", "A")),
    ("DELTA_DEAD", 5, ("A", "D")),
    ("DELTA_DEAD", -5, ("A", "D")),
    ("DELTA_RESTART_REAPPEAR", 10, ("D", "H")),
])
def test_each_offset_moves_its_own_kind_only(field, delta, moved):
    inputs = _inputs(BASE_EVENTS)
    before = _edges(_run(inputs))
    after = _edges(_run(inputs, **{field: delta}))
    assert set(before) == set(after)
    for pair, frame in before.items():
        assert after[pair] == frame + (delta if pair == moved else 0), pair


def test_kick_offset_does_not_move_a_reappear_exit():
    inputs = _inputs(BASE_EVENTS)
    assert _edges(_run(inputs, DELTA_RESTART_KICK=10)) == _edges(_run(inputs))


def test_offsets_never_change_which_restart_ends_the_stoppage():
    # A reappear candidate 10 frames after the onset is inside RESTART_ARM_S (25 frames), so the
    # M1 state machine skips it. A +20 shift must not make it the exit.
    early = ONSET + 10
    inputs = _inputs(BASE_EVENTS, restarts=((early, "reappear", "A"), (RESTART, "kick", "H")))
    edges = _edges(_run(inputs, DELTA_RESTART_REAPPEAR=20))
    assert edges[("D", "H")] == RESTART


def test_collective_stop_entries_and_kickoffs_do_not_move():
    stoppage = np.zeros(N, dtype=bool)
    stoppage[0:50] = True                   # before a kickoff at 50
    stoppage[200:260] = True                # entry on a collective_stop row only
    onsets = pd.DataFrame({"frame": [200], "mechanism": ["collective_stop"],
                           "confidence": [0.3]})
    cands = _inputs([])["restarts"]
    cfg = dataclasses.replace(CFG, DELTA_DEAD=-5, DELTA_RESTART_REAPPEAR=10)
    out, restarts = possession.shift_stoppage(stoppage, [(50, None)], onsets, cands, cfg,
                                              BLOCKS, {1: 50})
    assert np.array_equal(out, stoppage)
    assert restarts == [(50, None)]


# -- flight assignment -------------------------------------------------------------------------

def test_flight_assignment_paints_the_flight_for_the_interceptor():
    events = [(H_KICK, "release", "H", 0.8, ""), (200, "release", "H", 0.8, ""),
              (250, "control", "A", 0.8, "end=260")]
    inputs = _inputs(events)
    assert _edges(_run(inputs))[("H", "A")] == 250
    labels, parts = possession.run(dataclasses.replace(CFG, FLIGHT_ASSIGNMENT=True),
                                   return_parts=True, inputs=inputs)
    assert _edges(labels)[("H", "A")] == 200
    assert parts["flights"] == [(200, "H", "A", 250)]


def test_flight_assignment_stops_at_a_dead_ball_onset():
    # The release at 360 is followed by the onset at 400 and an A event inside D (17380).
    events = BASE_EVENTS[:1] + [(360, "release", "H", 0.8, ""), (420, "control", "A", 0.8, "")]
    inputs = _inputs(events)
    labels, parts = possession.run(dataclasses.replace(CFG, FLIGHT_ASSIGNMENT=True),
                                   return_parts=True, inputs=inputs)
    assert parts["flights"] == []
    assert parts["flights_blocked"] == [(360, "H", ONSET)]
    assert set(labels[H_KICK + 20:ONSET].tolist()) == {"H"}


@pytest.mark.parametrize("next_event", [
    (250, "contested", None, 0.8, ""),
    (250, "touch", "A", 0.8, ""),           # a touch of the other team does not set the owner
    (250, "release", "H", 0.8, ""),         # a pass to a team mate
])
def test_flight_assignment_needs_an_owner_event_of_the_other_team_next(next_event):
    events = [(H_KICK, "release", "H", 0.8, ""), (200, "release", "H", 0.8, ""), next_event,
              (300, "control", "A", 0.8, "")]
    _, parts = possession.run(dataclasses.replace(CFG, FLIGHT_ASSIGNMENT=True),
                              return_parts=True, inputs=_inputs(events))
    assert [f for f in parts["flights"] if f[0] == 200] == []


def test_apply_flights_changes_the_team_of_the_release_only():
    rows = [(10, "H"), (20, "H"), (30, "A")]
    assert possession.apply_flights(rows, [(20, "H", "A", 30)]) == [(10, "H"), (20, "A"),
                                                                     (30, "A")]


# -- possession rule ---------------------------------------------------------------------------

def test_release_rule_ignores_controls():
    events = [(H_KICK, "release", "H", 0.8, ""), (A_EVENT, "control", "A", 0.8, "end=310")]
    table = _events(events)
    release = dataclasses.replace(CFG, POSSESSION_RULE="release")
    assert possession.owner_events(table, CFG) == [(H_KICK, "H"), (A_EVENT, "A")]
    assert possession.owner_events(table, release) == [(H_KICK, "H")]
    kinds = possession.RULE_TEAM_KINDS["release"]
    assert possession.team_event_rows(table, kinds=kinds) == [(H_KICK, "H")]
    inputs = _inputs(events)
    assert _edges(_run(inputs))[("H", "A")] == A_EVENT
    labels = _run(inputs, POSSESSION_RULE="release")
    assert set(labels[H_KICK + 20:ONSET].tolist()) == {"H"}


def test_inputs_from_another_detection_config_are_refused():
    inputs = _inputs(BASE_EVENTS)
    with pytest.raises(ValueError):
        possession.run(dataclasses.replace(CFG, DV_TOUCH=2.0), inputs=inputs)
    assert "DELTA_SWITCH" in possession.SEMANTIC_FIELDS

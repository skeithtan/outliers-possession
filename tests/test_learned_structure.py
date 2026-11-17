"""The match-structure step of mode "model" (Config.LEARNED_STRUCTURE, R3_DEVIATION_STRUCTURE.md).

Rule (a): the frames of a graded block before its kickoff of games.py are D, the rule of
possession.dead_intervals() and possession.decode_blocks() in c22 and c23. Rule (b), dead ball
from a GOALS tag to the kick-off that follows, has no counterpart in c23 (GOALS windows are never
a D mask there) and is not part of the step: a test checks that the step reads no tagging.
numpy only, synthetic inputs, no match data.
"""

import dataclasses
import inspect
import json

import numpy as np
import pytest

from engine import config, data, learned, possession
from engine.config import Config


def _probs(n, h=0.9):
    """(n, 3) frame probabilities that say H with probability h everywhere."""
    p = np.empty((n, 3))
    p[:, 0], p[:, 1], p[:, 2] = h, (1 - h) / 2, (1 - h) / 2
    return p


# -- rule (a): the mask ---------------------------------------------------------------------------

def test_pre_kickoff_mask_follows_the_c23_condition():
    blocks = [(1, 10, 99), (2, 200, 299)]
    # half 1 kicks off at its first frame (no mask), half 2 at 250: [200, 250) is masked
    mask = learned.pre_kickoff_mask(320, blocks, {1: 10, 2: 250})
    assert mask.sum() == 50 and mask[200:250].all() and not mask[250] and not mask[:200].any()
    # a kickoff after the first frame of half 1 masks [first, kick) there too
    mask = learned.pre_kickoff_mask(320, blocks, {1: 13, 2: 200})
    assert np.flatnonzero(mask).tolist() == [10, 11, 12]
    # first < kick <= last: a kickoff outside its block, or a missing half, masks nothing
    for kick in ({1: 5, 2: 300}, {1: 100}, {}):
        assert not learned.pre_kickoff_mask(320, blocks, kick).any()
    # kick == last masks all but the last frame, as dead_intervals() does
    assert learned.pre_kickoff_mask(320, blocks, {2: 299}).sum() == 99


def test_mask_equals_the_forced_frames_of_c23():
    """possession.dead_intervals() (c22, c23) sets dead[first:kick] with the same condition."""
    import pandas as pd
    blocks = [(1, 0, 99), (2, 120, 239)]
    kickoffs = {1: 0, 2: 170}
    cfg = Config.default()
    onsets = pd.DataFrame({"frame": np.array([], dtype=np.int64),
                           "mechanism": np.array([], dtype=object)})
    cands = pd.DataFrame({"frame": np.array([], dtype=np.int64), "team": [], "kind": [],
                          "confidence": [], "confirmed": np.array([], dtype=bool)})
    dead, restarts = possession.dead_intervals(260, onsets, np.zeros(260), cands, blocks,
                                               kickoffs, cfg, 25.0)
    assert np.array_equal(dead, learned.pre_kickoff_mask(260, blocks, kickoffs))
    assert restarts == [(170, None)]


# -- rule (a): the decoder ------------------------------------------------------------------------

def test_forced_decode_writes_d_before_the_kickoff_and_x_outside_the_blocks():
    n, blocks = 300, [(1, 10, 99), (2, 150, 279)]
    probs = _probs(n)
    mask = learned.pre_kickoff_mask(n, blocks, {1: 10, 2: 190})
    free = learned.decode_model(probs, blocks, 4.0, 4.0)
    forced = learned.decode_model(probs, blocks, 4.0, 4.0, forced_dead=mask)
    assert (free[10:100] == learned.H).all() and (free[150:280] == learned.H).all()
    assert (forced[150:190] == learned.D).all()
    assert (forced[190:280] == learned.H).all() and (forced[10:100] == learned.H).all()
    for codes in (free, forced):     # X outside the graded blocks, never X inside
        assert (codes[:10] == learned.X).all() and (codes[100:150] == learned.X).all()
        assert (codes[280:] == learned.X).all()
        assert not (codes[10:100] == learned.X).any() and not (codes[150:280] == learned.X).any()


def test_forced_none_and_an_empty_mask_change_nothing():
    rng = np.random.default_rng(0)
    p = rng.dirichlet((1.0, 1.0, 1.0), size=400)
    blocks = [(1, 0, 199), (2, 220, 399)]
    base = learned.decode_model(p, blocks, 2.0, 4.0)
    assert np.array_equal(base, learned.decode_model(p, blocks, 2.0, 4.0, forced_dead=None))
    empty = np.zeros(400, dtype=bool)
    assert np.array_equal(base, learned.decode_model(p, blocks, 2.0, 4.0, forced_dead=empty))


def test_forcing_is_a_constraint_on_the_viterbi_not_an_overwrite():
    """The path after the kickoff is decoded given D before it, as decode_blocks() does in c23.

    The network says A across the kickoff for a short while, then H. With a high H-A cost the
    free path keeps the short A run; forced D before the kickoff, the path enters H at the kickoff
    (one alive switch) instead of A then H (two switches).
    """
    n, blocks = 200, [(1, 0, 199)]
    p = _probs(n, h=0.8)
    p[40:105] = (0.1, 0.8, 0.1)               # A from 40 to 105, kickoff at 100
    mask = learned.pre_kickoff_mask(n, blocks, {1: 100})
    free = learned.decode_model(p, blocks, 20.0, 20.0)
    forced = learned.decode_model(p, blocks, 20.0, 20.0, forced_dead=mask)
    overwrite = free.copy()
    overwrite[mask] = learned.D
    assert (free[40:105] == learned.A).all() and (overwrite[100:105] == learned.A).all()
    assert (forced[:100] == learned.D).all() and (forced[100:] == learned.H).all()


# -- the switch -----------------------------------------------------------------------------------

def test_switch_is_on_by_default_and_loads_from_json(tmp_path):
    # On by default since branch r6e-adopt (LEARNED_STAGE_RESULT_R6.md Section 2).
    cfg = Config.default()
    assert cfg.LEARNED_STRUCTURE is True
    assert "LEARNED_STRUCTURE" in cfg.as_dict()
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"LEARNED_MODE": "model", "LEARNED_STRUCTURE": False}))
    loaded = config.load(str(path))
    assert loaded.LEARNED_STRUCTURE is False and loaded.config_hash() != cfg.config_hash()


def _patched_apply(monkeypatch, probs, kickoffs, structure, mode="model", base=None):
    monkeypatch.setattr(learned, "load_weights", lambda path: [])
    monkeypatch.setattr(learned, "frame_probabilities",
                        lambda nets, tac, game, cfg=None: (None, probs))
    seen = []

    def kicks(game=None):
        seen.append(game)
        return kickoffs

    monkeypatch.setattr(data, "kickoff_frames", kicks)
    # The network path only (LEARNED_PF is on by default since branch r6e-adopt), and the
    # kickoff team rule off (on by default since round 8; it reads the kickoff frames too:
    # tests/test_learned_kickoff_team.py).
    cfg = dataclasses.replace(Config.default(), LEARNED_MODE=mode, LEARNED_STRUCTURE=structure,
                              LEARNED_COST_HA=4.0, LEARNED_COST_AD=4.0, LEARNED_PF="off",
                              LEARNED_KICKOFF_TEAM=False)
    labels = learned.apply(cfg, "SYNTH", [(1, 0, 99), (2, 120, 239)], base, tactical={})
    return labels, seen


def test_apply_uses_the_kickoffs_of_games_py_only_with_the_switch(monkeypatch):
    probs = _probs(260)
    on, seen_on = _patched_apply(monkeypatch, probs, {1: 0, 2: 170}, True)
    off, seen_off = _patched_apply(monkeypatch, probs, {1: 0, 2: 170}, False)
    assert seen_on == ["SYNTH"] and seen_off == []
    assert (on[120:170] == "D").all() and (on[170:240] == "H").all()
    assert (off[120:240] == "H").all()
    assert np.array_equal(on[:120], off[:120]) and (on[100:120] == "X").all()


def test_hybrid_mode_ignores_the_switch(monkeypatch):
    probs = _probs(260)
    base = np.where(np.arange(260) < 150, "D", "A").astype("<U1")
    base[100:120] = "X"
    on, seen = _patched_apply(monkeypatch, probs, {1: 0, 2: 170}, True, "hybrid", base)
    off, _ = _patched_apply(monkeypatch, probs, {1: 0, 2: 170}, False, "hybrid", base)
    assert seen == [] and np.array_equal(on, off)


def test_the_step_reads_no_tagging_and_no_label():
    """Rule (b) is not in the step: c23 never makes a GOALS window dead (deadball.py, PLAN 1.6)."""
    for fn in (learned.pre_kickoff_mask, learned.decode_model, learned.apply):
        src = inspect.getsource(fn)
        assert "tagging" not in src and "GOALS" not in src and "evaluation" not in src
    src = inspect.getsource(learned.apply)
    assert "data.kickoff_frames(game)" in src

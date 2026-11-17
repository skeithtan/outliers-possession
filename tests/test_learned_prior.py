"""The rule-engine prior of mode "model" (Config.LEARNED_PRIOR_BONUS, engine.learned.apply_prior).

On every frame where the rule engine's own label (the labels LEARNED_MODE "off" writes) is H, A or
D, the bonus is added to the natural-log probability of that class before the Viterbi. X frames get
none. Bonus 0 leaves the decoder untouched. numpy only, synthetic inputs, no match data.
"""

import dataclasses
import json

import numpy as np
import pytest

from engine import config, data, learned, possession
from engine.config import Config

from tests import test_calibration as tc


def _probs(n, h=0.9):
    p = np.empty((n, 3))
    p[:, 0], p[:, 1], p[:, 2] = h, (1 - h) / 2, (1 - h) / 2
    return p


def _rule(n, blocks):
    """Rule codes with H, A and D runs inside the blocks and X outside."""
    codes = np.full(n, learned.X, dtype=np.int8)
    for _, first, last in blocks:
        span = np.arange(first, last + 1)
        codes[span] = np.array([learned.H, learned.A, learned.D])[((span - first) // 40) % 3]
    return codes


BLOCKS = [(1, 10, 209), (2, 240, 479)]


# -- apply_prior ----------------------------------------------------------------------------------

def test_bonus_zero_returns_the_same_array_untouched():
    rng = np.random.default_rng(1)
    logp = np.log(rng.dirichlet((1.0, 1.0, 1.0), size=500))
    keep = logp.copy()
    assert learned.apply_prior(logp, _rule(500, BLOCKS), 0.0) is logp
    assert learned.apply_prior(logp, None, 0) is logp          # no rule array is read at 0
    assert np.array_equal(logp, keep)


def test_bonus_adds_to_the_rule_class_in_natural_log_units_and_skips_x():
    n = 500
    rule = _rule(n, BLOCKS)
    logp = learned.log_emissions(_probs(n))
    out = learned.apply_prior(logp, rule, 1.5)
    assert out is not logp and np.array_equal(logp, learned.log_emissions(_probs(n)))
    diff = out - logp
    for t in range(n):
        want = np.zeros(3)
        if rule[t] != learned.X:
            want[rule[t]] = 1.5
        assert np.array_equal(diff[t], want), t
    x = rule == learned.X
    assert x.sum() == 10 + 30 + 20 and (diff[x] == 0).all()        # (d) X frames: no bonus
    # labels give the same result as codes
    assert np.array_equal(out, learned.apply_prior(logp, learned.codes_to_labels(rule), 1.5))
    # the bonus multiplies the probability by e**bonus
    assert np.allclose(np.exp(out[rule == learned.A, learned.A]),
                       0.05 * np.exp(1.5))


def test_apply_prior_refuses_bad_inputs():
    logp = np.zeros((10, 3))
    with pytest.raises(ValueError):
        learned.apply_prior(logp, np.zeros(9, dtype=np.int8), 1.0)
    with pytest.raises(ValueError):
        learned.apply_prior(logp, np.full(10, 4, dtype=np.int8), 1.0)
    with pytest.raises(ValueError):
        learned.apply_prior(logp, np.zeros(10, dtype=np.int8), -1.0)
    with pytest.raises(ValueError):
        learned.decode_model(np.full((10, 3), 1 / 3), [(1, 0, 9)], 1.0, 1.0, prior_bonus=1.0)


# -- decode_model ---------------------------------------------------------------------------------

def test_a_bonus_zero_decode_is_identical():                       # (a)
    rng = np.random.default_rng(2)
    p = rng.dirichlet((1.0, 1.0, 1.0), size=500)
    rule = _rule(500, BLOCKS)
    mask = learned.pre_kickoff_mask(500, BLOCKS, {1: 10, 2: 260})
    for forced in (None, mask):
        base = learned.decode_model(p, BLOCKS, 4.0, 2.0, forced)
        same = learned.decode_model(p, BLOCKS, 4.0, 2.0, forced, rule_codes=rule,
                                    prior_bonus=0.0)
        assert base.dtype == same.dtype and base.tobytes() == same.tobytes()


def test_flat_probabilities_and_a_positive_bonus_give_the_rule_labels():   # (b)
    n = 500
    rule = _rule(n, BLOCKS)
    flat = np.full((n, 3), 1.0 / 3.0)
    out = learned.decode_model(flat, BLOCKS, 4.0, 2.0, rule_codes=rule, prior_bonus=1.0)
    assert np.array_equal(out, rule)
    # a flat network alone gives no information: the plain decode is all H (lowest code)
    plain = learned.decode_model(flat, BLOCKS, 4.0, 2.0)
    assert not np.array_equal(plain, rule)


def test_confident_probabilities_and_a_small_bonus_give_the_plain_decode():   # (c)
    n = 500
    rule = _rule(n, BLOCKS)
    # the network is confident (0.98) about a class sequence that disagrees with the rule
    net = np.where((np.arange(n) // 55) % 2 == 0, learned.A, learned.H)
    p = np.full((n, 3), 0.01)
    p[np.arange(n), net] = 0.98
    plain = learned.decode_model(p, BLOCKS, 4.0, 2.0)
    out = learned.decode_model(p, BLOCKS, 4.0, 2.0, rule_codes=rule, prior_bonus=0.5)
    assert np.array_equal(out, plain)
    assert not np.array_equal(out, rule)
    # a bonus above log(0.98 / 0.01) ~ 4.6 lets the rule win frame by frame (no switch cost)
    big = learned.decode_model(p, BLOCKS, 0.0, 0.0, rule_codes=rule, prior_bonus=5.0)
    assert np.array_equal(big, rule)


def test_x_frames_of_the_rule_get_no_bonus_in_the_decode():                   # (d)
    n, blocks = 200, [(1, 0, 199)]
    p = _probs(n, h=0.6)                                    # the network says H, mildly
    rule = np.full(n, learned.A, dtype=np.int8)
    rule[50:150] = learned.X                                # X inside the block: no prior
    out = learned.decode_model(p, blocks, 0.0, 0.0, rule_codes=rule, prior_bonus=3.0)
    assert (out[:50] == learned.A).all() and (out[150:] == learned.A).all()
    assert (out[50:150] == learned.H).all()


def test_forced_kickoff_frames_stay_dead_whatever_the_rule_says():             # (e)
    n, blocks = 300, [(1, 10, 99), (2, 150, 279)]
    mask = learned.pre_kickoff_mask(n, blocks, {1: 10, 2: 200})
    rule = np.full(n, learned.H, dtype=np.int8)             # rule says H, even before kickoff
    rule[:10] = learned.X
    rule[100:150] = learned.X
    rule[280:] = learned.X
    for bonus in (0.0, 1.0, 50.0):
        out = learned.decode_model(_probs(n), blocks, 4.0, 4.0, mask, rule_codes=rule,
                                   prior_bonus=bonus)
        assert (out[150:200] == learned.D).all()
        assert (out[200:280] == learned.H).all() and (out[10:100] == learned.H).all()
        assert (out[:10] == learned.X).all() and (out[100:150] == learned.X).all()
        assert (out[280:] == learned.X).all()


def test_engine_decode_equals_the_tool_recipe():
    """A tool decoding stored probabilities with the stored rule labels gets the engine's codes."""
    n = 500
    rng = np.random.default_rng(4)
    p = rng.dirichlet((2.0, 2.0, 1.0), size=n)
    rule = learned.codes_to_labels(_rule(n, BLOCKS))
    mask = learned.pre_kickoff_mask(n, BLOCKS, {1: 10, 2: 260})
    engine = learned.decode_model(p, BLOCKS, 8.0, 2.0, mask,
                                  rule_codes=learned.labels_to_codes(rule), prior_bonus=1.0)
    # by hand: emissions, prior, forcing, Viterbi per block
    logp = learned.apply_prior(learned.log_emissions(p), rule, 1.0)
    logp[mask, :2] = -np.inf
    manual = np.full(n, learned.X, dtype=np.int8)
    for _, first, last in BLOCKS:
        manual[first:last + 1] = learned.viterbi(logp[first:last + 1], 8.0, 2.0)
    assert np.array_equal(engine, manual)


# -- the setting, apply() and run() ---------------------------------------------------------------

def test_setting_is_zero_by_default_and_loads_from_json(tmp_path):
    cfg = Config.default()
    assert cfg.LEARNED_PRIOR_BONUS == 0.0 and isinstance(cfg.LEARNED_PRIOR_BONUS, float)
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"LEARNED_MODE": "model", "LEARNED_PRIOR_BONUS": 1}))
    loaded = config.load(str(path))
    assert loaded.LEARNED_PRIOR_BONUS == 1.0 and isinstance(loaded.LEARNED_PRIOR_BONUS, float)
    with pytest.raises(ValueError):
        learned.prior_bonus(dataclasses.replace(cfg, LEARNED_PRIOR_BONUS=-0.5))


def _apply(monkeypatch, probs, bonus, base, mode="model"):
    monkeypatch.setattr(learned, "load_weights", lambda path: [])
    monkeypatch.setattr(learned, "frame_probabilities",
                        lambda nets, tac, game, cfg=None: (None, probs))
    monkeypatch.setattr(data, "kickoff_frames", lambda game=None: {1: 10, 2: 240})
    # The network path only (LEARNED_PF is on by default since branch r6e-adopt), and the
    # kickoff team rule off (on by default since round 8; tests/test_learned_kickoff_team.py).
    cfg = dataclasses.replace(Config.default(), LEARNED_MODE=mode, LEARNED_PRIOR_BONUS=bonus,
                              LEARNED_COST_HA=4.0, LEARNED_COST_AD=2.0, LEARNED_PF="off",
                              LEARNED_KICKOFF_TEAM=False)
    return learned.apply(cfg, "SYNTH", BLOCKS, base, tactical={})


def test_apply_reads_the_base_labels_only_with_a_bonus(monkeypatch):
    n = 500
    flat = np.full((n, 3), 1.0 / 3.0)
    rule = learned.codes_to_labels(_rule(n, BLOCKS))
    off = _apply(monkeypatch, flat, 0.0, None)              # no labels needed at 0
    assert np.array_equal(off, _apply(monkeypatch, flat, 0.0, rule))
    on = _apply(monkeypatch, flat, 1.0, rule)
    assert np.array_equal(on, rule)
    with pytest.raises(ValueError):
        _apply(monkeypatch, flat, 1.0, None)
    # hybrid ignores the bonus
    hyb = _apply(monkeypatch, _probs(n), 0.0, rule, "hybrid")
    assert np.array_equal(hyb, _apply(monkeypatch, _probs(n), 2.0, rule, "hybrid"))


def _synthetic_inputs(cfg=tc.CFG):
    """test_calibration's synthetic prepare() result, keyed for `cfg`. The LEARNED_* fields are
    part of possession.detection_key(), so each config gets its own key."""
    inputs = dict(tc._inputs(tc.BASE_EVENTS), game="SYNTH")
    inputs["key"] = possession.detection_key(cfg)
    return inputs


def test_run_passes_the_final_off_labels_to_the_decoder(monkeypatch):
    """The rule labels are those LEARNED_MODE "off" returns on the same inputs (after step 8)."""
    off_cfg = dataclasses.replace(tc.CFG, LEARNED_MODE="off")
    off = possession.run(off_cfg, inputs=_synthetic_inputs(off_cfg))
    seen = {}

    def fake_apply(cfg, game, blocks, base_labels=None, return_probs=False, tactical=None):
        seen["base"] = None if base_labels is None else np.array(base_labels)
        seen["blocks"] = blocks
        return np.asarray(base_labels), np.zeros((len(base_labels), 3))

    monkeypatch.setattr(learned, "apply", fake_apply)
    monkeypatch.setattr(possession, "engine_tactical", lambda cfg, game: {})
    cfg = dataclasses.replace(tc.CFG, LEARNED_MODE="model", LEARNED_PRIOR_BONUS=1.0)
    inputs = _synthetic_inputs(cfg)
    labels, parts = possession.run(cfg, inputs=inputs, return_parts=True)
    assert np.array_equal(seen["base"], off) and np.array_equal(parts["c22_labels"], off)
    assert seen["blocks"] == inputs["blocks"] and len(off) == inputs["n_frames"]
    assert set(off.tolist()) >= {"H", "A", "D"}


def test_run_takes_the_full_path_in_model_mode_only_with_a_bonus(monkeypatch):
    base = dataclasses.replace(tc.CFG, LEARNED_MODE="model")
    with_bonus = dataclasses.replace(base, LEARNED_PRIOR_BONUS=2.0)
    inputs = _synthetic_inputs(with_bonus)
    calls = {"prepare": 0, "match_inputs": 0, "base": []}

    def fake_prepare(cfg, game=None):
        calls["prepare"] += 1
        return inputs

    def fake_match_inputs(game=None):
        calls["match_inputs"] += 1
        return "SYNTH", None, inputs["blocks"], None, None

    def fake_apply(cfg, game, blocks, base_labels=None, return_probs=False, tactical=None):
        calls["base"].append(base_labels)
        labels = np.full(inputs["n_frames"], "D", dtype="<U1")
        return (labels, None) if return_probs else labels

    monkeypatch.setattr(possession, "prepare", fake_prepare)
    monkeypatch.setattr(possession, "match_inputs", fake_match_inputs)
    monkeypatch.setattr(possession, "engine_tactical", lambda cfg, game: {})
    monkeypatch.setattr(learned, "apply", fake_apply)
    possession.run(base)                                   # bonus 0: the early return
    assert calls == {"prepare": 0, "match_inputs": 1, "base": [None]}
    possession.run(with_bonus)
    assert calls["prepare"] == 1 and calls["match_inputs"] == 1
    off_cfg = dataclasses.replace(tc.CFG, LEARNED_MODE="off")
    off = possession.run(off_cfg, inputs=_synthetic_inputs(off_cfg))
    assert np.array_equal(calls["base"][1], off)

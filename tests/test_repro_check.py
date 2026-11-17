"""Tests of tools/learned_spike/repro_check.py (LEARNED_STAGE_PROTOCOL.md Section 6, criterion 8).

The first tests need numpy only. The last test needs torch. It skips in the engine venv. Run it
with the torch venv: it checks the torch path of the tool against engine.learned on random
weights and a random input.
"""

import numpy as np
import pytest

from engine import learned
from engine import learned_features as lf
from tools.learned_spike import repro_check as rc


def test_battery_line():
    assert rc.battery_ok("11-09 16:24:33 28% charging 'AC Power' paused=0")
    assert rc.battery_ok("11-09 16:24:33 26% charging 'AC Power' paused=0")
    assert not rc.battery_ok("11-09 16:24:33 25% charging 'AC Power' paused=0")
    assert not rc.battery_ok("11-09 16:24:33 80% discharging 'Battery Power' paused=0")
    assert not rc.battery_ok("11-09 16:24:33 80% charging 'AC Power' paused=1")
    assert not rc.battery_ok("")


def test_compare_codes_lists_every_difference():
    a = np.array([0, 0, 1, 1, 2, 3, 3], dtype=np.int8)
    b = np.array([0, 1, 1, 0, 2, 3, 0], dtype=np.int8)
    graded = np.array([True] * 5 + [False] * 2)
    rec = rc.compare_codes(a, b, graded)
    assert rec["frames"] == 3 and rec["graded_frames"] == 2
    assert rec["listed"] == [[1, 0, 1], [3, 1, 0], [6, 3, 0]]
    assert rec["runs"] == [[1, 2], [3, 4], [6, 7]]


def _record(n_graded, m_diff, h_diff, e2e=True):
    def cand(n):
        return {"torch_vs_numpy": {"graded_frames": n, "frames": n, "listed": [[0, 0, 1]] * n},
                "end_to_end_equal": e2e}
    return {"n_graded": n_graded, "M": cand(m_diff), "H": cand(h_diff)}


def test_verdict_follows_the_protocol_rule():
    assert rc.verdict({"a": _record(100000, 0, 0), "b": _record(100000, 0, 0)})["pass"]
    # 19 of 200,000 frames is below 0.0001: a pass when every frame is logged.
    out = rc.verdict({"a": _record(100000, 19, 0), "b": _record(100000, 0, 0)})
    assert out["pass"] and out["M"]["disagree_graded_frames"] == 19
    # 20 of 200,000 is not below 0.0001.
    assert not rc.verdict({"a": _record(100000, 20, 0), "b": _record(100000, 0, 0)})["pass"]
    # The end-to-end engine must equal the numpy path on every match.
    out = rc.verdict({"a": _record(100000, 0, 0), "b": _record(100000, 0, 0, e2e=False)})
    assert not out["pass"] and not out["M"]["pass"] and not out["H"]["pass"]


def test_torch_path_equals_the_numpy_network(tmp_path):
    torch = pytest.importorskip("torch")
    from tools.learned_spike import export, model
    torch.manual_seed(0)
    states = []
    for scale in (1.0, 2.0):
        net = model.Net(hidden=16)
        states.append({k: v * scale for k, v in net.state_dict().items()})
    path = str(tmp_path / "random.npz")
    export.write(path, states)
    nets = learned.load_weights(path)
    assert rc.weights_equal(states, nets)
    rng = np.random.RandomState(7)
    feats = rng.normal(0.0, 1.0, (80, lf.NF)).astype(np.float32)
    p_np = learned.predict_ensemble(nets, feats)
    p64 = rc.torch_ensemble(states, feats, "float64")
    p32 = rc.torch_ensemble(states, feats, "float32")
    assert np.abs(p_np - p64).max() < 1e-9
    assert np.abs(p_np - p32).max() < 1e-5          # the protocol bound of Section 6
    blocks = [(1, 0, 399)]
    frames_np = learned.upsample(np.round(p_np, learned.PROB_DECIMALS), 400)
    frames_t = learned.upsample(np.round(p64, learned.PROB_DECIMALS), 400)
    assert np.array_equal(learned.decode_model(frames_np, blocks, 4.0, 4.0),
                          learned.decode_model(frames_t, blocks, 4.0, 4.0))
    states[1]["head.bias"] = states[1]["head.bias"] + 1.0
    assert not rc.weights_equal(states, nets)


def test_guarded_tactical_equals_engine_tactical(monkeypatch):
    """--tactical guarded reads the ATD of possession.engine_tactical() (R2 Section 9, D1)."""
    from engine import data, possession
    from engine.config import Config
    from tests.test_learned import _keeper_swap_tactical
    raw = _keeper_swap_tactical()
    blocks = [(1, 0, 2999)]
    monkeypatch.setattr(data, "load_tactical", lambda game=None, *a, **k: raw)
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: blocks)
    monkeypatch.setattr(data, "pitch_m", lambda game=None: (105.0, 68.0))
    cfg = Config.default()
    game = "METRICA-S3"
    guarded, rows = rc.guarded_tactical(raw, game, blocks, cfg)
    want = possession.engine_tactical(cfg, game)
    assert rows > 900
    for key in ("px", "py", "ps", "bx", "by", "team_of_slot"):
        assert np.array_equal(guarded[key], want[key], equal_nan=True)
    assert (lf.matrix_md5(lf.frame_features(guarded, game, cfg=cfg))
            != lf.matrix_md5(lf.frame_features(raw, game, cfg=cfg)))

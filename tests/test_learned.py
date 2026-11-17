"""Tests of engine/learned.py and engine/learned_features.py (LEARNED_STAGE_PROTOCOL.md 1, 2, 6).

The tests need numpy only. The fixtures under tests/fixtures/ came once from torch 2.8.0 with
two random model.Net instances (64 replaced by 16 hidden units, the second with every weight
doubled) and a random input of 60 bins. learned_net_outputs.npz holds the torch outputs: each
layer in float64, the symmetric probabilities in float64 and on the float32 path of the spike.
"""

import ast
import dataclasses
import json
import os
import types

import numpy as np
import pytest

import games
from engine import data, learned, submission
from engine import learned_features as lf
from engine.config import Config
from tools.analysis import switch_off_identity as soi
from tools.learned_spike import export
from tools.learned_spike import features as spike_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
WEIGHTS = os.path.join(FIXTURES, "learned_net_weights.npz")
C22_MD5 = "6b7b93c4cf412a45f02b81e70c6ea60a"
# Round 3: the learned stage is merged into the c23 engine (tag `submission`), so LEARNED_MODE off
# writes c23 (LEARNED_STAGE_PROTOCOL_R3.md Section 0.3).
C23_MD5 = "7844cad862e38d55cf9a3fc29b18ae85"
# The learned default on branch r6e-adopt (14 Nov 2026): the round 6 candidate E2, the network
# engine/models/learned_v7.npz averaged with the per-frame member engine/models/learned_pf_v7.npz
# at weight 0.25, costs (12, 12), prior bonus 0, LEARNED_STRUCTURE on. M_MD5 is the md5 of its
# output on BAR-ATH and the weights md5s are those of docs/plan/calibration/
# LEARNED_STAGE_RESULT_R6.md (Sections 2 and 10). Since round 8 the default is E3 = E2 plus the
# kickoff team rule (Config.LEARNED_KICKOFF_TEAM, LEARNED_STAGE_RESULT_R8.md): M_MD5 is E3's output,
# which differs from E2's on frames 113741..113761 only; E2_MD5 (the rule off) stays reachable.
M_MD5 = "d7f167a2401387bd8339d1bcbb292f2c"
E2_MD5 = "7dafcd82b0db8557fa9caf8ede57f4c8"
NPZ_MD5 = "31e88976ea8da0169b15f07ff44ee17c"
PF_NPZ_MD5 = "0b5e81aa00c0486057aa7eb790c98109"
OFF = {"LEARNED_MODE": "off"}
_ASSETS = games.asset_paths()
HAVE_DATA = all(os.path.isfile(_ASSETS[k])
                for k in ("atd_meta", "atd_raw", "homography", "smart_tagging"))


@pytest.fixture(scope="module")
def nets():
    return learned.load_weights(WEIGHTS)


@pytest.fixture(scope="module")
def torch_out():
    with np.load(os.path.join(FIXTURES, "learned_net_outputs.npz")) as archive:
        return {key: archive[key] for key in archive.files}


# -- (a) the numpy network against torch ---------------------------------------------------------

def test_each_layer_matches_torch_float64(nets, torch_out):
    copies = learned.symmetric_copies(torch_out["x"])
    for k, net in enumerate(nets):
        h = learned.gelu(learned.linear(copies, net["inp.0.weight"], net["inp.0.bias"]))
        assert np.abs(h - torch_out["m%d/inp" % k]).max() < 1e-9
        h = h + learned.gelu(learned.conv1d(h, net["conv.weight"], net["conv.bias"]))
        assert np.abs(h - torch_out["m%d/conv" % k]).max() < 1e-9
        h0 = learned.gru_bidirectional(h, net, 0)
        assert np.abs(h0 - torch_out["m%d/gru0" % k]).max() < 1e-9
        h1 = learned.gru_bidirectional(h0, net, 1)
        assert np.abs(h1 - torch_out["m%d/gru1" % k]).max() < 1e-9
        logits = learned.forward(net, copies)
        assert np.abs(logits - torch_out["m%d/logits" % k]).max() < 1e-9


def test_probabilities_match_torch(nets, torch_out):
    x = torch_out["x"]
    for k, net in enumerate(nets):
        p = learned.predict_symmetric(net, x)
        assert np.abs(p - torch_out["m%d/sym64" % k]).max() < 1e-9
        # The spike predicts in float32. The protocol bound on probabilities is 1e-5.
        assert np.abs(p - torch_out["m%d/sym32" % k]).max() < 1e-5
    ens = learned.predict_ensemble(nets, x)
    assert np.abs(ens - torch_out["ens64"]).max() < 1e-9
    assert np.abs(ens - torch_out["ens32"]).max() < 1e-5
    assert np.allclose(ens.sum(axis=1), 1.0)


def test_gelu_uses_exact_erf():
    x = np.array([-6.0, -1.0, -0.1, 0.0, 0.3, 2.0, 9.0])
    want = [v * 0.5 * (1.0 + __import__("math").erf(v / np.sqrt(2.0))) for v in x]
    assert np.abs(learned.gelu(x) - want).max() < 1e-15
    assert learned.gelu(np.zeros((2, 3, 4))).shape == (2, 3, 4)


def test_bin_probabilities_round_once_to_four_decimals(nets, torch_out):
    p5 = learned.bin_probabilities(nets, torch_out["x"])
    assert learned.PROB_DECIMALS == 4
    assert np.array_equal(p5, np.round(learned.predict_ensemble(nets, torch_out["x"]), 4))
    frames = learned.upsample(p5, 300)
    assert frames.shape == (300, 3)
    assert np.array_equal(frames[2::5][:60], p5)   # bin centres keep the rounded values


# -- (b) symmetry ---------------------------------------------------------------------------------

def test_swapped_input_swaps_h_and_a(nets, torch_out):
    x = torch_out["x"]
    p = learned.predict_ensemble(nets, x)
    swapped = learned.predict_ensemble(nets, lf.transform(x, swap=True))
    mirrored = learned.predict_ensemble(nets, lf.transform(x, mirror=True))
    assert np.abs(swapped - p[:, [1, 0, 2]]).max() < 1e-12
    assert np.abs(mirrored - p).max() < 1e-12


# -- (c) decoders against the spike ---------------------------------------------------------------

def _spike_functions():
    """viterbi, decode, upsample of model.py and team_viterbi of hybrid.py, from their source.

    model.py imports torch, so the test executes only the numpy functions of the live files.
    """
    found = {}
    spike = os.path.join(ROOT, "tools", "learned_spike")
    namespace = {"np": np, "features": types.SimpleNamespace(BIN=lf.BIN)}
    for name, wanted in (("model.py", ("viterbi", "decode", "upsample")),
                         ("hybrid.py", ("runs_of", "team_viterbi"))):
        with open(os.path.join(spike, name)) as handle:
            tree = ast.parse(handle.read())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), namespace)
        if name == "model.py":
            namespace["model"] = types.SimpleNamespace(viterbi=namespace["viterbi"])
        found.update({n: namespace[n] for n in wanted})
    return found


def _random_probs(rng, n):
    raw = rng.gamma(0.3, size=(n, 3))
    return raw / raw.sum(axis=1, keepdims=True)


def test_model_decoder_matches_the_spike():
    spike = _spike_functions()
    rng = np.random.RandomState(3)
    p5 = np.round(_random_probs(rng, 400), 4)
    probs = learned.upsample(p5, 2000)
    assert np.array_equal(probs, spike["upsample"](p5, 2000))
    blocks = [(1, 10, 900), (2, 1000, 1990)]
    for cost_ha, cost_ad in ((32.0, 24.0), (1.0, 2.0), (8.0, 64.0)):
        mine = learned.decode_model(probs, blocks, cost_ha, cost_ad)
        assert np.array_equal(mine, spike["decode"](probs, blocks, cost_ha, cost_ad))
        assert (mine[:10] == learned.X).all() and (mine[901:1000] == learned.X).all()
        assert set(np.unique(mine[10:901])) <= {0, 1, 2}


def test_hybrid_decoder_matches_the_spike():
    spike = _spike_functions()
    rng = np.random.RandomState(4)
    probs = learned.upsample(np.round(_random_probs(rng, 400), 4), 2000)
    base = np.full(2000, learned.X, dtype=np.int8)
    base[10:1990] = rng.choice([0, 1, 2], size=1980, p=[0.4, 0.3, 0.3])
    base[10:1990] = np.repeat(base[10:1990:30], 30)[:1980]
    for cost in (32.0, 4.0):
        mine = learned.decode_hybrid(probs, base, cost)
        # hybrid.py hyb_c22dead, written out from its main().
        team = probs[:, :2] / np.maximum(probs[:, :2].sum(axis=1, keepdims=True), 1e-9)
        logt = np.log(np.clip(team, 1e-4, 1.0))
        want = base.copy()
        for a, b in spike["runs_of"](np.isin(base, (0, 1))):
            want[a:b] = spike["team_viterbi"](logt[a:b], cost)
        assert np.array_equal(mine, want)
        dead = ~np.isin(base, (0, 1))
        assert np.array_equal(mine[dead], base[dead])
        assert set(np.unique(mine[~dead])) <= {0, 1}


def test_label_codes_round_trip():
    labels = np.array(list("HADXXH"))
    codes = learned.labels_to_codes(labels)
    assert list(codes) == [0, 1, 2, 3, 3, 0]
    assert list(learned.codes_to_labels(codes)) == list(labels)
    with pytest.raises(ValueError):
        learned.labels_to_codes(np.array(["H", "Q"]))


# -- features -------------------------------------------------------------------------------------

def _synthetic_tactical(n=2000, seed=5):
    rng = np.random.RandomState(seed)
    px = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, (n, 22)), axis=0)
                 + np.where(np.arange(22) < 11, -0.15, 0.15), 0.01, 0.99).astype(np.float32)
    py = np.clip(np.linspace(0.05, 0.95, 22) + np.cumsum(rng.normal(0, 0.002, (n, 22)), axis=0),
                 0.01, 0.99).astype(np.float32)
    px[300:420, 3] = np.nan
    py[300:420, 3] = np.nan
    bx = np.clip(0.5 + np.cumsum(rng.normal(0, 0.003, n)), 0.0, 1.0).astype(np.float32)
    by = np.clip(0.5 + np.cumsum(rng.normal(0, 0.003, n)), 0.0, 1.0).astype(np.float32)
    bx[700:760] = np.nan
    by[700:760] = np.nan
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    return {"px": px, "py": py, "bx": bx, "by": by, "team_of_slot": team, "n_frames": n}


@pytest.fixture
def small_match(monkeypatch):
    """A synthetic match: two blocks of 1,000 frames on a 105 x 68 m pitch."""
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, 989), (2, 1010, 1999)])
    monkeypatch.setattr(data, "pitch_m", lambda game=None: (105.0, 68.0))
    return _synthetic_tactical()


def test_engine_features_equal_the_spike_features_byte_for_byte(small_match):
    eng = lf.downsample(lf.frame_features(small_match, "METRICA-S3"))
    spike = spike_features.downsample(spike_features.frame_features(small_match, "METRICA-S3"))
    assert eng.dtype == np.float32 and eng.shape == (400, lf.NF)
    assert eng.tobytes() == spike.tobytes()
    assert lf.matrix_md5(eng) == lf.matrix_md5(spike)
    assert lf.matrix_md5(eng) != lf.matrix_md5(eng[:-1])
    assert lf.SPEC == spike_features.SPEC and lf.NF == spike_features.NF == 50


def test_apply_model_and_hybrid_on_a_synthetic_match(small_match, monkeypatch):
    monkeypatch.setattr(data, "load_tactical", lambda game=None: small_match)
    blocks = data.graded_blocks()
    cfg = dataclasses.replace(Config.default(), LEARNED_MODE="model",
                              LEARNED_WEIGHTS=os.path.relpath(WEIGHTS, ROOT))
    labels, probs = learned.apply(cfg, "METRICA-S3", blocks, return_probs=True)
    assert labels.shape == (2000,) and probs.shape == (2000, 3)
    assert set(labels[990:1010]) == {"X"} and "X" not in set(labels[:990])
    base = np.where(np.arange(2000) % 200 < 50, "D", "H").astype("<U1")
    base[990:1010] = "X"
    hybrid = learned.apply(dataclasses.replace(cfg, LEARNED_MODE="hybrid"), "METRICA-S3",
                           blocks, base)
    assert np.array_equal(hybrid == "D", base == "D")
    assert np.array_equal(hybrid == "X", base == "X")
    with pytest.raises(ValueError):
        learned.apply(dataclasses.replace(cfg, LEARNED_MODE="hybrid"), "METRICA-S3", blocks)


# -- D1: the learned stage reads the ATD with the C9 goalkeeper guard (R2 Section 9) -------------

def _keeper_swap_tactical(n=3000, swap_from=2000, seed=7):
    """A synthetic match where tracks.guard_goalkeeper_swaps() fires.

    The home goalkeeper (slot 0) stays in its own box, then from `swap_from` its slot runs next
    to the away goalkeeper (slot 11), whose slot freezes there with no ATD speed.
    """
    rng = np.random.RandomState(seed)
    base_x = np.where(np.arange(22) < 11, 0.35, 0.65)
    px = np.clip(base_x + np.cumsum(rng.normal(0, 0.002, (n, 22)), axis=0), 0.2, 0.8)
    py = np.clip(np.linspace(0.1, 0.9, 22) + np.cumsum(rng.normal(0, 0.002, (n, 22)), axis=0),
                 0.05, 0.95)
    t = np.arange(n)
    px[:, 0], py[:, 0] = 0.03 + 0.01 * np.sin(t / 50.0), 0.5 + 0.02 * np.cos(t / 40.0)
    px[:, 11], py[:, 11] = 0.97 - 0.01 * np.sin(t / 60.0), 0.5 + 0.02 * np.sin(t / 45.0)
    px[swap_from:, 0] = 0.92 + 0.01 * np.sin(t[swap_from:] / 30.0)
    px[swap_from:, 11], py[swap_from:, 11] = 0.97, 0.52
    ps = np.abs(rng.normal(2.0, 0.5, (n, 22)))
    ps[swap_from:, 11] = np.nan
    bx = np.clip(0.5 + np.cumsum(rng.normal(0, 0.003, n)), 0.0, 1.0)
    by = np.clip(0.5 + np.cumsum(rng.normal(0, 0.003, n)), 0.0, 1.0)
    f = np.float32
    return {"px": px.astype(f), "py": py.astype(f), "ps": ps.astype(f), "bx": bx.astype(f),
            "by": by.astype(f), "team_of_slot": np.array([0] * 11 + [1] * 11, dtype=np.int8),
            "n_frames": n}


def test_learned_features_read_the_guarded_atd(monkeypatch):
    from engine import possession, tracks
    raw = _keeper_swap_tactical()
    monkeypatch.setattr(data, "load_tactical", lambda game=None, *a, **k: raw)
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, 2999)])
    monkeypatch.setattr(data, "pitch_m", lambda game=None: (105.0, 68.0))
    game = "METRICA-S3"
    # The network path only: the per-frame member (LEARNED_PF, on by default since branch
    # r6e-adopt) has its own tests in tests/test_learned_pf.py.
    cfg = dataclasses.replace(Config.default(), LEARNED_MODE="model", LEARNED_PF="off",
                              LEARNED_WEIGHTS=os.path.relpath(WEIGHTS, ROOT))
    _, swaps, _ = tracks.guard_goalkeeper_swaps(raw, data.graded_blocks(), cfg, (105.0, 68.0))
    assert swaps.shape[0] > 900
    guarded = possession.engine_tactical(cfg, game)
    md5 = {name: lf.matrix_md5(lf.frame_features(tac, game, cfg=cfg))
           for name, tac in (("raw", raw), ("guarded", guarded))}
    assert md5["raw"] != md5["guarded"]

    seen = []
    original = lf.frame_features

    def spy(tac, game, ball_override=None, cfg=None):
        out = original(tac, game, ball_override=ball_override, cfg=cfg)
        seen.append(lf.matrix_md5(out))
        return out

    monkeypatch.setattr(lf, "frame_features", spy)
    # The model mode of possession.run() (the early return).
    labels = possession.run(cfg, game=game)
    assert labels.shape == (3000,) and seen == [md5["guarded"]]
    # apply() reads the tactical it is given, in both modes. None keeps data.load_tactical().
    nets = learned.load_weights(WEIGHTS)
    want = learned.frame_probabilities(nets, guarded, game, cfg)[1]
    base = np.where(np.arange(3000) % 300 < 60, "D", "H").astype("<U1")
    hybrid_cfg = dataclasses.replace(cfg, LEARNED_MODE="hybrid")
    for mode_cfg, base_labels in ((cfg, None), (hybrid_cfg, base)):
        seen.clear()
        _, probs = learned.apply(mode_cfg, game, data.graded_blocks(), base_labels,
                                 return_probs=True, tactical=guarded)
        assert seen == [md5["guarded"]] and np.array_equal(probs, want)
        seen.clear()
        learned.apply(mode_cfg, game, data.graded_blocks(), base_labels)
        assert seen == [md5["raw"]]


def test_bar_report_feature_band_reads_the_guarded_atd(monkeypatch):
    """The 8c features of bar_report.py come from engine_tactical() (R2 Section 9, D1)."""
    from engine import possession
    from tools.learned_spike import bar_report as br
    raw = _keeper_swap_tactical()
    monkeypatch.setattr(data, "load_tactical", lambda game=None, *a, **k: raw)
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, 2999)])
    monkeypatch.setattr(data, "pitch_m", lambda game=None: (105.0, 68.0))
    game = "METRICA-S3"
    guarded = possession.engine_tactical(None, game)
    want = spike_features.downsample(spike_features.frame_features(guarded, game))
    unguarded = spike_features.downsample(spike_features.frame_features(raw, game))
    got = br.bar_features(game)
    assert got.dtype == np.float64 and np.array_equal(got, want.astype(np.float64))
    assert not np.array_equal(got, unguarded.astype(np.float64))
    source = open(os.path.join(ROOT, "tools", "learned_spike", "bar_report.py")).read()
    body = source[source.index("def build_report"):source.index("def print_report")]
    assert "bar_feats = bar_features(game, load, build.folder_version(args.feature_cache))" in body and "frame_features" not in body


def test_run_and_the_bar_tools_pass_the_guarded_atd():
    """possession.run() passes engine_tactical() on the model and hybrid paths. bar_probs.py and
    the BAR-ATH loader of evaluate.py read engine_tactical() too (R2 Section 9, D1)."""
    from engine import possession
    tree = ast.parse(open(os.path.join(ROOT, "engine", "possession.py")).read())
    run = next(node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "run")
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "apply"
             and isinstance(node.func.value, ast.Name) and node.func.value.id == "learned"]
    assert len(calls) == 2
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw.get("tactical"), ast.Call)
        assert kw["tactical"].func.id == "engine_tactical"
    assert possession.engine_tactical.__module__ == "engine.possession"
    spike = os.path.join(ROOT, "tools", "learned_spike")
    bar_probs = open(os.path.join(spike, "bar_probs.py")).read()
    assert "engine_tactical(None, game)" in bar_probs and "load_tactical" not in bar_probs
    evaluate = open(os.path.join(spike, "evaluate.py")).read()
    loader = evaluate[evaluate.index("def ext_loader"):evaluate.index("def cmd_bar")]
    assert "if game == COMPETITION:\n            return engine_tactical(None, game)" in loader


# -- weights and config ---------------------------------------------------------------------------

def test_weights_file_layout(nets):
    assert len(nets) == 2
    assert set(nets[0]) == set(learned.LAYER_NAMES) and len(learned.LAYER_NAMES) == 22
    assert all(v.dtype == np.float64 for v in nets[0].values())


def test_load_weights_refuses_a_bad_file(tmp_path, nets):
    state = dict(nets[0])
    good = str(tmp_path / "good.npz")
    export.write(good, [state, state])
    assert len(learned.load_weights(good)) == 2
    arrays = export.arrays_of([state])
    bad = dict(arrays, feature_version=np.str_("other"))
    np.savez(str(tmp_path / "version.npz"), **bad)
    with pytest.raises(ValueError, match="feature_version"):
        learned.load_weights(str(tmp_path / "version.npz"))
    missing = {k: v for k, v in arrays.items() if k != "m0/head.bias"}
    np.savez(str(tmp_path / "missing.npz"), **missing)
    with pytest.raises(ValueError, match="missing"):
        learned.load_weights(str(tmp_path / "missing.npz"))
    wrong = dict(arrays)
    wrong["m0/conv.weight"] = wrong["m0/conv.weight"][:, :, :3]
    np.savez(str(tmp_path / "shape.npz"), **wrong)
    with pytest.raises(ValueError, match="shape"):
        learned.load_weights(str(tmp_path / "shape.npz"))
    with pytest.raises(ValueError):
        export.arrays_of([{k: v for k, v in state.items() if k != "conv.bias"}])


def test_config_fields_and_modes(tmp_path):
    cfg = Config.default()
    assert cfg.LEARNED_MODE == "model"
    assert cfg.LEARNED_WEIGHTS == "engine/models/learned_v7.npz"
    assert (cfg.LEARNED_COST_HA, cfg.LEARNED_COST_AD) == (12.0, 12.0)
    assert cfg.LEARNED_PRIOR_BONUS == 0.0
    assert cfg.LEARNED_STRUCTURE is True
    assert (cfg.LEARNED_PF, cfg.LEARNED_PF_WEIGHT) == ("on", 0.25)
    assert cfg.LEARNED_PF_WEIGHTS == "engine/models/learned_pf_v7.npz"
    assert learned.weights_path(cfg) == os.path.join(ROOT, "engine", "models", "learned_v7.npz")
    assert learned.check_mode(cfg) == "model"
    assert learned.check_mode(dataclasses.replace(cfg, **OFF)) == "off"
    path = tmp_path / "fold.json"
    path.write_text(json.dumps({"LEARNED_MODE": "hybrid", "LEARNED_WEIGHTS": "/abs/fold3.npz",
                                "LEARNED_COST_HA_HYBRID": 16}))
    from engine import config
    loaded = config.load(str(path))
    assert loaded.LEARNED_COST_HA_HYBRID == 16.0 and isinstance(loaded.LEARNED_COST_HA_HYBRID,
                                                                 float)
    assert learned.weights_path(loaded) == "/abs/fold3.npz"
    assert learned.check_mode(loaded) == "hybrid"
    with pytest.raises(ValueError):
        learned.check_mode(dataclasses.replace(cfg, LEARNED_MODE="on"))
    with pytest.raises(ValueError):                     # apply() refuses LEARNED_MODE off
        learned.apply(dataclasses.replace(cfg, **OFF), "METRICA-S3", [])


def test_default_weights_files_are_the_recorded_ones():
    cfg = Config.default()
    assert submission.md5_of(learned.weights_path(cfg)) == NPZ_MD5
    assert submission.md5_of(os.path.join(ROOT, cfg.LEARNED_PF_WEIGHTS)) == PF_NPZ_MD5


def test_switch_off_record_c23_covers_sixteen_outputs():
    if not os.path.isfile(soi.RECORD_C23):
        pytest.skip("the public copy leaves out the records under docs/ (tools/export_public.py)")
    with open(soi.RECORD_C23) as handle:
        recorded = json.load(handle)
    assert sorted(recorded) == sorted("{}/{}".format(g, v) for g, v, _ in soi.jobs())
    # The raw outputs are those of LEARNED_STAGE_PROTOCOL_R3.md Section 8.
    assert recorded["METRICA-S1/raw"] == "61d4b60a5dcf9491d5998f4c5aeace61"
    assert recorded["DFL-J03WPY/raw"] == "6025967bd31a4f05f948713412e5905c"


def test_switch_off_check_runs_the_rule_engine_explicitly(monkeypatch, tmp_path):
    # --check must measure LEARNED_MODE off whatever the default (the learned candidate since
    # branch r6e-adopt); --record measures the default of the engine root it is given.
    seen = []

    def fake_measure(engine_root, cache, n_jobs, config=None):
        seen.append(None if config is None else json.load(open(config)))
        return {}

    monkeypatch.setattr(soi, "measure", fake_measure)
    record = tmp_path / "r.json"
    record.write_text("{}")
    assert soi.main(["--check", "--against", str(record)]) == 0
    assert soi.main(["--record", str(tmp_path / "out.json")]) == 0
    assert seen == [{"LEARNED_MODE": "off"}, None]


def test_switch_off_record_covers_sixteen_outputs():
    if not os.path.isfile(soi.RECORD):
        pytest.skip("the public copy leaves out the records under docs/ (tools/export_public.py)")
    with open(soi.RECORD) as handle:
        recorded = json.load(handle)
    assert len(recorded) == 16
    assert sorted(recorded) == sorted("{}/{}".format(g, v) for g, v, _ in soi.jobs())
    assert soi.compare(recorded, dict(recorded)) == []
    changed = dict(recorded, **{"METRICA-S3/raw": "0" * 32})
    assert soi.compare(recorded, changed) == ["METRICA-S3/raw"]
    assert soi.compare(recorded, {}) == sorted(recorded)


# -- (d) switch-off identity on the competition match ---------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present: the engine needs it")
def test_learned_mode_off_writes_c23_byte_for_byte(tmp_path):
    from engine import possession
    labels = possession.run(dataclasses.replace(Config.default(), **OFF))
    path = submission.write_submission(labels, str(tmp_path / "off.csv"))
    assert submission.md5_of(path) == C23_MD5


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present: the engine needs it")
def test_default_config_writes_the_learned_output_byte_for_byte(tmp_path):
    from engine import possession
    labels = possession.run(Config.default())
    path = submission.write_submission(labels, str(tmp_path / "default.csv"))
    assert submission.md5_of(path) == M_MD5


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present: the engine needs it")
def test_kickoff_team_off_writes_e2_byte_for_byte(tmp_path):
    from engine import possession
    labels = possession.run(dataclasses.replace(Config.default(), LEARNED_KICKOFF_TEAM=False))
    path = submission.write_submission(labels, str(tmp_path / "e2.csv"))
    assert submission.md5_of(path) == E2_MD5
